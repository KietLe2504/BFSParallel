#include "bfs_hybrid.h"
#include "utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <mpi.h>

#define WORD_BITS  64
#define BIT_WORD(v)  ((v) / WORD_BITS)
#define BIT_MASK(v)  (1ULL << ((v) % WORD_BITS))

typedef struct {
    uint64_t *bitmap;     
    vertex_t *dense;    
    vertex_t  size;       
    vertex_t  n;          
} Frontier;

static Frontier *frontier_create(vertex_t n)
{
    Frontier *f = (Frontier *)malloc(sizeof(Frontier));
    f->n      = n;
    f->size   = 0;
    size_t nw = ((size_t)n + WORD_BITS - 1) / WORD_BITS;
    f->bitmap = (uint64_t *)calloc(nw, sizeof(uint64_t));
    f->dense  = (vertex_t *)malloc((size_t)n * sizeof(vertex_t));
    return f;
}

static void frontier_free(Frontier *f)
{
    if (!f) return;
    free(f->bitmap);
    free(f->dense);
    free(f);
}

static void frontier_clear(Frontier *f)
{
    size_t nw = ((size_t)f->n + WORD_BITS - 1) / WORD_BITS;
    memset(f->bitmap, 0, nw * sizeof(uint64_t));
    f->size = 0;
}

static inline int frontier_has(const Frontier *f, vertex_t v)
{
    return (f->bitmap[BIT_WORD(v)] & BIT_MASK(v)) != 0;
}

static inline void frontier_add_atomic(Frontier *f, vertex_t v)
{
    uint64_t mask = BIT_MASK(v);
    uint64_t word = BIT_WORD(v);
    #pragma omp atomic
    f->bitmap[word] |= mask;
}

static void frontier_build_dense(Frontier *f)
{
    f->size = 0;
    size_t nw = ((size_t)f->n + WORD_BITS - 1) / WORD_BITS;
    for (size_t w = 0; w < nw; w++) {
        uint64_t word = f->bitmap[w];
        while (word) {
            int bit = __builtin_ctzll(word);
            f->dense[f->size++] = (vertex_t)(w * WORD_BITS + bit);
            word &= word - 1;  
        }
    }
}

static void top_down_step(const Graph *g,
                          const Frontier *curr,
                          Frontier       *next,
                          int            *dist,
                          int             level,
                          int             my_rank,
                          int             num_ranks,
                          edge_t         *out_frontier_edges)
{
    frontier_clear(next);
    edge_t frontier_edges = 0;

    /* Chia frontier cho các rank */
    vertex_t chunk    = (curr->size + num_ranks - 1) / num_ranks;
    vertex_t rank_s   = my_rank * chunk;
    vertex_t rank_e   = rank_s + chunk;
    if (rank_e > curr->size) rank_e = curr->size;

    #pragma omp parallel reduction(+:frontier_edges)
    {
        #pragma omp for schedule(dynamic, 64) nowait
        for (vertex_t fi = rank_s; fi < rank_e; fi++) {
            vertex_t u = curr->dense[fi];
            edge_t   start = g->row_ptr[u];
            edge_t   end   = g->row_ptr[u + 1];

            for (edge_t e = start; e < end; e++) {
                frontier_edges++;
                vertex_t v = g->adj[e];

                /* CAS để tránh race condition giữa các thread */
                if (dist[v] == -1) {
                    int old = __sync_val_compare_and_swap(&dist[v], -1, level);
                    if (old == -1) {
                        frontier_add_atomic(next, v);
                    }
                }
            }
        }
    }

    *out_frontier_edges = frontier_edges;
}

static void bottom_up_step(const Graph    *g,
                           const Frontier *curr,
                           Frontier       *next,
                           int            *dist,
                           int             level,
                           vertex_t        local_start,
                           vertex_t        local_end)
{
    frontier_clear(next);

    #pragma omp parallel for schedule(dynamic, 256)
    for (vertex_t u = local_start; u < local_end; u++) {
        if (dist[u] != -1) continue;  

        edge_t start = g->row_ptr[u];
        edge_t end   = g->row_ptr[u + 1];

        for (edge_t e = start; e < end; e++) {
            vertex_t v = g->adj[e];
            if (frontier_has(curr, v)) {
                dist[u] = level;
                frontier_add_atomic(next, u);
                break;  
            }
        }
    }
}

static double sync_dist(int *dist, vertex_t n)
{
    double t0 = MPI_Wtime();
    MPI_Allreduce(MPI_IN_PLACE, dist, (int)n, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
    return MPI_Wtime() - t0;
}

static double sync_frontier(Frontier *f)
{
    size_t nw = ((size_t)f->n + WORD_BITS - 1) / WORD_BITS;
    double t0 = MPI_Wtime();
    MPI_Allreduce(MPI_IN_PLACE, f->bitmap, (int)nw, MPI_UINT64_T,
                  MPI_BOR, MPI_COMM_WORLD);
    double comm_dt = MPI_Wtime() - t0;
    frontier_build_dense(f);
    return comm_dt;
}

BFSResult bfs_hybrid(const Graph *g, vertex_t source)
{
    int my_rank, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    vertex_t n = g->num_vertices;

    
    vertex_t local_start = partition_start(n, num_ranks, my_rank);
    vertex_t local_end   = partition_end  (n, num_ranks, my_rank);

    int *dist = (int *)malloc((size_t)n * sizeof(int));
    if (!dist) {
        fprintf(stderr, "[HYBRID] rank %d: malloc dist failed\n", my_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    memset(dist, -1, (size_t)n * sizeof(int));
    dist[source] = 0;

    Frontier *curr = frontier_create(n);
    Frontier *next = frontier_create(n);

    curr->bitmap[BIT_WORD(source)] |= BIT_MASK(source);
    curr->dense[0] = source;
    curr->size     = 1;

    edge_t unvisited_edges = g->num_edges;   
    edge_t total_visited   = 0;

    int level = 1;
    int use_bottom_up = 0;   
    int num_levels    = 0;

    Timer wall;
    if (my_rank == 0) timer_start(&wall);

    double rank_compute_s = 0.0;
    double rank_comm_s    = 0.0;

    while (curr->size > 0) {
        num_levels++;

        edge_t frontier_edges = 0;

        double t_compute0 = MPI_Wtime();
        edge_t fe_local = 0;
        #pragma omp parallel for reduction(+:fe_local)
        for (vertex_t fi = 0; fi < curr->size; fi++)
            fe_local += graph_degree(g, curr->dense[fi]);
        rank_compute_s += MPI_Wtime() - t_compute0;

        double t_comm0 = MPI_Wtime();
        MPI_Allreduce(&fe_local, &frontier_edges, 1,
                      MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
        rank_comm_s += MPI_Wtime() - t_comm0;

        int should_bottom_up = 0;
        if (!use_bottom_up) {
            should_bottom_up =
                (frontier_edges > unvisited_edges / BFS_ALPHA);
        } else {
            should_bottom_up =
                (curr->size >= (vertex_t)(n / BFS_BETA));
        }

        if (my_rank == 0) {
            const char *dir = should_bottom_up ? "bottom-up" : "top-down ";
            printf("[HYBRID]   Level %2d: %s (frontier=%d, fe=%lld, ue=%lld)\n",
                   level, dir,
                   (int)curr->size,
                   (long long)frontier_edges,
                   (long long)unvisited_edges);
            fflush(stdout);
        }

        use_bottom_up = should_bottom_up;

        if (!use_bottom_up) {
            double t0 = MPI_Wtime();
            edge_t step_fe = 0;
            top_down_step(g, curr, next, dist, level,
                          my_rank, num_ranks, &step_fe);
            rank_compute_s += MPI_Wtime() - t0;

            rank_comm_s += sync_dist(dist, n);
            rank_comm_s += sync_frontier(next);

            total_visited += frontier_edges;
            unvisited_edges -= frontier_edges;

        } else {
            double t0 = MPI_Wtime();
            bottom_up_step(g, curr, next, dist, level,
                           local_start, local_end);
            rank_compute_s += MPI_Wtime() - t0;

            rank_comm_s += sync_dist(dist, n);
            rank_comm_s += sync_frontier(next);

            double t1 = MPI_Wtime();
            edge_t next_fe = 0;
            #pragma omp parallel for reduction(+:next_fe)
            for (vertex_t fi = 0; fi < next->size; fi++)
                next_fe += graph_degree(g, next->dense[fi]);
            rank_compute_s += MPI_Wtime() - t1;

            unvisited_edges -= next_fe;
            total_visited   += next_fe;
        }

        Frontier *tmp = curr;
        curr = next;
        next = tmp;

        level++;
    }

    double elapsed = 0.0;
    if (my_rank == 0) elapsed = timer_elapsed_ms(&wall);

    frontier_free(curr);
    frontier_free(next);

    edge_t local_edges = g->row_ptr[local_end] - g->row_ptr[local_start];

    RankStats my_stats;
    my_stats.rank        = my_rank;
    my_stats.compute_ms  = rank_compute_s * 1000.0;
    my_stats.comm_ms     = rank_comm_s    * 1000.0;
    my_stats.total_ms    = my_stats.compute_ms + my_stats.comm_ms;
    my_stats.local_count = local_end - local_start;
    my_stats.local_edges = local_edges;

    RankStats *all_stats = NULL;
    if (my_rank == 0) {
        all_stats = (RankStats *)malloc((size_t)num_ranks * sizeof(RankStats));
    }
    MPI_Gather(&my_stats, sizeof(RankStats), MPI_BYTE,
               all_stats, sizeof(RankStats), MPI_BYTE,
               0, MPI_COMM_WORLD);

    BFSResult result;
    result.num_levels    = num_levels;
    result.visited_edges = total_visited;
    result.time_ms       = elapsed;
    result.compute_ms    = my_stats.compute_ms;  
    result.comm_ms       = my_stats.comm_ms;
    result.rank_stats    = all_stats;             
    result.num_ranks     = num_ranks;

    if (my_rank == 0) {
        result.dist = dist;
    } else {
        free(dist);
        result.dist = NULL;
    }

    return result;
}

void bfs_result_free(BFSResult *r)
{
    if (!r) return;
    if (r->dist) {
        free(r->dist);
        r->dist = NULL;
    }
    if (r->rank_stats) {
        free(r->rank_stats);
        r->rank_stats = NULL;
    }
}