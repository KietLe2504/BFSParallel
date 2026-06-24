#include "graph.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

static uint64_t xorshift64(uint64_t *state)
{
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

/* double trong [0, 1) */
static inline double rand_double(uint64_t *state)
{
    return (double)(xorshift64(state) >> 11) / (double)(1ULL << 53);
}


static void rmat_edge(vertex_t n, const RMATParams *p,
                      uint64_t *state,
                      vertex_t *out_u, vertex_t *out_v)
{
    vertex_t u = 0, v = 0;
    vertex_t step = n;         

    while (step > 1) {
        step >>= 1;             
        double r = rand_double(state);
        if (r < p->a) {
        } else if (r < p->a + p->b) {
            v += step;         
        } else if (r < p->a + p->b + p->c) {
            u += step;        
        } else {
            u += step;        
            v += step;
        }
    }
    *out_u = u;
    *out_v = v;
}

typedef struct {
    vertex_t u, v;
} Edge;

static int edge_cmp(const void *a, const void *b)
{
    const Edge *ea = (const Edge *)a;
    const Edge *eb = (const Edge *)b;
    if (ea->u != eb->u) return (ea->u > eb->u) - (ea->u < eb->u);
    return (ea->v > eb->v) - (ea->v < eb->v);
}

Graph *graph_rmat_generate(vertex_t num_vertices,
                           int      scale_factor,
                           uint64_t seed,
                           const RMATParams *params)
{
 
    RMATParams default_params = {0.57, 0.19, 0.19, 0.05};
    if (!params) params = &default_params;

    vertex_t n = 1;
    while (n < num_vertices) n <<= 1;

    edge_t target_edges = (edge_t)n * scale_factor;

    fprintf(stderr, "[GRAPH] Generating R-MAT: n=%d, target_edges=%lld, seed=%llu\n",
            n, (long long)target_edges, (unsigned long long)seed);


    edge_t raw_cap = target_edges * 2 + 16;
    Edge  *edges   = (Edge *)malloc((size_t)raw_cap * sizeof(Edge));
    if (!edges) {
        fprintf(stderr, "[GRAPH] malloc failed for edge list\n");
        return NULL;
    }

    uint64_t state  = seed ^ 0xdeadbeefcafe1234ULL;
    edge_t   nedges = 0;

    for (edge_t i = 0; i < target_edges; i++) {
        vertex_t u, v;
        rmat_edge(n, params, &state, &u, &v);

        if (u == v) continue;

        edges[nedges].u = u;  edges[nedges].v = v;  nedges++;
        edges[nedges].u = v;  edges[nedges].v = u;  nedges++;

        if (nedges >= raw_cap - 4) {
            raw_cap *= 2;
            Edge *tmp = (Edge *)realloc(edges, (size_t)raw_cap * sizeof(Edge));
            if (!tmp) { free(edges); return NULL; }
            edges = tmp;
        }
    }

    qsort(edges, (size_t)nedges, sizeof(Edge), edge_cmp);

    edge_t unique = 0;
    for (edge_t i = 0; i < nedges; i++) {
        if (i == 0 ||
            edges[i].u != edges[i-1].u ||
            edges[i].v != edges[i-1].v)
        {
            edges[unique++] = edges[i];
        }
    }
    nedges = unique;

    fprintf(stderr, "[GRAPH] After dedup: %lld edges (avg degree %.2f)\n",
            (long long)nedges, (double)nedges / n);


    Graph *g = (Graph *)malloc(sizeof(Graph));
    if (!g) { free(edges); return NULL; }

    g->num_vertices = n;
    g->num_edges    = nedges;
    g->row_ptr = (edge_t   *)calloc((size_t)(n + 1), sizeof(edge_t));
    g->adj     = (vertex_t *)malloc((size_t)nedges  * sizeof(vertex_t));

    if (!g->row_ptr || !g->adj) {
        free(g->row_ptr); free(g->adj); free(g); free(edges);
        return NULL;
    }

    for (edge_t i = 0; i < nedges; i++)
        g->row_ptr[edges[i].u + 1]++;

    for (vertex_t v = 0; v < n; v++)
        g->row_ptr[v + 1] += g->row_ptr[v];

    edge_t *tmp_pos = (edge_t *)malloc((size_t)n * sizeof(edge_t));
    if (!tmp_pos) {
        free(g->row_ptr); free(g->adj); free(g); free(edges);
        return NULL;
    }
    memcpy(tmp_pos, g->row_ptr, (size_t)n * sizeof(edge_t));

    for (edge_t i = 0; i < nedges; i++) {
        vertex_t u = edges[i].u;
        g->adj[tmp_pos[u]++] = edges[i].v;
    }

    free(tmp_pos);
    free(edges);

    return g;
}

/* ------------------------------------------------------------------ */
void graph_free(Graph *g)
{
    if (!g) return;
    free(g->row_ptr);
    free(g->adj);
    free(g);
}

#define GRAPH_MAGIC  0x4246535f43535200ULL   /* "BFS_CSR\0" */

int graph_save(const Graph *g, const char *path)
{
    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "[GRAPH] Cannot open '%s' for writing: ", path);
        perror("");
        return -1;
    }

    uint64_t magic = GRAPH_MAGIC;
    int32_t  n     = (int32_t)g->num_vertices;
    int64_t  m     = (int64_t)g->num_edges;

    if (fwrite(&magic,       sizeof(uint64_t), 1,        f) != 1 ||
        fwrite(&n,           sizeof(int32_t),  1,        f) != 1 ||
        fwrite(&m,           sizeof(int64_t),  1,        f) != 1 ||
        fwrite(g->row_ptr,   sizeof(int64_t),  (size_t)(n + 1), f) != (size_t)(n + 1) ||
        fwrite(g->adj,       sizeof(int32_t),  (size_t)m,       f) != (size_t)m)
    {
        fprintf(stderr, "[GRAPH] Write error on '%s'\n", path);
        fclose(f);
        return -1;
    }

    fclose(f);
    fprintf(stderr, "[GRAPH] Saved graph to '%s' "
            "(n=%d, m=%lld, %.1f MB)\n",
            path, n, (long long)m,
            (double)(sizeof(int64_t)*(n+1) + sizeof(int32_t)*m) / (1024*1024));
    return 0;
}

Graph *graph_load(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[GRAPH] Cannot open '%s' for reading: ", path);
        perror("");
        return NULL;
    }

    uint64_t magic = 0;
    int32_t  n     = 0;
    int64_t  m     = 0;

    if (fread(&magic, sizeof(uint64_t), 1, f) != 1 || magic != GRAPH_MAGIC) {
        fprintf(stderr, "[GRAPH] '%s': bad magic — không phải file đồ thị hợp lệ\n", path);
        fclose(f); return NULL;
    }
    if (fread(&n, sizeof(int32_t), 1, f) != 1 || n <= 0) {
        fprintf(stderr, "[GRAPH] '%s': invalid num_vertices\n", path);
        fclose(f); return NULL;
    }
    if (fread(&m, sizeof(int64_t), 1, f) != 1 || m < 0) {
        fprintf(stderr, "[GRAPH] '%s': invalid num_edges\n", path);
        fclose(f); return NULL;
    }

    Graph *g = (Graph *)malloc(sizeof(Graph));
    if (!g) { fclose(f); return NULL; }

    g->num_vertices = (vertex_t)n;
    g->num_edges    = (edge_t)m;
    g->row_ptr = (edge_t   *)malloc((size_t)(n + 1) * sizeof(edge_t));
    g->adj     = (vertex_t *)malloc((size_t)m        * sizeof(vertex_t));

    if (!g->row_ptr || !g->adj) {
        fprintf(stderr, "[GRAPH] malloc failed when loading '%s'\n", path);
        free(g->row_ptr); free(g->adj); free(g); fclose(f); return NULL;
    }

    if (fread(g->row_ptr, sizeof(int64_t), (size_t)(n + 1), f) != (size_t)(n + 1) ||
        fread(g->adj,     sizeof(int32_t), (size_t)m,       f) != (size_t)m)
    {
        fprintf(stderr, "[GRAPH] Read error on '%s' — file bị cắt?\n", path);
        free(g->row_ptr); free(g->adj); free(g); fclose(f); return NULL;
    }

    fclose(f);
    fprintf(stderr, "[GRAPH] Loaded graph from '%s' "
            "(n=%d, m=%lld)\n", path, n, (long long)m);
    return g;
}

/* ------------------------------------------------------------------ */
void graph_print_info(const Graph *g)
{
    if (!g) return;
    edge_t max_deg = 0, min_deg = g->row_ptr[1] - g->row_ptr[0];
    for (vertex_t v = 0; v < g->num_vertices; v++) {
        edge_t d = graph_degree(g, v);
        if (d > max_deg) max_deg = d;
        if (d < min_deg) min_deg = d;
    }
    fprintf(stderr,
            "[GRAPH] Vertices: %d | Edges: %lld | "
            "Avg degree: %.2f | Min: %lld | Max: %lld\n",
            g->num_vertices,
            (long long)g->num_edges,
            (double)g->num_edges / g->num_vertices,
            (long long)min_deg,
            (long long)max_deg);
}