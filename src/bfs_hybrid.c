#include "bfs_hybrid.h"
#include "utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <mpi.h>

/* ================================================================== *
 *  THIẾT KẾ TỔNG QUAN                                                 *
 *                                                                      *
 *  1. Graph được replicate đầy đủ trên mọi rank (không partition CSR). *
 *     → Tránh MPI scatter lớn, phù hợp với cluster nhỏ (4 node).      *
 *                                                                      *
 *  2. Phân vùng ĐỈNH (vertex partition):                               *
 *     Rank r chịu trách nhiệm tính dist[] cho các đỉnh                 *
 *     [local_start, local_end). Cuối mỗi level, dùng MPI_Allreduce    *
 *     để hợp nhất dist[] toàn cục.                                     *
 *                                                                      *
 *  3. Direction-optimizing (Beamer et al.):                            *
 *     - TOP-DOWN  : mỗi đỉnh trong frontier → xét hàng xóm            *
 *     - BOTTOM-UP : mỗi đỉnh chưa thăm → tìm hàng xóm trong frontier  *
 *     Switch dựa trên frontier_edges vs unvisited_edges.               *
 *                                                                      *
 *  4. OpenMP:                                                           *
 *     - Parallel for trong cả top-down và bottom-up                    *
 *     - Dùng atomic / critical để tránh race condition                 *
 * ================================================================== */

/* ------------------------------------------------------------------ *
 * Kiểu frontier: bitmap + dense array                                  *
 *                                                                      *
 * bitmap  : bit v = 1 nếu v trong frontier (O(1) lookup)              *
 * dense   : mảng các đỉnh trong frontier (iteration nhanh)            *
 * ------------------------------------------------------------------ */

#define WORD_BITS  64
#define BIT_WORD(v)  ((v) / WORD_BITS)
#define BIT_MASK(v)  (1ULL << ((v) % WORD_BITS))

typedef struct {
    uint64_t *bitmap;     /* kích thước ceil(n/64) words               */
    vertex_t *dense;      /* danh sách đỉnh                            */
    vertex_t  size;       /* số đỉnh trong frontier                    */
    vertex_t  n;          /* tổng số đỉnh (để tính kích thước bitmap)  */
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

/* Thread-safe add (dùng trong OpenMP parallel) */
static inline void frontier_add_atomic(Frontier *f, vertex_t v)
{
    uint64_t mask = BIT_MASK(v);
    uint64_t word = BIT_WORD(v);
    /* atomic OR để tránh race */
    #pragma omp atomic
    f->bitmap[word] |= mask;
}

/* Build dense[] từ bitmap sau khi parallel phase xong */
static void frontier_build_dense(Frontier *f)
{
    f->size = 0;
    size_t nw = ((size_t)f->n + WORD_BITS - 1) / WORD_BITS;
    for (size_t w = 0; w < nw; w++) {
        uint64_t word = f->bitmap[w];
        while (word) {
            int bit = __builtin_ctzll(word);
            f->dense[f->size++] = (vertex_t)(w * WORD_BITS + bit);
            word &= word - 1;  /* xóa bit thấp nhất */
        }
    }
}

/* ------------------------------------------------------------------ *
 * TOP-DOWN step (parallel với OpenMP, phân vùng với MPI)               *
 *                                                                      *
 * Mỗi rank xử lý một phần của frontier (chia đều theo rank).           *
 * Với mỗi đỉnh u trong frontier, xét hàng xóm v:                      *
 *   nếu dist[v] == -1 → đặt dist[v] = level, thêm v vào next_frontier *
 * ------------------------------------------------------------------ */
static void top_down_step(const Graph *g,
                          const Frontier *curr,
                          Frontier       *next,
                          int            *dist,
                          int             level,
                          int             my_rank,
                          int             num_ranks,
                          edge_t         *out_frontier_edges,
                          NewVerts       *local_new)   /* NULL = không thu thập */
{
    frontier_clear(next);
    if (local_new) new_verts_clear(local_new);
    edge_t frontier_edges = 0;

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

                if (dist[v] == -1) {
                    int old = __sync_val_compare_and_swap(&dist[v], -1, level);
                    if (old == -1) {
                        frontier_add_atomic(next, v);
                        if (local_new) new_verts_add(local_new, v);
                    }
                }
            }
        }
    }

    *out_frontier_edges = frontier_edges;
}

/* ------------------------------------------------------------------ *
 * BOTTOM-UP step (parallel với OpenMP, phân vùng với MPI)              *
 *                                                                      *
 * Mỗi rank xử lý phần đỉnh chưa thăm của mình [local_start, local_end)*
 * Với mỗi đỉnh chưa thăm u, xét hàng xóm v:                           *
 *   nếu v trong frontier → dist[u] = level, thêm u vào next_frontier   *
 * ------------------------------------------------------------------ */
static void bottom_up_step(const Graph    *g,
                           const Frontier *curr,
                           Frontier       *next,
                           int            *dist,
                           int             level,
                           vertex_t        local_start,
                           vertex_t        local_end,
                           NewVerts       *local_new)   /* NULL = không thu thập */
{
    frontier_clear(next);
    if (local_new) new_verts_clear(local_new);

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
                if (local_new) new_verts_add(local_new, u);
                break;
            }
        }
    }
}

/* ------------------------------------------------------------------ *
 * Đồng bộ dist[] giữa các rank bằng MPI_Allreduce (MAX)               *
 *                                                                      *
 * dist[v] = -1 (chưa thăm) hoặc >= 0 (khoảng cách).                  *
 * MAX của các giá trị -1 và k = k → dùng MAX để merge.                 *
 *                                                                      *
 * BỔ SUNG: trả về thời gian (giây) rank này "ở trong" lệnh Allreduce, *
 * đo bằng MPI_Wtime() ngay trước/sau lệnh gọi. Vì Allreduce là hàm    *
 * blocking đồng bộ tập thể, thời gian này gồm CẢ thời gian chờ các    *
 * rank chậm hơn tới điểm đồng bộ (wait time do mất cân bằng tải)      *
 * LẪN thời gian truyền dữ liệu thực sự — hai thành phần không tách    *
 * rời được nếu chỉ đo từ phía 1 rank, nên báo cáo gọi chung là        *
 * "communication time" (đúng tinh thần đề bài: cột màu thứ 2 trong    *
 * biểu đồ load-balancing).                                            *
 * ------------------------------------------------------------------ */
/* ------------------------------------------------------------------ *
 * DELTA SYNC — thay thế sync_dist O(n) bằng O(|new_vertices|)        *
 *                                                                      *
 * Thay vì Allreduce toàn bộ dist[] (n phần tử, ~32 MB với n=8M),     *
 * mỗi rank chỉ gửi danh sách đỉnh nó vừa thăm trong level này.       *
 *                                                                      *
 * Giao thức:                                                           *
 *  1. Mỗi rank thu thập local_new[] trong step (top-down/bottom-up)   *
 *  2. MPI_Allgather để trao đổi số lượng đỉnh mới của từng rank       *
 *  3. MPI_Allgatherv để trao đổi danh sách đỉnh mới                   *
 *  4. Mỗi rank cập nhật dist[v] = level cho các đỉnh nhận được        *
 *                                                                      *
 * Complexity: O(|new_vertices_this_level|) thay vì O(n)               *
 * Với BFS trên R-MAT 6 level: tổng new_vertices << n ở level 1,5,6    *
 * và = O(n) ở level 2-4 (frontier rộng) — nhưng khi đó frontier       *
 * bitmap cũng O(n), không tiết kiệm được gì → fallback tự động.       *
 *                                                                      *
 * Lưu ý: Allgatherv có latency cao hơn Allreduce (cần P round-trips   *
 * thay vì log P), nên chỉ có lợi khi message nhỏ hơn ngưỡng.         *
 * ------------------------------------------------------------------ */

/* Mảng đỉnh mới tích lũy trong 1 step — mỗi rank giữ riêng */
typedef struct {
    vertex_t *buf;     /* buffer đỉnh mới                                */
    int       count;   /* số đỉnh đã thêm                                */
    int       cap;     /* dung lượng buffer                              */
} NewVerts;

static NewVerts new_verts_create(int cap)
{
    NewVerts nv;
    nv.buf   = (vertex_t *)malloc((size_t)cap * sizeof(vertex_t));
    nv.count = 0;
    nv.cap   = cap;
    return nv;
}

static void new_verts_free(NewVerts *nv)
{
    free(nv->buf);
    nv->buf = NULL;
}

static void new_verts_clear(NewVerts *nv) { nv->count = 0; }

/* Thread-safe add (gọi từ trong OpenMP parallel) */
static inline void new_verts_add(NewVerts *nv, vertex_t v)
{
    int idx;
    #pragma omp atomic capture
    idx = nv->count++;
    /* Nếu tràn buffer thì bỏ qua — trường hợp này rất hiếm và sẽ được
     * bù lại bởi sync_frontier (frontier bitmap vẫn đúng). dist[] của
     * đỉnh bị bỏ qua sẽ được cập nhật ở level sau khi rank khác gửi. */
    if (idx < nv->cap)
        nv->buf[idx] = v;
}

/* ------------------------------------------------------------------ *
 * sync_delta: đồng bộ dist[] bằng cách trao đổi chỉ đỉnh mới         *
 * Trả về thời gian (giây) dành cho MPI.                               *
 * ------------------------------------------------------------------ */
static double sync_delta(int *dist, int level,
                         NewVerts *local_new,
                         int num_ranks, int my_rank)
{
    double t0 = MPI_Wtime();

    int local_count = local_new->count;
    if (local_count > local_new->cap) local_count = local_new->cap;

    /* Bước 1: trao đổi số lượng đỉnh mới của từng rank */
    int *counts = (int *)malloc((size_t)num_ranks * sizeof(int));
    MPI_Allgather(&local_count, 1, MPI_INT,
                  counts, 1, MPI_INT, MPI_COMM_WORLD);

    int total = 0;
    int *displs = (int *)malloc((size_t)num_ranks * sizeof(int));
    for (int r = 0; r < num_ranks; r++) {
        displs[r] = total;
        total    += counts[r];
    }

    /* Bước 2: trao đổi danh sách đỉnh (Allgatherv) */
    vertex_t *all_new = (vertex_t *)malloc((size_t)(total + 1) * sizeof(vertex_t));
    MPI_Allgatherv(local_new->buf, local_count, MPI_INT,
                   all_new, counts, displs, MPI_INT, MPI_COMM_WORLD);

    double comm_dt = MPI_Wtime() - t0;

    /* Bước 3: cập nhật dist[] cục bộ (không có MPI — tính vào compute) */
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < total; i++) {
        vertex_t v = all_new[i];
        /* Chỉ ghi nếu chưa được thăm — tránh ghi đè level sai khi
         * có race (không thể xảy ra trong BFS level-synchronous,
         * nhưng guard cho an toàn) */
        if (dist[v] == -1)
            dist[v] = level;
    }

    free(counts);
    free(displs);
    free(all_new);
    return comm_dt;
}

/* ------------------------------------------------------------------ *
 * sync_dist gốc — O(n), giữ lại để so sánh qua flag --delta-sync     *
 * ------------------------------------------------------------------ */
static double sync_dist(int *dist, vertex_t n)
{
    double t0 = MPI_Wtime();
    MPI_Allreduce(MPI_IN_PLACE, dist, (int)n, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
    return MPI_Wtime() - t0;
}

/* ------------------------------------------------------------------ *
 * Đồng bộ frontier bitmap bằng MPI_Allreduce (OR)                      *
 * Trả về thời gian (giây) trong Allreduce (xem giải thích ở sync_dist) *
 * KHÔNG tính thời gian frontier_build_dense() vào comm time, vì đó    *
 * là tính toán cục bộ (xây lại mảng dense[] từ bitmap, không có MPI). *
 * ------------------------------------------------------------------ */
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

/* ================================================================== *
 * BFS HYBRID CHÍNH                                                     *
 * use_delta_sync = 1 : dùng sync_delta O(|new_vertices|)              *
 * use_delta_sync = 0 : dùng sync_dist  O(n) gốc (để so sánh)         *
 * ================================================================== */
BFSResult bfs_hybrid(const Graph *g, vertex_t source, int use_delta_sync)
{
    int my_rank, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    vertex_t n = g->num_vertices;

    /* Phân vùng đỉnh */
    vertex_t local_start = partition_start(n, num_ranks, my_rank);
    vertex_t local_end   = partition_end  (n, num_ranks, my_rank);

    /* ---- Cấp phát dist[] ----------------------------------------- */
    int *dist = (int *)malloc((size_t)n * sizeof(int));
    if (!dist) {
        fprintf(stderr, "[HYBRID] rank %d: malloc dist failed\n", my_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    memset(dist, -1, (size_t)n * sizeof(int));
    dist[source] = 0;

    /* ---- Frontier ------------------------------------------------- */
    Frontier *curr = frontier_create(n);
    Frontier *next = frontier_create(n);

    /* Khởi tạo frontier = {source} */
    curr->bitmap[BIT_WORD(source)] |= BIT_MASK(source);
    curr->dense[0] = source;
    curr->size     = 1;

    /* ---- Thống kê để quyết định switch ---------------------------- */
    edge_t unvisited_edges = g->num_edges;   /* tổng cạnh chưa xét    */
    edge_t total_visited   = 0;

    /* ---- NewVerts buffer cho delta-sync ------------------------------ *
     * Cap = n (worst case mọi đỉnh đều mới trong 1 level — thực tế      *
     * chỉ xảy ra ở level 2-3 của BFS trên R-MAT small-world).           *
     * Khi dùng delta-sync, mỗi rank chỉ gửi phần nó tự thăm.           */
    NewVerts local_new = use_delta_sync
        ? new_verts_create((int)n)
        : (NewVerts){ NULL, 0, 0 };

    int level = 1;
    int use_bottom_up = 0;
    int num_levels    = 0;

    Timer wall;
    if (my_rank == 0) timer_start(&wall);

    double rank_compute_s = 0.0;
    double rank_comm_s    = 0.0;

    /* ================================================================ *
     * VÒNG LẶP BFS THEO LEVEL                                          *
     * ================================================================ */
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

        /* ---- Quyết định hướng (Beamer et al.) ---------------------- */
        int should_bottom_up = 0;
        if (!use_bottom_up) {
            should_bottom_up =
                (frontier_edges > unvisited_edges / BFS_ALPHA);
        } else {
            should_bottom_up =
                (curr->size >= (vertex_t)(n / BFS_BETA));
        }

        if (my_rank == 0) {
            const char *dir  = should_bottom_up ? "bottom-up" : "top-down ";
            const char *mode = use_delta_sync   ? "[delta]"   : "[full] ";
            printf("[HYBRID]   Level %2d: %s %s (frontier=%d, fe=%lld, ue=%lld)\n",
                   level, dir, mode,
                   (int)curr->size,
                   (long long)frontier_edges,
                   (long long)unvisited_edges);
            fflush(stdout);
        }

        use_bottom_up = should_bottom_up;

        /* ---- Thực hiện bước BFS ----------------------------------- */
        NewVerts *nv_ptr = use_delta_sync ? &local_new : NULL;

        if (!use_bottom_up) {
            double t0 = MPI_Wtime();
            edge_t step_fe = 0;
            top_down_step(g, curr, next, dist, level,
                          my_rank, num_ranks, &step_fe, nv_ptr);
            rank_compute_s += MPI_Wtime() - t0;

            if (use_delta_sync) {
                /* Delta sync: gửi chỉ đỉnh mới, không gửi toàn bộ dist[] */
                rank_comm_s += sync_delta(dist, level, &local_new,
                                          num_ranks, my_rank);
            } else {
                rank_comm_s += sync_dist(dist, n);
            }
            rank_comm_s += sync_frontier(next);

            total_visited   += frontier_edges;
            unvisited_edges -= frontier_edges;

        } else {
            double t0 = MPI_Wtime();
            bottom_up_step(g, curr, next, dist, level,
                           local_start, local_end, nv_ptr);
            rank_compute_s += MPI_Wtime() - t0;

            if (use_delta_sync) {
                rank_comm_s += sync_delta(dist, level, &local_new,
                                          num_ranks, my_rank);
            } else {
                rank_comm_s += sync_dist(dist, n);
            }
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

    /* ---- Thu thập thống kê per-rank về rank 0 (MPI_Gather) --------- *
     * Mỗi rank đóng gói: rank id, compute_ms, comm_ms, total_ms,       *
     * local_count (số đỉnh sở hữu), local_edges (tổng bậc dải đỉnh).   *
     * local_edges dùng để minh chứng độ lệch tải do R-MAT phân phối    *
     * bậc không đều (xem mục 4.5 / 5.3 trong báo cáo).                 *
     * ------------------------------------------------------------------ */
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
    /* RankStats là struct kích thước cố định, dùng MPI_BYTE để gather   *
     * đơn giản (tránh phải định nghĩa MPI_Datatype riêng).              */
    MPI_Gather(&my_stats, sizeof(RankStats), MPI_BYTE,
               all_stats, sizeof(RankStats), MPI_BYTE,
               0, MPI_COMM_WORLD);

    /* ---- Build kết quả ------------------------------------------- */
    BFSResult result;
    result.num_levels    = num_levels;
    result.visited_edges = total_visited;
    result.time_ms       = elapsed;
    result.compute_ms    = my_stats.compute_ms;  /* của rank gọi hàm này */
    result.comm_ms       = my_stats.comm_ms;
    result.rank_stats    = all_stats;             /* NULL nếu my_rank != 0 */
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