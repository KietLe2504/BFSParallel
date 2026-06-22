#ifndef BFS_HYBRID_H
#define BFS_HYBRID_H

#include "graph.h"
#include <mpi.h>

/* ------------------------------------------------------------------ *
 * Tham số direction-optimizing (Beamer et al. 2012)                    *
 *                                                                      *
 * Chuyển sang bottom-up khi:                                           *
 *   frontier_edges > unvisited_edges / ALPHA                           *
 *   (frontier quá lớn → top-down lãng phí)                            *
 *                                                                      *
 * Quay lại top-down khi:                                               *
 *   frontier_size < num_vertices / BETA                                *
 *   (frontier đã nhỏ trở lại)                                          *
 * ------------------------------------------------------------------ */
#define BFS_ALPHA  14
#define BFS_BETA   24

/* ------------------------------------------------------------------ *
 * Phân vùng đỉnh giữa các MPI rank                                     *
 *                                                                      *
 * Rank r quản lý các đỉnh: [part_start(r), part_end(r))               *
 * ------------------------------------------------------------------ */
typedef struct {
    vertex_t total_vertices;  /* |V| toàn cục                          */
    int      num_ranks;       /* số MPI process                        */
    int      my_rank;         /* rank hiện tại                         */
    vertex_t local_start;     /* đỉnh đầu tiên của rank này            */
    vertex_t local_end;       /* đỉnh cuối + 1                         */
    vertex_t local_count;     /* = local_end - local_start             */
} Partition;

/* Tính phân vùng đều cho rank r trong num_ranks */
static inline vertex_t partition_start(vertex_t n, int num_ranks, int rank)
{
    return (vertex_t)((long long)n * rank / num_ranks);
}

static inline vertex_t partition_end(vertex_t n, int num_ranks, int rank)
{
    return (vertex_t)((long long)n * (rank + 1) / num_ranks);
}

/* ------------------------------------------------------------------ *
 * Thống kê thời gian theo từng MPI rank                                *
 *                                                                      *
 * compute_ms : tổng thời gian rank này thực sự tính toán               *
 *              (top_down_step / bottom_up_step + đếm fe_local),        *
 *              cộng dồn qua mọi level.                                 *
 * comm_ms    : tổng thời gian rank này đứng "trong" các lệnh           *
 *              MPI_Allreduce (gồm cả thời gian chờ rank chậm nhất       *
 *              tới điểm đồng bộ — đây chính là phần thời gian rảnh/     *
 *              chờ theo nghĩa load-imbalance), cộng dồn qua mọi level.  *
 * total_ms   : compute_ms + comm_ms (xấp xỉ thời gian sống của rank    *
 *              trong toàn bộ vòng lặp BFS).                            *
 * ------------------------------------------------------------------ */
typedef struct {
    int      rank;
    double   compute_ms;
    double   comm_ms;
    double   total_ms;
    vertex_t local_count;    /* số đỉnh sở hữu (local_end - local_start) */
    edge_t   local_edges;    /* tổng bậc (số cạnh) của dải đỉnh sở hữu   */
} RankStats;

/* ------------------------------------------------------------------ *
 * Kết quả BFS                                                          *
 * ------------------------------------------------------------------ */
typedef struct {
    int     *dist;           /* dist[v] toàn cục (chỉ hợp lệ ở rank 0) */
    int      num_levels;     /* số mức BFS                              */
    edge_t   visited_edges;  /* tổng số cạnh duyệt qua                 */
    double   time_ms;        /* thời gian tổng (wall-clock của rank 0)  */

    /* ---- Bổ sung: phân tách compute / communication --------------- *
     * Đo trên rank 0 (rank gọi bfs_hybrid và giữ kết quả).             *
     * compute_ms + comm_ms xấp xỉ time_ms (có thể lệch nhỏ do phần     *
     * code ngoài 2 vùng đo, ví dụ overhead vòng lặp, swap con trỏ...). */
    double   compute_ms;     /* tổng thời gian tính toán của rank 0     */
    double   comm_ms;        /* tổng thời gian trong Allreduce của rank 0 */

    /* ---- Bổ sung: thống kê toàn bộ rank (chỉ hợp lệ ở rank 0) ------ *
     * Mảng kích thước num_ranks, lấy được qua MPI_Gather.              *
     * NULL ở các rank != 0.                                            */
    RankStats *rank_stats;
    int        num_ranks;
} BFSResult;

/* ------------------------------------------------------------------ *
 * API chính                                                             *
 * ------------------------------------------------------------------ */

/*
 * Chạy BFS hybrid trên đồ thị g từ đỉnh source.
 *
 * Tất cả MPI rank đều gọi hàm này.
 * g phải giống nhau trên tất cả rank (replicated graph).
 *
 * Kết quả:
 *   - result->dist chỉ được ghi đầy đủ ở rank 0
 *   - Các rank khác: result->dist = NULL
 *
 * Gọi bfs_result_free() để giải phóng.
 */
BFSResult bfs_hybrid(const Graph *g, vertex_t source);

/* Giải phóng bộ nhớ trong BFSResult (gồm cả dist[] và rank_stats[]) */
void bfs_result_free(BFSResult *r);

#endif /* BFS_HYBRID_H */