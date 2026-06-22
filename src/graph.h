#ifndef GRAPH_H
#define GRAPH_H

#include <stdint.h>
#include <stdlib.h>

/* ------------------------------------------------------------------ *
 * Kiểu dữ liệu cơ bản                                                *
 * ------------------------------------------------------------------ */
typedef int32_t  vertex_t;   /* ID đỉnh                               */
typedef int64_t  edge_t;     /* Chỉ số cạnh (có thể > 2^31)           */

/* ------------------------------------------------------------------ *
 * CSR (Compressed Sparse Row) – lưu đồ thị vô hướng                  *
 *                                                                      *
 * Với đỉnh v:                                                          *
 *   hàng xóm: adj[row_ptr[v] .. row_ptr[v+1]-1]                       *
 * ------------------------------------------------------------------ */
typedef struct {
    vertex_t  num_vertices;  /* |V|                                    */
    edge_t    num_edges;     /* |E| (mỗi cạnh tính 2 chiều)            */
    edge_t   *row_ptr;       /* kích thước num_vertices+1              */
    vertex_t *adj;           /* kích thước num_edges                   */
} Graph;

/* ------------------------------------------------------------------ *
 * Tham số R-MAT                                                        *
 *   a+b+c+d = 1.0, a >= b >= c >= d                                   *
 *   Mặc định Graph500: a=0.57, b=0.19, c=0.19, d=0.05                 *
 * ------------------------------------------------------------------ */
typedef struct {
    double a, b, c, d;
} RMATParams;

/* ------------------------------------------------------------------ *
 * API                                                                  *
 * ------------------------------------------------------------------ */

/*
 * Sinh đồ thị R-MAT vô hướng.
 *   num_vertices : số đỉnh (nên là lũy thừa 2, nhưng không bắt buộc)
 *   scale_factor : trung bình bậc mỗi đỉnh (~số cạnh = V * scale)
 *   seed         : random seed để tái hiện kết quả
 *   params       : NULL → dùng tham số Graph500 mặc định
 * Trả về Graph được cấp phát động; giải phóng bằng graph_free().
 */
Graph *graph_rmat_generate(vertex_t num_vertices,
                           int      scale_factor,
                           uint64_t seed,
                           const RMATParams *params);

/* Giải phóng bộ nhớ */
void graph_free(Graph *g);

/* In thông tin tóm tắt (stderr) */
void graph_print_info(const Graph *g);

/* Bậc của đỉnh v */
static inline edge_t graph_degree(const Graph *g, vertex_t v)
{
    return g->row_ptr[v + 1] - g->row_ptr[v];
}


/* ------------------------------------------------------------------ *
 * Save / Load graph sang file nhị phân                                *
 *                                                                      *
 * Format file:                                                         *
 *   [4 bytes] magic  0x42465347 ("BFSG")                              *
 *   [4 bytes] num_vertices                                             *
 *   [8 bytes] num_edges                                                *
 *   [(num_vertices+1) * 8 bytes] row_ptr[]                             *
 *   [num_edges * 4 bytes] adj[]                                        *
 * ------------------------------------------------------------------ */
int    graph_save(const Graph *g, const char *filename);
Graph *graph_load(const char *filename);

#endif /* GRAPH_H */