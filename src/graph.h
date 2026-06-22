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
 * Lưu / Load đồ thị dạng binary CSR                                   *
 *                                                                      *
 * Format file (.bin):                                                  *
 *   [8 bytes] magic   = 0x4246535f43535200  ("BFS_CSR\0")             *
 *   [4 bytes] n       = num_vertices  (int32_t)                        *
 *   [8 bytes] m       = num_edges     (int64_t)                        *
 *   [8*(n+1) bytes]   row_ptr[]       (int64_t × n+1)                  *
 *   [4*m bytes]       adj[]           (int32_t × m)                    *
 *                                                                      *
 * graph_save() : trả về 0 nếu thành công, -1 nếu lỗi.                 *
 * graph_load() : trả về Graph* mới, NULL nếu lỗi (file sai / corrupt). *
 * ------------------------------------------------------------------ */
int    graph_save(const Graph *g, const char *path);
Graph *graph_load(const char *path);

#endif /* GRAPH_H */