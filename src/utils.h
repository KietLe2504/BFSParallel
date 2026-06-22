#ifndef UTILS_H
#define UTILS_H

#include <stdint.h>
#include <time.h>
#include "graph.h"

/* ------------------------------------------------------------------ *
 * Timer (wall-clock, microsecond precision)                            *
 * ------------------------------------------------------------------ */
typedef struct {
    struct timespec _start;
} Timer;

void   timer_start(Timer *t);
double timer_elapsed_ms(const Timer *t);   /* milliseconds */

/* ------------------------------------------------------------------ *
 * Metrics                                                              *
 * ------------------------------------------------------------------ */

/*
 * Tính MTEPS (Mega Traversed Edges Per Second).
 *   visited_edges : tổng số cạnh đã duyệt (thường = |E| nếu BFS thăm hết)
 *   elapsed_ms    : thời gian chạy tính bằng milliseconds
 */
double compute_mteps(edge_t visited_edges, double elapsed_ms);

/*
 * Đếm số đỉnh được thăm trong mảng dist[].
 *   dist[v] >= 0 → đã thăm
 *   dist[v] == -1 → chưa thăm
 */
vertex_t count_visited(const int *dist, vertex_t num_vertices);

/*
 * Xác minh kết quả BFS: so sánh dist_ref[] (sequential) với dist_test[].
 * Trả về 0 nếu khớp, số lượng đỉnh sai nếu không khớp.
 */
int verify_bfs(const int *dist_ref, const int *dist_test,
               vertex_t num_vertices);

/* ------------------------------------------------------------------ *
 * Misc                                                                 *
 * ------------------------------------------------------------------ */

/* Chọn đỉnh nguồn hợp lệ (bậc > 0) */
vertex_t pick_source(const Graph *g, uint64_t seed);

/* In thanh tiến trình đơn giản (dùng khi generate graph lớn) */
void print_progress(long long current, long long total, const char *label);

#endif /* UTILS_H */
