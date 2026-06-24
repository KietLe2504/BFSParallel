#include "utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void timer_start(Timer *t)
{
    clock_gettime(CLOCK_MONOTONIC, &t->_start);
}

double timer_elapsed_ms(const Timer *t)
{
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    double sec  = (double)(now.tv_sec  - t->_start.tv_sec);
    double nsec = (double)(now.tv_nsec - t->_start.tv_nsec);
    return sec * 1000.0 + nsec / 1e6;
}

double compute_mteps(edge_t visited_edges, double elapsed_ms)
{
    if (elapsed_ms <= 0.0) return 0.0;
    /* MTEPS = edges / (time_s * 1e6) */
    return (double)visited_edges / (elapsed_ms / 1000.0) / 1e6;
}

vertex_t count_visited(const int *dist, vertex_t num_vertices)
{
    vertex_t cnt = 0;
    for (vertex_t v = 0; v < num_vertices; v++)
        if (dist[v] >= 0) cnt++;
    return cnt;
}

int verify_bfs(const int *dist_ref, const int *dist_test,
               vertex_t num_vertices)
{
    int errors = 0;
    for (vertex_t v = 0; v < num_vertices; v++) {
        if (dist_ref[v] != dist_test[v]) {
            if (errors < 10)  /* chỉ in 10 lỗi đầu */
                fprintf(stderr, "[VERIFY] v=%d: ref=%d, test=%d\n",
                        v, dist_ref[v], dist_test[v]);
            errors++;
        }
    }
    return errors;
}

vertex_t pick_source(const Graph *g, uint64_t seed)
{
    /* Thử seed trước */
    vertex_t s = (vertex_t)(seed % (uint64_t)g->num_vertices);
    if (graph_degree(g, s) > 0) return s;

    /* Quét từ đầu để chắc chắn */
    for (vertex_t v = 0; v < g->num_vertices; v++)
        if (graph_degree(g, v) > 0) return v;

    return 0;  /* đồ thị rỗng? */
}

void print_progress(long long current, long long total, const char *label)
{
    int pct = (int)((double)current / total * 100);
    int filled = pct / 5;
    fprintf(stderr, "\r[%s] [", label);
    for (int i = 0; i < 20; i++)
        fputc(i < filled ? '#' : '-', stderr);
    fprintf(stderr, "] %3d%%", pct);
    if (current >= total) fputc('\n', stderr);
    fflush(stderr);
}