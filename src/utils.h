#ifndef UTILS_H
#define UTILS_H

#include <stdint.h>
#include <time.h>
#include "graph.h"


typedef struct {
    struct timespec _start;
} Timer;

void   timer_start(Timer *t);
double timer_elapsed_ms(const Timer *t);   


double compute_mteps(edge_t visited_edges, double elapsed_ms);

vertex_t count_visited(const int *dist, vertex_t num_vertices);

int verify_bfs(const int *dist_ref, const int *dist_test,
               vertex_t num_vertices);

vertex_t pick_source(const Graph *g, uint64_t seed);

void print_progress(long long current, long long total, const char *label);

#endif /* UTILS_H */
