#ifndef BFS_SEQUENTIAL_H
#define BFS_SEQUENTIAL_H

#include "graph.h"

edge_t bfs_sequential(const Graph *g, vertex_t source, int *dist);

#endif 