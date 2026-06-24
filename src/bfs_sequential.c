#include "bfs_sequential.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

edge_t bfs_sequential(const Graph *g, vertex_t source, int *dist)
{
    vertex_t n = g->num_vertices;

    //Khởi tạo dist[] = -1 
    memset(dist, -1, (size_t)n * sizeof(int));

    //Queue vòng tròn 
    vertex_t *queue = (vertex_t *)malloc((size_t)n * sizeof(vertex_t));
    if (!queue) {
        fprintf(stderr, "[SEQ] malloc queue failed\n");
        return 0;
    }

    edge_t visited_edges = 0;
    vertex_t head = 0, tail = 0;

    dist[source] = 0;
    queue[tail++] = source;

    while (head < tail) {
        vertex_t u = queue[head++];
        int      d = dist[u] + 1;

        edge_t start = g->row_ptr[u];
        edge_t end   = g->row_ptr[u + 1];

        for (edge_t e = start; e < end; e++) {
            visited_edges++;
            vertex_t v = g->adj[e];
            if (dist[v] == -1) {
                dist[v]      = d;
                queue[tail++] = v;
            }
        }
    }

    free(queue);
    return visited_edges;
}