#ifndef GRAPH_H
#define GRAPH_H

#include <stdint.h>
#include <stdlib.h>

typedef int32_t  vertex_t;   
typedef int64_t  edge_t;    


typedef struct {
    vertex_t  num_vertices;  
    edge_t    num_edges;     
    edge_t   *row_ptr;      
    vertex_t *adj;           
} Graph;


typedef struct {
    double a, b, c, d;
} RMATParams;


Graph *graph_rmat_generate(vertex_t num_vertices,
                           int      scale_factor,
                           uint64_t seed,
                           const RMATParams *params);

void graph_free(Graph *g);

void graph_print_info(const Graph *g);

static inline edge_t graph_degree(const Graph *g, vertex_t v)
{
    return g->row_ptr[v + 1] - g->row_ptr[v];
}

int    graph_save(const Graph *g, const char *path);
Graph *graph_load(const char *path);

#endif /* GRAPH_H */