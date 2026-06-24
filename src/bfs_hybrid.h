#ifndef BFS_HYBRID_H
#define BFS_HYBRID_H

#include "graph.h"
#include <mpi.h>


#define BFS_ALPHA  14
#define BFS_BETA   24


typedef struct {
    vertex_t total_vertices;  
    int      num_ranks;       
    int      my_rank;         
    vertex_t local_start;     
    vertex_t local_end;       
    vertex_t local_count;     
} Partition;

static inline vertex_t partition_start(vertex_t n, int num_ranks, int rank)
{
    return (vertex_t)((long long)n * rank / num_ranks);
}

static inline vertex_t partition_end(vertex_t n, int num_ranks, int rank)
{
    return (vertex_t)((long long)n * (rank + 1) / num_ranks);
}


typedef struct {
    int      rank;
    double   compute_ms;
    double   comm_ms;
    double   total_ms;
    vertex_t local_count;    
    edge_t   local_edges;    
} RankStats;

typedef struct {
    int     *dist;           
    int      num_levels;     
    edge_t   visited_edges;  
    double   time_ms;        

    double   compute_ms;     
    double   comm_ms;        

    RankStats *rank_stats;
    int        num_ranks;
} BFSResult;


BFSResult bfs_hybrid(const Graph *g, vertex_t source);

void bfs_result_free(BFSResult *r);

#endif 