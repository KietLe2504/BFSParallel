#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "graph.h"
#include "bfs_sequential.h"
#include "utils.h"

static void usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s <num_vertices> <scale_factor> <seed> [options]\n"
            "  num_vertices       : number of graph vertices (e.g. 1000000)\n"
            "  scale_factor       : avg degree (e.g. 16)\n"
            "  seed               : random seed (e.g. 42)\n"
            "\nOptions:\n"
            "  --graph <file>     : load pre-generated graph from binary file\n"
            "  --save-graph <file>: save generated graph to binary file\n",
            prog);
}

/* Tìm flag "--flag value" trong argv (bắt đầu từ argv[4]) */
static const char *find_flag_value(int argc, char *argv[], const char *flag)
{
    for (int i = 4; i < argc - 1; i++)
        if (strcmp(argv[i], flag) == 0) return argv[i + 1];
    return NULL;
}

int main(int argc, char *argv[])
{
    if (argc < 4) { usage(argv[0]); return 1; }

    vertex_t    num_vertices    = (vertex_t)atoi(argv[1]);
    int         scale           = atoi(argv[2]);
    uint64_t    seed            = (uint64_t)atoll(argv[3]);
    const char *graph_load_path = find_flag_value(argc, argv, "--graph");
    const char *graph_save_path = find_flag_value(argc, argv, "--save-graph");

    if (num_vertices <= 0 || scale <= 0) {
        fprintf(stderr, "[SEQ] Invalid arguments\n");
        return 1;
    }

    /* ---- Load hoặc Gen đồ thị ------------------------------------- */
    Graph *g    = NULL;
    double gen_ms = 0.0;

    if (graph_load_path) {
        printf("[SEQ] Loading graph from '%s' ...\n", graph_load_path);
        Timer t_load;
        timer_start(&t_load);
        g = graph_load(graph_load_path);
        gen_ms = timer_elapsed_ms(&t_load);
        if (!g) { fprintf(stderr, "[SEQ] Failed to load graph\n"); return 1; }
        graph_print_info(g);
        printf("[SEQ] Graph loaded in %.2f ms\n\n", gen_ms);
    } else {
        printf("[SEQ] Generating R-MAT graph: %d vertices, scale=%d, seed=%llu\n",
               num_vertices, scale, (unsigned long long)seed);
        Timer t_gen;
        timer_start(&t_gen);
        g = graph_rmat_generate(num_vertices, scale, seed, NULL);
        gen_ms = timer_elapsed_ms(&t_gen);
        if (!g) { fprintf(stderr, "[SEQ] Graph generation failed\n"); return 1; }
        graph_print_info(g);
        printf("[SEQ] Graph generated in %.2f ms\n\n", gen_ms);

        if (graph_save_path) {
            if (graph_save(g, graph_save_path) == 0)
                printf("[SEQ] Graph saved to '%s'\n\n", graph_save_path);
            else
                fprintf(stderr, "[SEQ] Warning: failed to save graph\n");
        }
    }

    /* ---- Chọn source --------------------------------------------- */
    vertex_t source = pick_source(g, seed);
    printf("[SEQ] BFS from source vertex %d\n", source);

    /* ---- Chạy BFS ------------------------------------------------ */
    int *dist = (int *)malloc((size_t)g->num_vertices * sizeof(int));
    if (!dist) { fprintf(stderr, "[SEQ] malloc dist failed\n"); graph_free(g); return 1; }

    Timer t_bfs;
    timer_start(&t_bfs);
    edge_t visited_edges = bfs_sequential(g, source, dist);
    double bfs_ms = timer_elapsed_ms(&t_bfs);

    /* ---- In kết quả --------------------------------------------- */
    vertex_t visited_verts = count_visited(dist, g->num_vertices);
    double   mteps         = compute_mteps(visited_edges, bfs_ms);

    printf("[SEQ] Graph: %d vertices, %lld edges\n",
           g->num_vertices, (long long)g->num_edges);
    printf("[SEQ] Visited: %d vertices, %lld edges\n",
           (int)visited_verts, (long long)visited_edges);
    printf("[SEQ] Time: %.2f ms\n", bfs_ms);
    printf("[SEQ] MTEPS: %.2f\n", mteps);

    free(dist);
    graph_free(g);
    return 0;
}