#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>

#include "graph.h"
#include "bfs_hybrid.h"
#include "bfs_sequential.h"
#include "utils.h"

static void usage(const char *prog)
{
    fprintf(stderr,
            "Usage: mpirun -np <N> %s <num_vertices> <scale_factor> <seed> [options]\n"
            "  num_vertices       : number of graph vertices (e.g. 1000000)\n"
            "  scale_factor       : avg degree (e.g. 16)\n"
            "  seed               : random seed (e.g. 42)\n"
            "\nOptions:\n"
            "  --verify           : verify result against sequential BFS\n"
            "  --graph <file>     : load pre-generated graph from binary file\n"
            "                       (skips R-MAT generation — faster for benchmarks)\n"
            "                       num_vertices/scale/seed args still required but\n"
            "                       ignored for graph structure (used for CSV logging)\n"
            "  --save-graph <file>: after generating, save graph to binary file\n"
            "                       (can be reloaded later with --graph)\n"
            "  --csv <file>       : append 1 summary row to CSV file\n"
            "  --csv-ranks <file> : append per-rank breakdown rows to CSV file\n"
            "\nWorkflow (recommended):\n"
            "  # 1. Gen + save once (run with -np 1, rank 0 saves the file)\n"
            "  mpirun -np 1 %s 4000000 16 42 --save-graph graphs/g4M.bin\n"
            "  # 2. All subsequent runs load from file (fast)\n"
            "  mpirun -np 12 %s 4000000 16 42 --graph graphs/g4M.bin --csv results/out.csv\n",
            prog, prog, prog);
}

/* Tìm giá trị đi kèm sau 1 flag dạng "--flag value" trong argv.        *
 * Trả về con trỏ tới value, hoặc NULL nếu không tìm thấy flag.         */
static const char *find_flag_value(int argc, char *argv[], const char *flag)
{
    for (int i = 4; i < argc - 1; i++) {
        if (strcmp(argv[i], flag) == 0) return argv[i + 1];
    }
    return NULL;
}

static int has_flag(int argc, char *argv[], const char *flag)
{
    for (int i = 4; i < argc; i++) {
        if (strcmp(argv[i], flag) == 0) return 1;
    }
    return 0;
}

/* ---- Ghi 1 dòng tổng hợp vào CSV (mục 5.1-5.2-5.4: so sánh N, P) --- *
 * Format cột:                                                          *
 * timestamp,num_vertices,scale,seed,num_ranks,omp_threads,total_cores, *
 * gen_ms,bfs_time_ms,compute_ms,comm_ms,num_levels,visited_edges,mteps,*
 * seq_time_ms,speedup,verify_status                                    *
 * Nếu không chạy --verify thì seq_time_ms/speedup/verify_status để      *
 * trống (rỗng), không phải 0, để khỏi gây nhầm khi vẽ biểu đồ.          */
static void csv_append_summary(const char *path,
                               vertex_t num_vertices, int scale, uint64_t seed,
                               int num_ranks, int omp_threads,
                               double gen_ms, const BFSResult *r,
                               double mteps,
                               int has_verify, double seq_ms, double speedup,
                               int verify_errors)
{
    FILE *f = fopen(path, "r");
    int need_header = (f == NULL);
    if (f) fclose(f);

    f = fopen(path, "a");
    if (!f) {
        fprintf(stderr, "[HYBRID] WARNING: cannot open csv file '%s' for append\n", path);
        return;
    }
    if (need_header) {
        fprintf(f,
            "num_vertices,scale,seed,num_ranks,omp_threads,total_cores,"
            "gen_ms,bfs_time_ms,compute_ms,comm_ms,num_levels,visited_edges,"
            "mteps,seq_time_ms,speedup,verify_errors\n");
    }
    fprintf(f, "%d,%d,%llu,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%d,%lld,%.3f,",
            num_vertices, scale, (unsigned long long)seed,
            num_ranks, omp_threads, num_ranks * omp_threads,
            gen_ms, r->time_ms, r->compute_ms, r->comm_ms,
            r->num_levels, (long long)r->visited_edges, mteps);
    if (has_verify) {
        fprintf(f, "%.3f,%.4f,%d\n", seq_ms, speedup, verify_errors);
    } else {
        fprintf(f, ",,\n");  /* để trống nếu không verify */
    }
    fclose(f);
}

/* ---- Ghi per-rank breakdown vào CSV (mục 5.3: granularity/load) ---- *
 * Format cột:                                                          *
 * num_vertices,scale,seed,num_ranks,omp_threads,rank,                  *
 * local_vertices,local_edges,compute_ms,comm_ms,total_ms               *
 * 1 dòng / rank / lần chạy → dễ group theo (num_vertices,num_ranks)     *
 * khi vẽ biểu đồ cột chồng trong mục 5.3.                               */
static void csv_append_rank_stats(const char *path,
                                  vertex_t num_vertices, int scale, uint64_t seed,
                                  int num_ranks, int omp_threads,
                                  const RankStats *stats, int n_stats)
{
    FILE *f = fopen(path, "r");
    int need_header = (f == NULL);
    if (f) fclose(f);

    f = fopen(path, "a");
    if (!f) {
        fprintf(stderr, "[HYBRID] WARNING: cannot open csv-ranks file '%s' for append\n", path);
        return;
    }
    if (need_header) {
        fprintf(f,
            "num_vertices,scale,seed,num_ranks,omp_threads,rank,"
            "local_vertices,local_edges,compute_ms,comm_ms,total_ms\n");
    }
    for (int i = 0; i < n_stats; i++) {
        fprintf(f, "%d,%d,%llu,%d,%d,%d,%d,%lld,%.3f,%.3f,%.3f\n",
                num_vertices, scale, (unsigned long long)seed,
                num_ranks, omp_threads, stats[i].rank,
                stats[i].local_count, (long long)stats[i].local_edges,
                stats[i].compute_ms, stats[i].comm_ms, stats[i].total_ms);
    }
    fclose(f);
}

int main(int argc, char *argv[])
{
    /* ---- Khởi tạo MPI -------------------------------------------- */
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr,
                "[HYBRID] Warning: MPI thread support level %d < MPI_THREAD_FUNNELED\n",
                provided);
    }

    int my_rank, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    /* ---- Parse arguments ----------------------------------------- */
    if (argc < 4) {
        if (my_rank == 0) usage(argv[0]);
        MPI_Finalize();
        return 1;
    }

    vertex_t num_vertices   = (vertex_t)atoi(argv[1]);
    int      scale          = atoi(argv[2]);
    uint64_t seed           = (uint64_t)atoll(argv[3]);
    int      do_verify      = has_flag(argc, argv, "--verify");
    int      use_delta_sync = has_flag(argc, argv, "--delta-sync");
    const char *csv_path        = find_flag_value(argc, argv, "--csv");
    const char *csv_ranks_path  = find_flag_value(argc, argv, "--csv-ranks");
    const char *graph_load_path = find_flag_value(argc, argv, "--graph");
    const char *graph_save_path = find_flag_value(argc, argv, "--save-graph");

    if (num_vertices <= 0 || scale <= 0) {
        if (my_rank == 0)
            fprintf(stderr, "[HYBRID] Invalid arguments\n");
        MPI_Finalize();
        return 1;
    }

    /* ---- In thông tin môi trường ---------------------------------- */
    int omp_threads = omp_get_max_threads();
    if (my_rank == 0) {
        printf("[HYBRID] ============================================\n");
        printf("[HYBRID] BFS Hybrid (MPI + OpenMP + Direction-Opt)\n");
        printf("[HYBRID] MPI ranks   : %d\n", num_ranks);
        printf("[HYBRID] OMP threads : %d per rank\n", omp_threads);
        printf("[HYBRID] Total cores : %d\n", num_ranks * omp_threads);
        printf("[HYBRID] Alpha       : %d\n", BFS_ALPHA);
        printf("[HYBRID] Beta        : %d\n", BFS_BETA);
        printf("[HYBRID] ============================================\n\n");
        fflush(stdout);
    }

    /* ---- Sinh hoặc Load đồ thị ---------------------------------------- *
     * --graph <file> : load từ file binary (bỏ qua R-MAT gen, nhanh hơn)   *
     * Mặc định      : gen R-MAT như cũ (+ tuỳ chọn --save-graph để lưu)    *
     * -------------------------------------------------------------------- */
    Graph *g    = NULL;
    double gen_ms = 0.0;

    if (graph_load_path) {
        /* Load từ file — tất cả rank load độc lập (replicated graph) */
        if (my_rank == 0) {
            printf("[HYBRID] Loading graph from '%s' ...\n", graph_load_path);
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        Timer t_load;
        timer_start(&t_load);
        g = graph_load(graph_load_path);
        gen_ms = timer_elapsed_ms(&t_load);   /* đổi tên ý nghĩa: load time */

        if (!g) {
            if (my_rank == 0)
                fprintf(stderr, "[HYBRID] Failed to load graph from '%s'\n",
                        graph_load_path);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (my_rank == 0) {
            graph_print_info(g);
            printf("[HYBRID] Graph loaded in %.2f ms\n\n", gen_ms);
            fflush(stdout);
        }
    } else {
        /* Gen R-MAT như cũ */
        if (my_rank == 0) {
            printf("[HYBRID] Generating R-MAT graph: %d vertices, scale=%d, seed=%llu\n",
                   num_vertices, scale, (unsigned long long)seed);
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        Timer t_gen;
        timer_start(&t_gen);
        g = graph_rmat_generate(num_vertices, scale, seed, NULL);
        gen_ms = timer_elapsed_ms(&t_gen);

        if (!g) {
            fprintf(stderr, "[HYBRID] rank %d: graph generation failed\n", my_rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (my_rank == 0) {
            graph_print_info(g);
            printf("[HYBRID] Graph generated in %.2f ms\n\n", gen_ms);
            fflush(stdout);
        }

        /* --save-graph: chỉ rank 0 lưu (graph giống nhau trên mọi rank) */
        if (graph_save_path && my_rank == 0) {
            if (graph_save(g, graph_save_path) == 0)
                printf("[HYBRID] Graph saved to '%s'\n\n", graph_save_path);
            else
                fprintf(stderr, "[HYBRID] Warning: failed to save graph\n");
            fflush(stdout);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    /* ---- Chọn source --------------------------------------------- */
    vertex_t source = pick_source(g, seed);
    if (my_rank == 0) {
        printf("[HYBRID] Sync mode  : %s\n",
               use_delta_sync ? "DELTA (Allgatherv new vertices)"
                              : "FULL  (Allreduce dist[], O(n))");
        printf("[HYBRID] BFS from source vertex %d\n\n", source);
        fflush(stdout);
    }

    /* ---- Chạy BFS Hybrid ----------------------------------------- */
    MPI_Barrier(MPI_COMM_WORLD);

    BFSResult result = bfs_hybrid(g, source, use_delta_sync);

    MPI_Barrier(MPI_COMM_WORLD);

    /* ---- In kết quả (chỉ rank 0) ---------------------------------- */
    double mteps = 0.0;
    if (my_rank == 0) {
        vertex_t visited_verts = count_visited(result.dist, g->num_vertices);
        mteps = compute_mteps(result.visited_edges, result.time_ms);

        printf("\n[HYBRID] ============================================\n");
        printf("[HYBRID] Graph    : %d vertices, %lld edges\n",
               g->num_vertices, (long long)g->num_edges);
        printf("[HYBRID] Visited  : %d vertices, %lld edges\n",
               (int)visited_verts, (long long)result.visited_edges);
        printf("[HYBRID] Levels   : %d\n", result.num_levels);
        printf("[HYBRID] Time     : %.2f ms  (compute: %.2f ms | comm: %.2f ms, rank 0)\n",
               result.time_ms, result.compute_ms, result.comm_ms);
        printf("[HYBRID] MTEPS    : %.2f\n", mteps);
        fflush(stdout);

        /* ---- In breakdown từng rank (mục 5.3 — granularity/load balance) */
        if (result.rank_stats) {
            printf("\n[HYBRID] Per-rank breakdown:\n");
            printf("[HYBRID]   %-5s %12s %14s %12s %12s %12s\n",
                   "rank", "vertices", "edges", "compute(ms)", "comm(ms)", "total(ms)");
            double min_total = result.rank_stats[0].total_ms;
            double max_total = result.rank_stats[0].total_ms;
            for (int i = 0; i < result.num_ranks; i++) {
                const RankStats *s = &result.rank_stats[i];
                printf("[HYBRID]   %-5d %12d %14lld %12.2f %12.2f %12.2f\n",
                       s->rank, s->local_count, (long long)s->local_edges,
                       s->compute_ms, s->comm_ms, s->total_ms);
                if (s->total_ms < min_total) min_total = s->total_ms;
                if (s->total_ms > max_total) max_total = s->total_ms;
            }
            double imbalance_pct = (max_total > 0.0)
                ? (max_total - min_total) / max_total * 100.0 : 0.0;
            printf("[HYBRID]   Load imbalance (max-min)/max = %.1f%% %s\n",
                   imbalance_pct,
                   imbalance_pct > 25.0 ? "(> 25% -> MAT CAN BANG TAI)"
                                        : "(<= 25% -> tam on)");
            fflush(stdout);
        }
    }

    /* ---- Verify (tùy chọn) --------------------------------------- */
    int    has_verify_result = 0;
    double seq_ms            = 0.0;
    double speedup           = 0.0;
    int    verify_errors     = -1;  /* -1 = không chạy verify */

    if (do_verify && my_rank == 0) {
        printf("\n[HYBRID] Running sequential BFS for verification...\n");
        fflush(stdout);

        int *dist_seq = (int *)malloc((size_t)g->num_vertices * sizeof(int));
        if (!dist_seq) {
            fprintf(stderr, "[HYBRID] malloc dist_seq failed\n");
        } else {
            Timer t_seq;
            timer_start(&t_seq);
            edge_t seq_edges = bfs_sequential(g, source, dist_seq);
            seq_ms = timer_elapsed_ms(&t_seq);

            verify_errors = verify_bfs(dist_seq, result.dist, g->num_vertices);

            double seq_mteps = compute_mteps(seq_edges, seq_ms);
            speedup           = seq_ms / result.time_ms;
            has_verify_result = 1;

            printf("[HYBRID] Sequential : %.2f ms | MTEPS: %.2f\n",
                   seq_ms, seq_mteps);
            printf("[HYBRID] Speedup    : %.2fx\n", speedup);

            if (verify_errors == 0)
                printf("[HYBRID] Verify     : PASSED \xE2\x9C\x93\n");
            else
                printf("[HYBRID] Verify     : FAILED \xE2\x9C\x97 (%d mismatches)\n", verify_errors);

            free(dist_seq);
        }
    }

    if (my_rank == 0) {
        printf("[HYBRID] ============================================\n");
        fflush(stdout);
    }

    /* ---- Xuất CSV (tùy chọn, chỉ rank 0 ghi file) ------------------ */
    if (my_rank == 0) {
        if (csv_path) {
            csv_append_summary(csv_path, num_vertices, scale, seed,
                               num_ranks, omp_threads,
                               gen_ms, &result, mteps,
                               has_verify_result, seq_ms, speedup, verify_errors);
            printf("[HYBRID] Summary appended to %s\n", csv_path);
        }
        if (csv_ranks_path && result.rank_stats) {
            csv_append_rank_stats(csv_ranks_path, num_vertices, scale, seed,
                                  num_ranks, omp_threads,
                                  result.rank_stats, result.num_ranks);
            printf("[HYBRID] Per-rank stats appended to %s\n", csv_ranks_path);
        }
        fflush(stdout);
    }

    /* ---- Dọn dẹp ------------------------------------------------- */
    bfs_result_free(&result);
    graph_free(g);

    MPI_Finalize();
    return 0;
}