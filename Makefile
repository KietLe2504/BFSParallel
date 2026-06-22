# ============================================================
# Makefile – BFS Hybrid (MPI + OpenMP + Direction-Optimizing)
# ============================================================

CC      = mpicc
CFLAGS  = -O2 -fopenmp -Wall -Wextra -std=c11
LDFLAGS = -lm

SRC_DIR = src

# ---- Source files -----------------------------------------------
SEQ_SRC = $(SRC_DIR)/main_seq.c       \
          $(SRC_DIR)/bfs_sequential.c  \
          $(SRC_DIR)/graph.c           \
          $(SRC_DIR)/utils.c

HYB_SRC = $(SRC_DIR)/main_hybrid.c   \
          $(SRC_DIR)/bfs_hybrid.c     \
          $(SRC_DIR)/bfs_sequential.c \
          $(SRC_DIR)/graph.c          \
          $(SRC_DIR)/utils.c

# ---- Targets ----------------------------------------------------
.PHONY: all clean verify test_local test_multi

all: bfs_seq bfs_hybrid

bfs_seq: $(SEQ_SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "[BUILD] bfs_seq OK"

bfs_hybrid: $(HYB_SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "[BUILD] bfs_hybrid OK"

# ---- Quick smoke tests ------------------------------------------

# Chạy sequential baseline
test_seq: bfs_seq
	./bfs_seq 500000 16 42

# Chạy hybrid 1 node, 1 rank, 4 threads
test_local: bfs_hybrid
	OMP_NUM_THREADS=4 mpirun -np 1 ./bfs_hybrid 500000 16 42 --verify

# Chạy hybrid 1 node, 4 rank, 2 threads — cần đúng hostfile
test_multi: bfs_hybrid
	OMP_NUM_THREADS=2 mpirun -np 4 --hostfile hostfile ./bfs_hybrid 1000000 16 42

# Verify correctness với graph nhỏ hơn
verify: bfs_hybrid
	OMP_NUM_THREADS=2 mpirun -np 2 ./bfs_hybrid 100000 8 42 --verify

# ---- Clean ------------------------------------------------------
clean:
	rm -f bfs_seq bfs_hybrid results/*.csv
	@echo "[CLEAN] done"
