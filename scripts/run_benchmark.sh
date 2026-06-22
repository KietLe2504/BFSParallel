#!/usr/bin/env bash
# ============================================================================
# run_benchmark.sh — Script benchmark tự động cho BFS Hybrid (MPI + OpenMP)
# ============================================================================
#
# Tự động hoá 4 thí nghiệm yêu cầu trong báo cáo:
#   1) verify   : kiểm tra tính đúng đắn (so với BFS tuần tự)            (5.1)
#   2) scan-n   : quét kích thước N để tìm N chạy ~2-3 phút              (5.2)
#   3) granular : đo granularity / load balancing tại N cố định          (5.3)
#   4) speedup  : quét số tiến trình P tại 2*N cố định                   (5.4)
#   5) all      : chạy lần lượt cả 4 mục trên
#
# CÁCH DÙNG:
#   chmod +x scripts/run_benchmark.sh
#   ./scripts/run_benchmark.sh verify
#   ./scripts/run_benchmark.sh scan-n
#   ./scripts/run_benchmark.sh granular
#   ./scripts/run_benchmark.sh speedup
#
# ⚠️ QUAN TRỌNG: Đọc kỹ phần CẤU HÌNH bên dưới và CHỈNH LẠI theo cluster
#    thật của bạn (số máy, số nhân/máy, hostfile, danh sách N cần quét...)
#    trước khi chạy thí nghiệm thật — các giá trị mặc định chỉ là ví dụ.
# ============================================================================

set -euo pipefail

# ============================================================================
# CẤU HÌNH — CHỈNH THEO CLUSTER THẬT CỦA BẠN
# ============================================================================

# ---- Thông tin cluster -----------------------------------------------------
# Tổng số nhân vật lý của TOÀN BỘ cluster (ví dụ: 3 máy x 4 nhân = 12).
# Đây chính là giá trị "X" trong yêu cầu đề bài (speedup quét đến 2X).
TOTAL_CORES=12

# Hostfile dùng cho mpirun khi chạy đa node. Để rỗng ("") nếu chỉ chạy
# trên 1 máy (mpirun sẽ tự dùng localhost).
HOSTFILE="hostfile"

# Có dùng --oversubscribe không (cần=1 nếu test trên máy ít core hơn
# -np yêu cầu, ví dụ khi debug trên laptop cá nhân trước khi lên cluster
# thật). Đặt =0 khi chạy thật trên cluster để đo hiệu năng chính xác.
OVERSUBSCRIBE=1

# ---- Tham số đồ thị cố định -------------------------------------------------
SCALE=16        # bậc trung bình mỗi đỉnh (avg degree)
SEED=42         # random seed — GIỮ NGUYÊN seed này xuyên suốt mọi thí
                # nghiệm để các lần chạy có thể so sánh công bằng với nhau
                # (cùng 1 đồ thị logic, chỉ khác N/P).

# ---- Mục 5.2: danh sách N để quét tìm khoảng 2-3 phút ----------------------
# Tăng dần theo cấp số nhân trước, sau khi biết khoảng nào gần 2-3 phút,
# BẠN NÊN tự thêm các giá trị N tinh chỉnh xung quanh khoảng đó (script
# không tự động "nhị phân tìm N" — vì thời gian chạy phụ thuộc phần cứng
# thật, cần quan sát kết quả từng bước rồi tự quyết định bước tiếp theo).
SCAN_N_LIST=(500000 1000000 2000000 4000000 8000000 16000000)

# ---- Mục 5.3: N cố định dùng để đo granularity/load-balancing -------------
# ĐIỀN GIÁ TRỊ N0 BẠN ĐÃ CHỌN Ở MỤC 5.2 (sau khi chạy scan-n và xác định
# N cho thời gian chạy 2-3 phút). Giá trị dưới đây CHỈ LÀ PLACEHOLDER.
GRANULARITY_N=4000000

# ---- Mục 5.4: N dùng cho thí nghiệm speedup = 2 * N0 (mục 5.2) ------------
# ĐIỀN GIÁ TRỊ THEO N0 bạn chọn ở mục 5.2.
SPEEDUP_N=$((2 * GRANULARITY_N))

# ---- Mục 5.4: danh sách P (số tiến trình) để quét speedup ------------------
# Theo đề bài: 1, 2, 4, 8, ..., 2X (X = TOTAL_CORES). Hàm dưới tự sinh
# danh sách luỹ thừa 2, dừng khi >= 2*TOTAL_CORES, cộng thêm 2 mốc quan
# trọng TOTAL_CORES và 2*TOTAL_CORES phòng khi TOTAL_CORES không phải
# luỹ thừa 2 (ví dụ TOTAL_CORES=12).
gen_speedup_p_list() {
    local p=1
    local limit=$((2 * TOTAL_CORES))
    while [ "$p" -le "$limit" ]; do
        echo "$p"
        p=$((p * 2))
    done
    echo "$TOTAL_CORES"
    echo "$limit"
}

# ============================================================================
# KHÔNG CẦN SỬA BÊN DƯỚI (logic chạy chung)
# ============================================================================

BIN_HYBRID="./bfs_hybrid"
RESULTS_DIR="results"
GRAPHS_DIR="graphs"
mkdir -p "$RESULTS_DIR" "$GRAPHS_DIR"

TS="$(date +%Y%m%d_%H%M%S)"

# Quyết định số OpenMP thread cho 1 giá trị P cho trước, sao cho
# P * threads xấp xỉ TOTAL_CORES (không vượt quá nhiều, tránh oversubscribe
# nặng khi không cần thiết). Quy tắc đơn giản: threads = max(1, TOTAL_CORES/P)
# làm tròn xuống — đúng tinh thần "ranks x threads <= core vật lý" trong
# README gốc.
threads_for_ranks() {
    local p="$1"
    local t=$(( TOTAL_CORES / p ))
    if [ "$t" -lt 1 ]; then t=1; fi
    echo "$t"
}

# Chạy 1 lệnh mpirun với hostfile (nếu có) + oversubscribe (nếu bật).
run_mpi() {
    local np="$1"; shift
    local omp_threads="$1"; shift
    # "$@" = phần còn lại: đường dẫn binary + tham số chương trình

    local mpi_opts=(-np "$np")
    if [ -n "$HOSTFILE" ] && [ -s "$HOSTFILE" ]; then
        mpi_opts+=(--hostfile "$HOSTFILE")
    fi
    if [ "$OVERSUBSCRIBE" = "1" ]; then
        mpi_opts+=(--oversubscribe)
    fi
    # --allow-run-as-root chỉ cần khi chạy trong container/CI dưới user
    # root; trên cluster thật chạy bằng user mpiuser thì có thể bỏ option
    # này — script vẫn để sẵn để không lỗi khi test nhanh.
    echo ">> OMP_NUM_THREADS=$omp_threads mpirun ${mpi_opts[*]} --allow-run-as-root $*"
    OMP_NUM_THREADS="$omp_threads" mpirun "${mpi_opts[@]}" --allow-run-as-root "$@"
}

check_binary() {
    if [ ! -x "$BIN_HYBRID" ]; then
        echo "[ERROR] Khong tim thay $BIN_HYBRID — chay 'make' truoc." >&2
        exit 1
    fi
}

# Trả về "--graph <file>" nếu file đồ thị ứng với N này đã có sẵn,
# hoặc rỗng nếu chưa có (sẽ gen lại). Dùng để tự động tái sử dụng
# graph đã lưu mà không cần chỉnh thủ công.
graph_flag() {
    local n="$1"
    local path="$GRAPHS_DIR/g${n}_s${SCALE}_seed${SEED}.bin"
    if [ -f "$path" ]; then
        echo "--graph $path"
    else
        echo ""
    fi
}

# ----------------------------------------------------------------------------
# 0) GEN-GRAPH — Sinh và lưu đồ thị ra file (chạy 1 lần trước benchmark)
# ----------------------------------------------------------------------------
exp_gen_graph() {
    echo "================================================================"
    echo " GEN-GRAPH — Sinh va luu do thi ra file (de dung lai)"
    echo "================================================================"
    check_binary

    if [ ${#SCAN_N_LIST[@]} -eq 0 ]; then
        echo "[ERROR] SCAN_N_LIST rong" >&2; exit 1
    fi

    # Sinh graph cho mọi N trong SCAN_N_LIST + GRANULARITY_N + SPEEDUP_N
    local all_n=("${SCAN_N_LIST[@]}" "$GRANULARITY_N" "$SPEEDUP_N")

    for n in "${all_n[@]}"; do
        local path="$GRAPHS_DIR/g${n}_s${SCALE}_seed${SEED}.bin"
        if [ -f "$path" ]; then
            echo "-- N=$n: da co san tai $path, bo qua."
            continue
        fi
        echo "-- N=$n: dang gen -> $path"
        # Gen bằng -np 1 (chỉ rank 0 lưu file), 1 thread là đủ
        OMP_NUM_THREADS=1 mpirun --allow-run-as-root \
            $( [ "$OVERSUBSCRIBE" = "1" ] && echo "--oversubscribe" ) \
            -np 1 "$BIN_HYBRID" "$n" "$SCALE" "$SEED" \
            --save-graph "$path" 2>&1 | grep -E "Saved|generated|vertices" || true
        echo "   -> $path"
        echo ""
    done
    echo "Xong. Cac lan chay benchmark sau se tu dong dung --graph <file>."
}

# ----------------------------------------------------------------------------
# 1) VERIFY — Mục 5.1: kiểm tra tính đúng đắn
# ----------------------------------------------------------------------------
exp_verify() {
    echo "================================================================"
    echo " THI NGHIEM 5.1 — VERIFY CORRECTNESS"
    echo "================================================================"
    check_binary

    local out="$RESULTS_DIR/verify_${TS}.log"
    # Quét vài cấu hình khác nhau (nhỏ + lớn, P khác nhau) để chứng minh
    # tính đúng đắn ổn định, không phải ăn may với 1 cấu hình duy nhất.
    local configs=(
        "10000 8 1"
        "10000 8 4"
        "100000 16 2"
        "100000 16 $TOTAL_CORES"
        "1000000 16 $TOTAL_CORES"
    )

    : > "$out"
    for cfg in "${configs[@]}"; do
        read -r n s p <<< "$cfg"
        local threads; threads=$(threads_for_ranks "$p")
        local gflag; gflag=$(graph_flag "$n")
        echo "" | tee -a "$out"
        echo "---- N=$n scale=$s ranks=$p threads=$threads $gflag----" | tee -a "$out"
        run_mpi "$p" "$threads" "$BIN_HYBRID" "$n" "$s" "$SEED" --verify $gflag \
            2>&1 | tee -a "$out" | grep -E "Verify|Speedup|mismatches|Loaded|Generated" || true
    done
    echo ""
    echo "Log day du: $out"
}

# ----------------------------------------------------------------------------
# 2) SCAN-N — Mục 5.2: quét N tại P = TOTAL_CORES, tìm thời gian 2-3 phút
# ----------------------------------------------------------------------------
exp_scan_n() {
    echo "================================================================"
    echo " THI NGHIEM 5.2 — SCAN N (tim N chay 2-3 phut), P=$TOTAL_CORES co dinh"
    echo "================================================================"
    check_binary

    local csv="$RESULTS_DIR/scan_n_${TS}.csv"
    local threads; threads=$(threads_for_ranks "$TOTAL_CORES")

    echo "P co dinh = $TOTAL_CORES ranks x $threads threads = $((TOTAL_CORES*threads)) cores"
    echo "CSV output: $csv"
    echo ""

    for n in "${SCAN_N_LIST[@]}"; do
        local gflag; gflag=$(graph_flag "$n")
        echo "---- N=$n $gflag----"
        run_mpi "$TOTAL_CORES" "$threads" "$BIN_HYBRID" "$n" "$SCALE" "$SEED" \
            $gflag --csv "$csv" 2>&1 | grep -E "Time|MTEPS|Graph (generated|loaded)" || true
        echo ""
    done

    echo "Xong. Mo file $csv, ve bieu do: truc X = num_vertices,"
    echo "truc Y = bfs_time_ms (duong 1) va (bfs_time_ms - comm_ms) (duong 2)."
    echo "Tim N sao cho bfs_time_ms nam trong khoang 120000-180000 ms (2-3 phut)."
    echo "Neu chua dat khoang nay, sua mang SCAN_N_LIST trong script roi chay lai."
}

# ----------------------------------------------------------------------------
# 3) GRANULAR — Mục 5.3: đo per-rank breakdown tại N cố định, P = TOTAL_CORES
# ----------------------------------------------------------------------------
exp_granular() {
    echo "================================================================"
    echo " THI NGHIEM 5.3 — GRANULARITY / LOAD BALANCING"
    echo " N=$GRANULARITY_N (co dinh), P=$TOTAL_CORES (co dinh)"
    echo "================================================================"
    check_binary

    local csv_ranks="$RESULTS_DIR/granularity_ranks_${TS}.csv"
    local csv_summary="$RESULTS_DIR/granularity_summary_${TS}.csv"
    local threads; threads=$(threads_for_ranks "$TOTAL_CORES")

    local gflag; gflag=$(graph_flag "$GRANULARITY_N")
    echo "CSV per-rank: $csv_ranks"
    if [ -n "$gflag" ]; then echo "Graph: $gflag"; fi
    echo ""

    run_mpi "$TOTAL_CORES" "$threads" "$BIN_HYBRID" "$GRANULARITY_N" "$SCALE" "$SEED" \
        $gflag --csv "$csv_summary" --csv-ranks "$csv_ranks" 2>&1 | tail -40

    echo ""
    echo "Xong. Mo file $csv_ranks de ve bieu do cot chong (1 cot = 1 rank,"
    echo "2 mau: compute_ms va comm_ms). Dong 'Load imbalance' in ra console"
    echo "o tren da tinh san % lech (max-min)/max — neu > 25% can dieu chinh"
    echo "lai granularity (vd: edge-balanced partition thay vi vertex-balanced)."
}

# ----------------------------------------------------------------------------
# 4) SPEEDUP — Mục 5.4: quét P tại N = 2*N0 cố định
# ----------------------------------------------------------------------------
exp_speedup() {
    echo "================================================================"
    echo " THI NGHIEM 5.4 — SPEEDUP, N=$SPEEDUP_N (co dinh = 2 x N0)"
    echo " Quet P tu 1 den $((2*TOTAL_CORES))"
    echo "================================================================"
    check_binary

    local csv="$RESULTS_DIR/speedup_${TS}.csv"
    local gflag; gflag=$(graph_flag "$SPEEDUP_N")
    echo "CSV output: $csv"
    if [ -n "$gflag" ]; then echo "Graph: $gflag"; fi
    echo ""

    # Lấy danh sách P, loại trùng lặp, sắp xếp tăng dần.
    local p_list
    p_list=$(gen_speedup_p_list | sort -n -u)

    for p in $p_list; do
        local threads; threads=$(threads_for_ranks "$p")
        echo "---- P=$p ranks x $threads threads = $((p*threads)) cores ----"
        run_mpi "$p" "$threads" "$BIN_HYBRID" "$SPEEDUP_N" "$SCALE" "$SEED" \
            $gflag --csv "$csv" 2>&1 | grep -E "Time|MTEPS" || true
        echo ""
    done

    echo "Xong. Mo file $csv:"
    echo "  - Bieu do 1: truc X = num_ranks, truc Y = bfs_time_ms (2 duong:"
    echo "    co/khong comm = bfs_time_ms va bfs_time_ms - comm_ms)"
    echo "  - Bieu do 2: Speedup(P) = bfs_time_ms(P=1) / bfs_time_ms(P)"
    echo "    ve kem duong ly tuong Speedup=P de so sanh."
}

# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
usage() {
    cat <<EOF
Usage: $0 <command>

Commands:
  gen-graph  Sinh va luu do thi cho moi N trong SCAN_N_LIST + GRANULARITY_N
             vao thu muc graphs/ (chay 1 lan duy nhat truoc benchmark)
  verify     Muc 5.1 - kiem tra tinh dung dan (nhieu cau hinh)
  scan-n     Muc 5.2 - quet N (P=$TOTAL_CORES co dinh) de tim N chay 2-3 phut
  granular   Muc 5.3 - do granularity/load-balancing tai N=$GRANULARITY_N co dinh
  speedup    Muc 5.4 - quet P (N=$SPEEDUP_N co dinh = 2xN0) de do speedup
  all        Chay lan luot ca 4 thi nghiem tren

Quy trinh khuyen nghi:
  1) ./scripts/run_benchmark.sh gen-graph   # gen + luu 1 lan
  2) ./scripts/run_benchmark.sh scan-n      # chon N0
  3) Dien N0 vao GRANULARITY_N trong script
  4) ./scripts/run_benchmark.sh granular
  5) ./scripts/run_benchmark.sh speedup

LUU Y: chinh cac bien cau hinh o dau file script (TOTAL_CORES, HOSTFILE,
SCAN_N_LIST, GRANULARITY_N...) truoc khi chay that tren cluster.
EOF
}

main() {
    local cmd="${1:-}"
    case "$cmd" in
        gen-graph) exp_gen_graph ;;
        verify)    exp_verify ;;
        scan-n)    exp_scan_n ;;
        granular)  exp_granular ;;
        speedup)   exp_speedup ;;
        all)
            exp_verify
            exp_scan_n
            exp_granular
            exp_speedup
            ;;
        *)
            usage
            exit 1
            ;;
    esac
}

main "$@"