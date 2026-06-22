# BFS Hybrid — MPI + OpenMP + Direction-Optimizing

Project **Lập trình song song (Parallel Programming)** cài đặt thuật toán
**Breadth-First Search song song** trên đồ thị R-MAT quy mô lớn, chạy
phân tán trên nhiều máy (cluster) sử dụng MPI + OpenMP hybrid.

---

## Mục lục

1. [Cấu trúc project](#1-cấu-trúc-project)
2. [Thuật toán](#2-thuật-toán)
3. [Cài đặt dependencies](#3-cài-đặt-dependencies)
4. [Build project](#4-build-project)
5. [Chạy trên 1 máy (local)](#5-chạy-trên-1-máy-local)
6. [Thiết lập Cluster — Hướng dẫn chi tiết](#6-thiết-lập-cluster--hướng-dẫn-chi-tiết)
   - [Yêu cầu phần cứng / mạng](#61-yêu-cầu-phần-cứng--mạng)
   - [Cài OpenMPI trên tất cả nodes](#62-cài-openmpi-trên-tất-cả-nodes)
   - [Tạo user chung và cấu hình SSH](#63-tạo-user-chung-và-cấu-hình-ssh-không-password)
   - [Cấu hình NFS chia sẻ code](#64-cấu-hình-nfs-chia-sẻ-thư-mục-code)
   - [Cấu hình /etc/hosts](#65-cấu-hình-etchosts)
   - [Tạo hostfile](#66-tạo-hostfile)
   - [Test kết nối cluster](#67-test-kết-nối-cluster)
   - [Chạy chương trình trên cluster](#68-chạy-chương-trình-trên-cluster)
7. [Tham số dòng lệnh](#7-tham-số-dòng-lệnh)
8. [Benchmark tự động (thí nghiệm báo cáo)](#8-benchmark-tự-động--thí-nghiệm-báo-cáo)
   - [Cấu hình script](#81-cấu-hình-script-trước-khi-chạy)
   - [Thí nghiệm 5.1 — Verify](#82-thí-nghiệm-51--verify-correctness)
   - [Thí nghiệm 5.2 — Scan N](#83-thí-nghiệm-52--scan-n-tìm-kích-thước-chạy-2-3-phút)
   - [Thí nghiệm 5.3 — Granularity / Load Balancing](#84-thí-nghiệm-53--granularity--load-balancing)
   - [Thí nghiệm 5.4 — Speedup](#85-thí-nghiệm-54--speedup)
   - [Cấu trúc CSV output](#86-cấu-trúc-csv-output)
9. [Kết quả mẫu](#9-kết-quả-mẫu)
10. [Xử lý lỗi thường gặp](#10-xử-lý-lỗi-thường-gặp)
11. [Các hàm MPI được dùng](#11-các-hàm-mpi-được-dùng)

---

## 1. Cấu trúc project

```
bfs-project/
├── src/
│   ├── main_seq.c          ← Entry point BFS tuần tự (baseline)
│   ├── main_hybrid.c       ← Entry point BFS hybrid (MPI + OpenMP)
│   ├── bfs_sequential.c/h  ← BFS queue-based tiêu chuẩn
│   ├── bfs_hybrid.c/h      ← Direction-optimizing + MPI + OpenMP
│   │                          (đã bổ sung đo compute/comm time per-rank)
│   ├── graph.c/h           ← R-MAT generator, định dạng CSR
│   └── utils.c/h           ← Timer, MTEPS, verify
├── scripts/
│   └── run_benchmark.sh    ← Benchmark tự động 4 thí nghiệm, xuất CSV
├── results/                ← Output CSV (tự tạo khi chạy benchmark)
├── hostfile                ← Danh sách nodes cluster (điền trước khi chạy)
├── Makefile
└── README.md
```

---

## 2. Thuật toán

### R-MAT Graph Generator

Sinh đồ thị ngẫu nhiên theo mô hình **R-MAT** (Recursive MATrix),
chuẩn Graph500. Đồ thị vô hướng, lưu dạng **CSR** (Compressed Sparse Row).

```
Ma trận kề được chia đệ quy thành 4 góc với xác suất a, b, c, d:
  ┌──────┬──────┐
  │  a   │  b   │    a=0.57, b=0.19
  │      │      │    c=0.19, d=0.05
  ├──────┼──────┤    (a+b+c+d = 1)
  │  c   │  d   │
  └──────┴──────┘
Mỗi cạnh được đặt vào một góc ngẫu nhiên, lặp lại log2(n) lần.
→ Tạo ra đồ thị có phân phối bậc power-law (giống mạng xã hội thực tế)
→ Đỉnh chỉ số nhỏ có xu hướng bậc cao hơn (do a >> d) — ảnh hưởng trực
   tiếp đến cân bằng tải giữa các MPI rank (xem mục 8.4)
```

### Direction-Optimizing BFS (Beamer et al. 2012)

BFS thông thường luôn duyệt **top-down** (từ frontier ra hàng xóm).
Khi frontier rất lớn, cách này lãng phí vì kiểm tra nhiều cạnh đã thăm.

Thuật toán tự động chuyển hướng theo từng level:

```
TOP-DOWN  : frontier → xét hàng xóm của mỗi đỉnh trong frontier
BOTTOM-UP : mỗi đỉnh CHƯA THĂM → tìm 1 hàng xóm trong frontier
            (dừng ngay khi tìm thấy → tiết kiệm hơn khi frontier rộng)

Chuyển sang BOTTOM-UP khi: frontier_edges > unvisited_edges / ALPHA (=14)
Quay lại TOP-DOWN khi:     frontier_size  < num_vertices    / BETA  (=24)
```

### Kiến trúc song song

```
┌─────────────────────────────────────────────────────┐
│                  MPI RANK 0                          │
│  ┌──────────┐  ┌──────────┐                          │
│  │ Thread 0 │  │ Thread 1 │  ← OpenMP threads        │
│  └──────────┘  └──────────┘                          │
├─────────────────────────────────────────────────────┤
│                  MPI RANK 1                          │
│  ┌──────────┐  ┌──────────┐                          │
│  │ Thread 0 │  │ Thread 1 │                          │
│  └──────────┘  └──────────┘                          │
├─────────────────────────────────────────────────────┤
│                  MPI RANK 2, 3, ...                  │
└─────────────────────────────────────────────────────┘

Phân vùng đỉnh (1D Vertex Partition):
  Rank r chịu trách nhiệm đỉnh [n*r/P, n*(r+1)/P)
  → Static 1D block partition (đều theo số đỉnh, không theo số cạnh)

Đồng bộ cuối mỗi level (blocking, level-synchronous):
  MPI_Allreduce (SUM) → tổng frontier_edges (quyết định top-down/bottom-up)
  MPI_Allreduce (MAX) → merge dist[] toàn cục
  MPI_Allreduce (BOR) → merge frontier bitmap
```

---

## 3. Cài đặt dependencies

Thực hiện trên **tất cả** các máy tham gia cluster:

```bash
sudo apt update && sudo apt install -y \
    openmpi-bin \
    openmpi-common \
    libopenmpi-dev \
    openssh-server \
    gcc \
    make
```

Kiểm tra version — phải **giống nhau** trên tất cả nodes:

```bash
mpirun --version
# Ví dụ: mpirun (Open MPI) 4.1.6

mpicc --version
# Ví dụ: gcc (Ubuntu 13.2.0) 13.2.0

# Kiểm tra OpenMP (có sẵn trong GCC >= 4.9)
echo "#include <omp.h>
int main(){return 0;}" | gcc -fopenmp -x c - -o /tmp/t && echo "OpenMP OK"
```

> ⚠️ **Quan trọng:** Version OpenMPI phải **giống hệt nhau** trên tất cả nodes.
> Nếu khác version, chương trình có thể crash hoặc treo.

---

## 4. Build project

```bash
make          # Build cả bfs_seq và bfs_hybrid
make clean    # Xóa binary và results/*.csv
```

Makefile cũng có các target tiện dụng để test nhanh:

```bash
make test_seq    # Chạy sequential 500k đỉnh
make test_local  # Chạy hybrid 1 rank + 4 thread + verify
make verify      # Verify correctness với graph nhỏ (100k đỉnh)
```

---

## 5. Chạy trên 1 máy (local)

### BFS tuần tự (baseline)

```bash
./bfs_seq <num_vertices> <scale_factor> <seed>

# Ví dụ: 1 triệu đỉnh, bậc trung bình 16, seed 42
./bfs_seq 1000000 16 42
```

### BFS hybrid — 1 máy

```bash
OMP_NUM_THREADS=<threads> mpirun -np <ranks> \
    ./bfs_hybrid <num_vertices> <scale_factor> <seed> [options]

# Ví dụ: 4 rank × 2 thread = 8 cores tổng
OMP_NUM_THREADS=2 mpirun -np 4 ./bfs_hybrid 1000000 16 42

# Thêm --verify để so sánh kết quả với sequential
OMP_NUM_THREADS=2 mpirun -np 4 ./bfs_hybrid 1000000 16 42 --verify

# Xuất kết quả ra CSV (dùng cho thí nghiệm 5.2, 5.4)
OMP_NUM_THREADS=2 mpirun -np 4 ./bfs_hybrid 1000000 16 42 \
    --csv results/summary.csv

# Xuất thêm per-rank breakdown (dùng cho thí nghiệm 5.3)
OMP_NUM_THREADS=2 mpirun -np 4 ./bfs_hybrid 1000000 16 42 \
    --csv results/summary.csv \
    --csv-ranks results/ranks.csv
```

> `ranks × threads ≤ số core vật lý` để tránh oversubscribe.
> Nếu máy ít core, thêm `--oversubscribe` để ép chạy (kết quả đúng, hiệu năng thấp hơn).

---

## 6. Thiết lập Cluster — Hướng dẫn chi tiết

Phần này hướng dẫn cấu hình **mỗi VM chạy trên 1 máy vật lý riêng**,
kết nối qua mạng LAN, để chạy BFS phân tán thật sự.

### Topology trong hướng dẫn này

```
┌────────────────────────────────────────────────────────────┐
│                     LAN / Switch / Router                  │
│                                                            │
│  ┌─────────────────┐   ┌──────────────┐  ┌──────────────┐ │
│  │  Máy vật lý 1   │   │ Máy vật lý 2 │  │ Máy vật lý 3 │ │
│  │  ┌───────────┐  │   │ ┌──────────┐ │  │ ┌──────────┐ │ │
│  │  │   VM      │  │   │ │   VM     │ │  │ │   VM     │ │ │
│  │  │  node01   │  │   │ │  node02  │ │  │ │  node03  │ │ │
│  │  │.1.101     │  │   │ │ .1.102   │ │  │ │ .1.103   │ │ │
│  │  │ (master)  │  │   │ │ (worker) │ │  │ │ (worker) │ │ │
│  │  └───────────┘  │   │ └──────────┘ │  │ └──────────┘ │ │
│  └─────────────────┘   └──────────────┘  └──────────────┘ │
└────────────────────────────────────────────────────────────┘
```

> Thay IP và hostname theo thực tế của bạn.
> Mỗi VM phải dùng **Bridged Adapter** trong VirtualBox để lấy IP thật trên LAN.

---

### 6.1. Yêu cầu phần cứng / mạng

- Tất cả VM phải **cùng subnet** (ping được nhau)
- OS: **Ubuntu 22.04+ / 24.04** (khuyến nghị)
- Mỗi VM: tối thiểu 2 vCPU, 1GB RAM, 15GB disk
- Network Adapter trong VirtualBox: **Bridged Adapter** (không phải NAT)
- Cổng cần mở nếu có firewall: **SSH (22)**, **NFS (2049)**

Kiểm tra ping trước khi làm bất cứ thứ gì:

```bash
# Trên node01, ping sang các node khác
ping -c 3 192.168.1.102
ping -c 3 192.168.1.103

# Nếu không ping được → kiểm tra lại Bridged Adapter trong VirtualBox
```

---

### 6.2. Cài OpenMPI trên tất cả nodes

Chạy lệnh sau trên **TỪNG VM** (node01, node02, node03, ...):

```bash
sudo apt update && sudo apt install -y \
    openmpi-bin \
    openmpi-common \
    libopenmpi-dev \
    nfs-common \
    openssh-server \
    gcc \
    make

# Kiểm tra version — phải giống nhau trên tất cả
mpirun --version
# Ví dụ: mpirun (Open MPI) 4.1.6
```

---

### 6.3. Tạo user chung và cấu hình SSH không password

MPI dùng SSH để spawn process trên các node worker.
Phải cấu hình để **node01 SSH sang các node khác mà không cần nhập password**.

#### Bước 1 — Tạo user `mpiuser` trên TẤT CẢ các VM

```bash
# Chạy trên TỪNG VM
sudo adduser mpiuser
# Nhập password, Enter bỏ qua các trường còn lại

# Tuỳ chọn: thêm sudo để tiện cài gói
sudo usermod -aG sudo mpiuser
```

> Tên user phải **giống nhau** trên tất cả VM.

#### Bước 2 — Tạo SSH key trên node01 (master)

```bash
# Chuyển sang user mpiuser trên node01
su - mpiuser

# Tạo SSH key — nhấn Enter 3 lần, không đặt passphrase
ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa

# Kiểm tra
ls ~/.ssh/
# Phải thấy: id_rsa  id_rsa.pub
```

#### Bước 3 — Copy public key sang tất cả nodes (kể cả chính node01)

```bash
# Vẫn trong session mpiuser trên node01
ssh-copy-id mpiuser@node01   # Localhost cũng cần — MPI đôi khi SSH vào chính nó
ssh-copy-id mpiuser@node02
ssh-copy-id mpiuser@node03
# Lệnh này hỏi password mpiuser trên node đích — đây là lần cuối cùng cần nhập
```

#### Bước 4 — Kiểm tra SSH không password

```bash
# Từ node01, SSH sang từng node — KHÔNG được hỏi password
ssh mpiuser@node02 "hostname && echo SSH OK"
ssh mpiuser@node03 "hostname && echo SSH OK"

# Kết quả mong đợi:
# node02
# SSH OK
```

Nếu vẫn hỏi password, kiểm tra trên node worker:

```bash
# Trên node02, node03, ...
ls -la ~/.ssh/authorized_keys
# Phải tồn tại và chứa public key của node01

chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
```

#### Bước 5 (tuỳ chọn) — Bỏ qua host key check

Thêm vào `~/.ssh/config` trên node01 để tránh hỏi "Are you sure?" lần đầu:

```bash
cat >> ~/.ssh/config << 'EOF'
Host node01 node02 node03 node04
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
EOF
```

---

### 6.4. Cấu hình NFS chia sẻ thư mục code

NFS cho phép tất cả worker đọc binary từ node01, không cần copy thủ công.
Build 1 lần trên node01 → tất cả nodes dùng được ngay.

#### Trên node01 — Cài và cấu hình NFS server

```bash
sudo apt install -y nfs-kernel-server

# Tạo thư mục project
mkdir -p /home/mpiuser/bfs-project
sudo chown -R mpiuser:mpiuser /home/mpiuser

# Thêm vào /etc/exports
echo "/home/mpiuser  192.168.1.0/24(rw,sync,no_subtree_check,no_root_squash)" \
    | sudo tee -a /etc/exports

# Apply và khởi động
sudo exportfs -a
sudo exportfs -v    # Kiểm tra đang export gì
sudo systemctl enable nfs-kernel-server
sudo systemctl start nfs-kernel-server
sudo systemctl status nfs-kernel-server  # Phải thấy "active (running)"
```

Giải thích các option trong `/etc/exports`:
- `rw` — cho phép đọc và ghi
- `sync` — ghi đồng bộ, an toàn hơn
- `no_subtree_check` — tắt subtree check, tăng performance
- `no_root_squash` — root trên client giữ quyền root

#### Trên node02, node03, ... — Mount NFS

```bash
# Tạo mount point cùng đường dẫn với node01
sudo mkdir -p /home/mpiuser

# Mount thử
sudo mount 192.168.1.101:/home/mpiuser /home/mpiuser

# Kiểm tra
df -h | grep mpiuser
ls /home/mpiuser/bfs-project/   # Phải thấy file từ node01
```

#### Cấu hình mount tự động khi khởi động

```bash
# Thêm vào /etc/fstab trên từng worker
echo "192.168.1.101:/home/mpiuser  /home/mpiuser  nfs  defaults,_netdev  0  0" \
    | sudo tee -a /etc/fstab

# Test fstab
sudo umount /home/mpiuser
sudo mount -a               # Mount lại theo fstab
df -h | grep mpiuser        # Phải mount được
```

---

### 6.5. Cấu hình /etc/hosts

Thêm hostname của tất cả nodes vào `/etc/hosts` trên **TỪNG VM**
để dùng hostname thay IP:

```bash
sudo tee -a /etc/hosts << 'EOF'
192.168.1.101   node01
192.168.1.102   node02
192.168.1.103   node03
192.168.1.104   node04
EOF
```

Kiểm tra:

```bash
ping -c 3 node02    # Từ node01
ping -c 3 node01    # Từ node02
```

---

### 6.6. Tạo hostfile

Trên **node01**, tạo file `~/bfs-project/hostfile`:

```bash
su - mpiuser
nano ~/bfs-project/hostfile
```

Nội dung — đặt `slots` bằng **số vCPU của VM** (dùng `nproc` để kiểm tra):

```
# 3 node, mỗi node 4 vCPU → tổng 12 slots
node01 slots=4
node02 slots=4
node03 slots=4
```

```
# 4 node, số vCPU khác nhau
node01 slots=8    # máy 8-core
node02 slots=4    # máy 4-core
node03 slots=4    # máy 4-core
# Tổng: -np tối đa 16
```

> `slots` xác định số MPI process tối đa trên node đó.
> Script benchmark (mục 8) đọc giá trị `TOTAL_CORES` từ cấu hình — hãy
> đặt khớp với tổng `slots` trong hostfile.

---

### 6.7. Test kết nối cluster

Trước khi chạy BFS, kiểm tra MPI có spawn đúng trên các node không:

```bash
# Test: mỗi process in hostname của nó
mpirun -np 6 --hostfile hostfile hostname

# Kết quả mong đợi (thứ tự có thể khác):
# node01
# node01
# node02
# node02
# node03
# node03
```

```bash
# Test hello world MPI + OpenMP
cat > /tmp/hello_mpi.c << 'EOF'
#include <stdio.h>
#include <mpi.h>
#include <omp.h>
int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size; char host[256];
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    gethostname(host, sizeof(host));
    #pragma omp parallel
    {
        #pragma omp critical
        printf("Rank %d/%d | Host: %s | Thread %d/%d\n",
               rank, size, host,
               omp_get_thread_num(), omp_get_num_threads());
    }
    MPI_Finalize();
    return 0;
}
EOF
mpicc -fopenmp /tmp/hello_mpi.c -o /tmp/hello_mpi

OMP_NUM_THREADS=2 mpirun -np 6 --hostfile hostfile /tmp/hello_mpi

# Kết quả mong đợi:
# Rank 0/6 | Host: node01 | Thread 0/2
# Rank 0/6 | Host: node01 | Thread 1/2
# Rank 2/6 | Host: node02 | Thread 0/2
# ...
```

Nếu thấy tất cả nodes xuất hiện → cluster sẵn sàng.

---

### 6.8. Chạy chương trình trên cluster

#### Quy trình đầy đủ từ đầu

```bash
# 1. Trên node01 — vào thư mục project
su - mpiuser
cd ~/bfs-project

# 2. Build (chỉ cần làm trên node01, NFS tự sync sang worker)
make

# 3. Chạy sequential lấy baseline
./bfs_seq 4000000 16 42

# 4. Chạy BFS hybrid trên cluster
# 3 node × 4 rank/node = 12 rank, mỗi rank 1 thread
OMP_NUM_THREADS=1 mpirun -np 12 --hostfile hostfile \
    ./bfs_hybrid 4000000 16 42

# 5. Kèm --verify để kiểm tra kết quả đúng/sai
OMP_NUM_THREADS=1 mpirun -np 12 --hostfile hostfile \
    ./bfs_hybrid 1000000 16 42 --verify

# 6. Xuất CSV để vẽ biểu đồ báo cáo
OMP_NUM_THREADS=1 mpirun -np 12 --hostfile hostfile \
    ./bfs_hybrid 4000000 16 42 \
    --csv results/summary.csv \
    --csv-ranks results/ranks.csv
```

#### Các tuỳ chọn hữu ích của mpirun

```bash
# 1 process mỗi node (test topology)
mpirun -np 3 --hostfile hostfile \
    --map-by node ./bfs_hybrid 1000000 16 42

# Bind process theo core (tăng performance trên cluster thật)
mpirun -np 12 --hostfile hostfile \
    --bind-to core ./bfs_hybrid 4000000 16 42

# Xem chi tiết process được spawn ở đâu
mpirun -np 12 --hostfile hostfile \
    --display-map ./bfs_hybrid 1000000 16 42

# Ép chạy dù không đủ slot (khi test oversubscribe)
mpirun -np 24 --hostfile hostfile \
    --oversubscribe ./bfs_hybrid 1000000 16 42
```

---

## 7. Tham số dòng lệnh

### `bfs_hybrid` — tất cả tham số

```
mpirun -np <P> ./bfs_hybrid <num_vertices> <scale_factor> <seed> [options]
```

| Tham số / Flag | Bắt buộc | Ý nghĩa | Ví dụ |
|---|---|---|---|
| `num_vertices` | ✓ | Số đỉnh (làm tròn lên lũy thừa 2) | `4000000` → thực tế `4194304` |
| `scale_factor` | ✓ | Bậc trung bình mỗi đỉnh (~mật độ cạnh) | `16` |
| `seed` | ✓ | Random seed — giữ cố định để so sánh | `42` |
| `--verify` | — | So sánh dist[] với BFS tuần tự | (flag) |
| `--csv <file>` | — | Ghi 1 dòng tổng hợp vào CSV (append) | `results/summary.csv` |
| `--csv-ranks <file>` | — | Ghi per-rank breakdown vào CSV (append) | `results/ranks.csv` |

### Biến môi trường

| Biến | Ý nghĩa | Gợi ý |
|---|---|---|
| `OMP_NUM_THREADS` | Số OpenMP thread / MPI rank | `= số vCPU / số rank trên node` |

### Hằng số thuật toán (trong `bfs_hybrid.h`)

| Hằng | Giá trị | Ý nghĩa |
|---|---|---|
| `BFS_ALPHA` | `14` | Ngưỡng chuyển sang bottom-up |
| `BFS_BETA`  | `24` | Ngưỡng quay lại top-down |

---

## 8. Benchmark tự động — Thí nghiệm báo cáo

Script `scripts/run_benchmark.sh` tự động hoá 4 thí nghiệm trong báo cáo,
xuất kết quả ra `results/*.csv` theo từng timestamp.

### 8.1. Cấu hình script trước khi chạy

Mở `scripts/run_benchmark.sh` và chỉnh vùng **CẤU HÌNH** ở đầu file:

```bash
# ---- Thông tin cluster ---------------------------------------------------
TOTAL_CORES=12      # Tổng số nhân vật lý toàn cluster (= tổng slots trong hostfile)
                    # Ví dụ: 3 máy × 4 nhân = 12
HOSTFILE="hostfile" # Đường dẫn hostfile (để rỗng "" nếu chỉ chạy 1 máy)
OVERSUBSCRIBE=0     # 0 = tắt (chạy thật trên cluster), 1 = bật (debug local)

# ---- Tham số đồ thị ------------------------------------------------------
SCALE=16            # bậc trung bình — GIỮ NGUYÊN xuyên suốt mọi thí nghiệm
SEED=42             # seed cố định — GIỮ NGUYÊN xuyên suốt mọi thí nghiệm

# ---- Mục 5.2: danh sách N để quét ----------------------------------------
SCAN_N_LIST=(500000 1000000 2000000 4000000 8000000 16000000)

# ---- Mục 5.3: N0 bạn chọn được sau khi chạy scan-n -----------------------
GRANULARITY_N=4000000    # ← ĐIỀN GIÁ TRỊ N0 thật sau khi chạy thí nghiệm 5.2

# SPEEDUP_N tự động = 2 × GRANULARITY_N (mục 5.4) — không cần sửa tay
```

> **Lưu ý quan trọng về `SEED`:** Giữ cùng 1 giá trị seed xuyên suốt
> toàn bộ 4 thí nghiệm để các lần chạy dùng cùng 1 đồ thị logic và có
> thể so sánh công bằng với nhau (chỉ khác N/P, không khác cấu trúc đồ thị).

---

### 8.2. Thí nghiệm 5.1 — Verify Correctness

**Mục tiêu:** Xác nhận `dist[]` của bản hybrid giống hệt bản tuần tự
trên nhiều cấu hình N/P khác nhau, chứng minh tính đúng đắn ổn định.

```bash
chmod +x scripts/run_benchmark.sh
./scripts/run_benchmark.sh verify
```

Script chạy `--verify` với 5 cấu hình (N nhỏ/lớn, P khác nhau).
Kết quả in ra console và lưu log đầy đủ vào `results/verify_<timestamp>.log`.

Kiểm tra nhanh: mỗi dòng phải có `PASSED ✓`, không có `FAILED ✗`.

---

### 8.3. Thí nghiệm 5.2 — Scan N (tìm kích thước chạy 2-3 phút)

**Mục tiêu:** Tìm N₀ sao cho thời gian BFS (không tính sinh đồ thị) nằm
trong khoảng 2-3 phút, dùng P = TOTAL_CORES cố định.

```bash
./scripts/run_benchmark.sh scan-n
# CSV: results/scan_n_<timestamp>.csv
```

**Thời gian chương trình vs thời gian BFS:** Cột `gen_ms` (sinh đồ thị) và
`bfs_time_ms` (BFS thật) được tách riêng trong CSV. Mục 5.2 đo `bfs_time_ms`,
không phải `gen_ms` (sinh đồ thị chạy tuần tự trên mọi rank, không hưởng lợi
từ song song hoá — không phải phần thuật toán đang đánh giá).

**Sau khi chạy xong:**
1. Mở file CSV, tìm N sao cho `bfs_time_ms` nằm trong 120000–180000 ms (2-3 phút).
2. Nếu chưa đạt: thêm giá trị N lớn hơn vào `SCAN_N_LIST` trong script, chạy lại.
3. Sau khi chọn được N₀: điền vào `GRANULARITY_N` trong script.

**Biểu đồ cần vẽ** (trục X = `num_vertices`, trục Y = thời gian ms):
- Đường 1: `bfs_time_ms` (có thời gian truyền thông)
- Đường 2: `bfs_time_ms - comm_ms` (không có thời gian truyền thông, rank 0)

---

### 8.4. Thí nghiệm 5.3 — Granularity / Load Balancing

**Mục tiêu:** Với N₀ và P = TOTAL_CORES cố định, kiểm tra tải có cân
bằng giữa các tiến trình không (ngưỡng đề bài: lệch > 25% = mất cân bằng).

```bash
./scripts/run_benchmark.sh granular
# CSV: results/granularity_ranks_<timestamp>.csv
#      results/granularity_summary_<timestamp>.csv
```

Kết quả in thẳng ra console bao gồm bảng per-rank:

```
[HYBRID] Per-rank breakdown:
[HYBRID]   rank      vertices          edges  compute(ms)  comm(ms)  total(ms)
[HYBRID]   0            65536        3267123         8.28     12.89      21.16
[HYBRID]   1            65536        1088780         2.86     17.30      20.16
...
[HYBRID]   Load imbalance (max-min)/max = 5.1% (<= 25% -> tam on)
```

**Ý nghĩa các cột:**
- `edges` — tổng số cạnh của dải đỉnh rank đó sở hữu (không đều nhau do R-MAT)
- `compute_ms` — thời gian rank đó thực tế tính toán (BFS steps + đếm cục bộ)
- `comm_ms` — thời gian rank đó "ở trong" các lệnh `MPI_Allreduce`, bao gồm
  cả thời gian **chờ** rank chậm nhất tới điểm đồng bộ (= thời gian rảnh do lệch tải)
- `(max-min)/max` — % lệch tải tự động tính và in ra

**Biểu đồ cần vẽ:** Stacked bar chart — 1 cột = 1 rank, 2 màu xếp chồng
(`compute_ms` + `comm_ms`). Lọc file CSV theo `num_vertices=N₀`, `num_ranks=TOTAL_CORES`.

**Nếu lệch > 25%:** Lý do là partition hiện tại chia đều theo số đỉnh (không
theo số cạnh), trong khi R-MAT phân phối bậc lệch mạnh (đỉnh chỉ số nhỏ bậc cao
hơn). Hướng cải tiến: chia theo số cạnh thay vì số đỉnh — đảm bảo mỗi rank
nhận xấp xỉ `|E|/P` cạnh thay vì `|V|/P` đỉnh.

---

### 8.5. Thí nghiệm 5.4 — Speedup

**Mục tiêu:** Với N = 2×N₀ cố định, quét P = 1, 2, 4, ..., 2×TOTAL_CORES.
Vẽ biểu đồ thời gian và speedup, so sánh với đường lý tưởng Speedup = P.

```bash
./scripts/run_benchmark.sh speedup
# CSV: results/speedup_<timestamp>.csv
```

**Biểu đồ 1** (trục X = `num_ranks`, trục Y = thời gian ms):
- Đường 1: `bfs_time_ms` (có communication)
- Đường 2: `bfs_time_ms - comm_ms` (không có communication)

**Biểu đồ 2** (trục X = `num_ranks`, trục Y = speedup):
- Speedup(P) = `bfs_time_ms(P=1)` / `bfs_time_ms(P)` (tính từ CSV, dòng P=1 lấy làm mốc)
- Đường lý tưởng: Speedup = P (linear speedup)

> Kỳ vọng: speedup không tuyến tính hoàn toàn khi P lớn, vì `Allreduce(dist[])`
> truyền O(n) dữ liệu mỗi level bất kể P — overhead communication tăng tỷ lệ
> khi thời gian tính toán/rank giảm. Đây là minh hoạ thực nghiệm của định luật Amdahl.

---

### 8.6. Cấu trúc CSV output

#### `--csv <file>` — 1 dòng / lần chạy (dùng cho 5.2, 5.4)

```
num_vertices, scale, seed, num_ranks, omp_threads, total_cores,
gen_ms, bfs_time_ms, compute_ms, comm_ms, num_levels, visited_edges,
mteps, seq_time_ms, speedup, verify_errors
```

- `compute_ms`, `comm_ms` là của **rank 0** (không phải toàn cluster)
- `seq_time_ms`, `speedup`, `verify_errors` — để trống nếu không chạy `--verify`
- File tự tạo header nếu chưa tồn tại, append nếu đã tồn tại

#### `--csv-ranks <file>` — 1 dòng / rank / lần chạy (dùng cho 5.3)

```
num_vertices, scale, seed, num_ranks, omp_threads, rank,
local_vertices, local_edges, compute_ms, comm_ms, total_ms
```

- Lọc theo `(num_vertices, num_ranks)` khi vẽ biểu đồ load-balancing

---

## 9. Kết quả mẫu

### BFS tuần tự

```
[SEQ] Generating R-MAT graph: 500000 vertices, scale=16, seed=42
[SEQ] Graph generated in 4868.78 ms

[SEQ] BFS from source vertex 42
[SEQ] Graph: 524288 vertices, 15480484 edges
[SEQ] Visited: 335349 vertices, 15480274 edges
[SEQ] Time: 42.34 ms
[SEQ] MTEPS: 365.65
```

### BFS hybrid (4 rank × 2 thread = 8 cores) với per-rank breakdown

```
[HYBRID] ============================================
[HYBRID] BFS Hybrid (MPI + OpenMP + Direction-Opt)
[HYBRID] MPI ranks   : 4
[HYBRID] OMP threads : 2 per rank
[HYBRID] Total cores : 8
[HYBRID] Alpha       : 14
[HYBRID] Beta        : 24
[HYBRID] ============================================

[HYBRID] Generating R-MAT graph: 500000 vertices, scale=16, seed=42
[HYBRID] BFS from source vertex 42

[HYBRID]   Level  1: top-down  (frontier=1,      fe=9888,     ue=15480484)
[HYBRID]   Level  2: bottom-up (frontier=2472,   fe=10882620, ue=15470596)
[HYBRID]   Level  3: bottom-up (frontier=224067, fe=50015036, ue=2966837)
[HYBRID]   Level  4: bottom-up (frontier=108000, fe=1010272,  ue=2714269)
[HYBRID]   Level  5: top-down  (frontier=808,    fe=3276,     ue=2713450)
[HYBRID]   Level  6: top-down  (frontier=1,      fe=4,        ue=2710174)

[HYBRID] ============================================
[HYBRID] Graph    : 524288 vertices, 15480484 edges
[HYBRID] Visited  : 335349 vertices, 12770314 edges
[HYBRID] Levels   : 6
[HYBRID] Time     : 40.62 ms  (compute: 31.40 ms | comm: 8.50 ms, rank 0)
[HYBRID] MTEPS    : 314.35

[HYBRID] Per-rank breakdown:
[HYBRID]   rank      vertices          edges  compute(ms)     comm(ms)    total(ms)
[HYBRID]   0           131072       8764201        31.40         8.50        39.90
[HYBRID]   1           131072       2922800        12.10        26.80        38.90
[HYBRID]   2           131072       2924000        13.20        25.60        38.80
[HYBRID]   3           131072        869483        18.40        20.30        38.70
[HYBRID]   Load imbalance (max-min)/max = 2.8% (<= 25% -> tam on)

[HYBRID] Sequential : 69.14 ms | MTEPS: 223.90
[HYBRID] Speedup    : 1.70x
[HYBRID] Verify     : PASSED ✓
[HYBRID] ============================================
[HYBRID] Summary appended to results/summary.csv
[HYBRID] Per-rank stats appended to results/ranks.csv
```

### Giải thích output

| Trường | Ý nghĩa |
|---|---|
| `top-down` / `bottom-up` | Hướng duyệt BFS của level đó |
| `frontier` | Số đỉnh đang trong frontier |
| `fe` | Frontier edges — tổng cạnh của các đỉnh frontier |
| `ue` | Unvisited edges — số cạnh chưa được duyệt |
| `compute(ms)` | Thời gian tính toán thực sự của rank đó (qua mọi level) |
| `comm(ms)` | Thời gian trong `MPI_Allreduce` của rank đó (gồm cả thời gian chờ) |
| `(max-min)/max` | % lệch tải giữa rank nhanh nhất và chậm nhất |
| `MTEPS` | Mega Traversed Edges Per Second — thước đo chuẩn Graph500 |
| `Speedup` | Thời gian sequential / thời gian hybrid |
| `Verify` | `PASSED ✓` = kết quả hybrid giống hệt sequential |

---

## 10. Xử lý lỗi thường gặp

**`ssh: connect to host node02 port 22: Connection refused`**
```bash
# Kiểm tra SSH service trên node worker
sudo systemctl status ssh
sudo systemctl enable --now ssh
```

**`Permission denied (publickey)`**
```bash
# Key chưa được copy đúng, làm lại bước 6.3
ssh-copy-id mpiuser@node02

# Kiểm tra quyền thư mục trên worker
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
```

**`There are not enough slots available`**
```bash
# Tăng slots trong hostfile, hoặc giảm -np
node01 slots=4    # tăng lên theo số vCPU thật

# Hoặc thêm --oversubscribe (chỉ dùng khi debug, không dùng đo hiệu năng)
mpirun -np 8 --hostfile hostfile --oversubscribe ./bfs_hybrid ...
```

**`No such file or directory` khi chạy trên worker**
```bash
# NFS chưa mount — kiểm tra trên worker:
ls ~/bfs-project/bfs_hybrid

# Nếu không thấy → mount lại NFS (bước 6.4)
sudo mount 192.168.1.101:/home/mpiuser /home/mpiuser
```

**IP VM thay đổi sau khi restart**
```bash
# Đặt IP tĩnh trong /etc/netplan/00-installer-config.yaml
sudo nano /etc/netplan/00-installer-config.yaml
```
```yaml
network:
  version: 2
  ethernets:
    enp0s3:                          # Tên card — dùng `ip addr` để xem
      dhcp4: no
      addresses: [192.168.1.101/24]  # Đổi theo từng node
      routes:
        - to: default
          via: 192.168.1.1           # Gateway router
      nameservers:
        addresses: [8.8.8.8]
```
```bash
sudo netplan apply
```

**`Verify FAILED` — kết quả sai**
```bash
# Chạy với graph nhỏ để debug
OMP_NUM_THREADS=1 mpirun -np 2 ./bfs_hybrid 10000 8 42 --verify
```

**Kiểm tra version OpenMPI nhất quán**
```bash
# Chạy trên từng node — phải ra cùng version
mpirun --version
# mpirun (Open MPI) 4.1.6   ← phải giống nhau trên mọi node
```

**Script benchmark báo lỗi `cannot open csv file`**
```bash
# Thư mục results/ chưa tồn tại — tạo thủ công hoặc chạy make trước
mkdir -p results
```

---

## 11. Các hàm MPI được dùng

| Hàm | Mục đích |
|---|---|
| `MPI_Init_thread` | Khởi tạo MPI với hỗ trợ thread (`MPI_THREAD_FUNNELED`) |
| `MPI_Comm_rank` | Lấy rank (ID) của process hiện tại |
| `MPI_Comm_size` | Lấy tổng số MPI process đang chạy |
| `MPI_Barrier` | Đồng bộ — tất cả rank phải đến đây mới tiếp tục |
| `MPI_Allreduce (SUM)` | Tổng hợp `frontier_edges` toàn cục đầu mỗi level |
| `MPI_Allreduce (MAX)` | Merge `dist[]` toàn cục cuối mỗi bước BFS |
| `MPI_Allreduce (BOR)` | Merge frontier bitmap giữa các rank |
| `MPI_Gather` | Thu thập `RankStats` từ mọi rank về rank 0 (đo load balancing) |
| `MPI_Abort` | Dừng khẩn cấp tất cả process khi có lỗi nghiêm trọng |
| `MPI_Finalize` | Kết thúc MPI, giải phóng tài nguyên |
| `MPI_Wtime` | Đo thời gian wall-clock (dùng để đo compute/comm time per-rank) |