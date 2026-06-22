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
   - [Quy trình khuyến nghị](#82-quy-trình-khuyến-nghị)
   - [gen-graph — Sinh đồ thị 1 lần](#83-gen-graph--sinh-đồ-thị-1-lần-dùng-lại-nhiều-lần)
   - [Thí nghiệm 5.1 — Verify](#84-thí-nghiệm-51--verify-correctness)
   - [Thí nghiệm 5.2 — Scan N](#85-thí-nghiệm-52--scan-n-tìm-kích-thước-chạy-2-3-phút)
   - [Thí nghiệm 5.3 — Granularity / Load Balancing](#86-thí-nghiệm-53--granularity--load-balancing)
   - [Thí nghiệm 5.4 — Speedup](#87-thí-nghiệm-54--speedup)
   - [Cấu trúc CSV output](#88-cấu-trúc-csv-output)
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
│   │                          (bổ sung đo compute/comm time per-rank)
│   ├── graph.c/h           ← R-MAT generator, CSR, save/load binary
│   └── utils.c/h           ← Timer, MTEPS, verify
├── scripts/
│   └── run_benchmark.sh    ← Benchmark tự động 4 thí nghiệm, xuất CSV
├── graphs/                 ← File đồ thị đã sinh (*.bin) — tự tạo khi chạy gen-graph
├── results/                ← Output CSV — tự tạo khi chạy benchmark
├── hostfile                ← Danh sách nodes cluster (điền trước khi chạy)
├── Makefile
└── README.md
```

> **Lưu ý `graphs/`:** Thư mục này lưu đồ thị đã sinh dưới dạng binary CSR.
> Sinh 1 lần, dùng lại nhiều lần — tránh tốn hàng chục phút gen đồ thị
> mỗi lần chạy benchmark. Xem mục 8.3 để biết cách sử dụng.

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
   tiếp đến cân bằng tải giữa các MPI rank (xem mục 8.6)
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
./bfs_seq <num_vertices> <scale_factor> <seed> [options]

# Gen + chạy BFS
./bfs_seq 1000000 16 42

# Gen + lưu đồ thị ra file để dùng lại
./bfs_seq 1000000 16 42 --save-graph graphs/g1M.bin

# Load đồ thị từ file (bỏ qua bước gen, nhanh hơn nhiều)
./bfs_seq 1000000 16 42 --graph graphs/g1M.bin
```

### BFS hybrid — 1 máy

```bash
OMP_NUM_THREADS=<threads> mpirun -np <ranks> \
    ./bfs_hybrid <num_vertices> <scale_factor> <seed> [options]

# Gen + chạy BFS cơ bản (4 rank × 2 thread = 8 cores)
OMP_NUM_THREADS=2 mpirun -np 4 ./bfs_hybrid 1000000 16 42

# Load từ file (tất cả rank load song song — bỏ qua gen)
OMP_NUM_THREADS=2 mpirun -np 4 ./bfs_hybrid 1000000 16 42 \
    --graph graphs/g1M.bin

# Load + verify kết quả
OMP_NUM_THREADS=2 mpirun -np 4 ./bfs_hybrid 1000000 16 42 \
    --graph graphs/g1M.bin --verify

# Load + xuất CSV tổng hợp (mục 5.2, 5.4)
OMP_NUM_THREADS=2 mpirun -np 4 ./bfs_hybrid 1000000 16 42 \
    --graph graphs/g1M.bin --csv results/summary.csv

# Load + xuất cả per-rank breakdown (mục 5.3)
OMP_NUM_THREADS=2 mpirun -np 4 ./bfs_hybrid 1000000 16 42 \
    --graph graphs/g1M.bin \
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

NFS cho phép tất cả worker đọc binary và file đồ thị từ node01 mà không
cần copy thủ công. Build 1 lần + gen graph 1 lần trên node01 → tất cả
nodes dùng được ngay.

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

# Kiểm tra — phải thấy cả binary lẫn thư mục graphs/
df -h | grep mpiuser
ls /home/mpiuser/bfs-project/
ls /home/mpiuser/bfs-project/graphs/    # Thấy *.bin sau khi gen-graph
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
> Script benchmark đọc `TOTAL_CORES` — đặt bằng tổng `slots` trong hostfile.

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

# 3. Gen + lưu đồ thị 1 lần (rank 0 lưu, NFS sync sang worker tự động)
OMP_NUM_THREADS=1 mpirun -np 1 --hostfile hostfile \
    ./bfs_hybrid 4000000 16 42 --save-graph graphs/g4M.bin

# 4. Chạy BFS hybrid load từ file (nhanh, không gen lại)
OMP_NUM_THREADS=1 mpirun -np 12 --hostfile hostfile \
    ./bfs_hybrid 4000000 16 42 --graph graphs/g4M.bin

# 5. Kèm --verify để kiểm tra kết quả đúng/sai
OMP_NUM_THREADS=1 mpirun -np 12 --hostfile hostfile \
    ./bfs_hybrid 1000000 16 42 --graph graphs/g1M.bin --verify

# 6. Xuất CSV để vẽ biểu đồ báo cáo
OMP_NUM_THREADS=1 mpirun -np 12 --hostfile hostfile \
    ./bfs_hybrid 4000000 16 42 --graph graphs/g4M.bin \
    --csv results/summary.csv \
    --csv-ranks results/ranks.csv
```

#### Các tuỳ chọn hữu ích của mpirun

```bash
# Bind process theo core (tăng performance trên cluster thật)
mpirun -np 12 --hostfile hostfile \
    --bind-to core ./bfs_hybrid 4000000 16 42 --graph graphs/g4M.bin

# Xem chi tiết process được spawn ở đâu
mpirun -np 12 --hostfile hostfile \
    --display-map ./bfs_hybrid 1000000 16 42 --graph graphs/g1M.bin

# Ép chạy dù không đủ slot (khi test oversubscribe)
mpirun -np 24 --hostfile hostfile \
    --oversubscribe ./bfs_hybrid 1000000 16 42 --graph graphs/g1M.bin
```

---

## 7. Tham số dòng lệnh

### `bfs_hybrid` và `bfs_seq` — đầy đủ tất cả tham số

```
mpirun -np <P> ./bfs_hybrid <num_vertices> <scale_factor> <seed> [options]
                ./bfs_seq   <num_vertices> <scale_factor> <seed> [options]
```

| Tham số / Flag | Bắt buộc | Ý nghĩa | Ví dụ |
|---|---|---|---|
| `num_vertices` | ✓ | Số đỉnh (làm tròn lên lũy thừa 2) | `4000000` → thực tế `4194304` |
| `scale_factor` | ✓ | Bậc trung bình mỗi đỉnh | `16` |
| `seed` | ✓ | Random seed — giữ cố định để so sánh | `42` |
| `--graph <file>` | — | **Load đồ thị từ file binary** thay vì gen R-MAT. Tất cả MPI rank load song song, độc lập. 3 tham số đầu vẫn phải truyền (dùng để ghi CSV) nhưng không ảnh hưởng cấu trúc đồ thị | `graphs/g4M.bin` |
| `--save-graph <file>` | — | Sau khi gen R-MAT, **lưu đồ thị ra file binary** (chỉ rank 0 ghi). Dùng kết hợp với lần chạy đầu tiên để tạo file cho các lần sau | `graphs/g4M.bin` |
| `--verify` | — | So sánh `dist[]` của hybrid với BFS tuần tự | (flag) |
| `--csv <file>` | — | Ghi 1 dòng tổng hợp vào CSV (append, tạo header nếu chưa có) | `results/summary.csv` |
| `--csv-ranks <file>` | — | Ghi per-rank breakdown vào CSV (1 dòng/rank, append) | `results/ranks.csv` |

> **`--graph` và `--save-graph` không dùng được cùng lúc.** Một là load từ
> file (không gen), một là gen xong thì lưu — hai hướng ngược nhau.

### Format file đồ thị binary (`.bin`)

```
[8 bytes]  magic    = 0x4246535f43535200  ("BFS_CSR\0")
[4 bytes]  n        = num_vertices  (int32_t)
[8 bytes]  m        = num_edges     (int64_t)
[8*(n+1)]  row_ptr  (int64_t × n+1)
[4*m]      adj      (int32_t × m)
```

Kích thước xấp xỉ: `8×(n+1) + 4×m` bytes.
Ví dụ: graph 4M đỉnh, 16 bậc trung bình ≈ 4M×16×4×2 ≈ **512 MB**.

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

Script `scripts/run_benchmark.sh` tự động hoá toàn bộ thí nghiệm báo cáo,
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
SCALE=16            # Bậc trung bình — GIỮ NGUYÊN xuyên suốt mọi thí nghiệm
SEED=42             # Seed cố định — GIỮ NGUYÊN xuyên suốt mọi thí nghiệm

# ---- Mục 5.2: danh sách N để quét ----------------------------------------
SCAN_N_LIST=(500000 1000000 2000000 4000000 8000000 16000000)

# ---- Mục 5.3: N0 bạn chọn được sau khi chạy scan-n -----------------------
GRANULARITY_N=4000000    # ← ĐIỀN GIÁ TRỊ N0 thật sau khi chạy thí nghiệm 5.2

# SPEEDUP_N tự động = 2 × GRANULARITY_N (mục 5.4) — không cần sửa tay
```

> **Lưu ý `SEED`:** Giữ cùng 1 giá trị seed xuyên suốt toàn bộ thí nghiệm.
> Script tự đặt tên file graph theo `g<N>_s<SCALE>_seed<SEED>.bin` —
> đổi seed là đổi file, không tái sử dụng được file cũ.

---

### 8.2. Quy trình khuyến nghị

```
Bước 0: Cấu hình script (TOTAL_CORES, HOSTFILE, SCAN_N_LIST...)
         ↓
Bước 1: gen-graph  → sinh + lưu đồ thị cho mọi N vào graphs/*.bin
         ↓
Bước 2: scan-n     → tìm N₀ (thời gian chạy 2-3 phút)
         ↓
Bước 3: Điền N₀ vào GRANULARITY_N trong script
         ↓
Bước 4: granular   → đo load balancing tại N₀
         ↓
Bước 5: speedup    → đo speedup tại 2×N₀
```

```bash
chmod +x scripts/run_benchmark.sh

./scripts/run_benchmark.sh gen-graph   # Bước 1 — chạy 1 lần duy nhất
./scripts/run_benchmark.sh scan-n      # Bước 2
# ... điền GRANULARITY_N vào script ...
./scripts/run_benchmark.sh granular    # Bước 4
./scripts/run_benchmark.sh speedup     # Bước 5
```

---

### 8.3. gen-graph — Sinh đồ thị 1 lần, dùng lại nhiều lần

**Tại sao cần bước này?**

R-MAT generation tốn nhiều thời gian (vài phút đến hàng chục phút tùy N)
và chạy **tuần tự trên từng rank** (không song song hóa, lặp lại giống hệt
trên mọi rank). Nếu không lưu, mỗi lần chạy benchmark đều phải gen lại →
tốn thời gian vô ích. Lệnh `gen-graph` giải quyết việc này: sinh 1 lần,
lưu vào `graphs/`, tất cả lần chạy sau load từ file (thường chỉ vài giây).

```bash
./scripts/run_benchmark.sh gen-graph
```

Script tự sinh graph cho **tất cả N** trong `SCAN_N_LIST` + `GRANULARITY_N`
+ `SPEEDUP_N` (= 2×GRANULARITY_N). File đặt tại
`graphs/g<N>_s<SCALE>_seed<SEED>.bin`. Nếu file đã tồn tại → bỏ qua, không
gen lại.

Sau khi `gen-graph` xong, tất cả lệnh `scan-n`, `granular`, `speedup` sẽ
**tự động phát hiện file** và thêm `--graph <file>` vào lệnh chạy — không
cần thêm thủ công.

> **Trên cluster với NFS:** gen-graph chỉ cần chạy 1 lần trên node01.
> Thư mục `graphs/` được NFS sync sang worker tự động — mọi rank có thể
> load ngay mà không cần copy thủ công sang từng máy.

---

### 8.4. Thí nghiệm 5.1 — Verify Correctness

**Mục tiêu:** Xác nhận `dist[]` của bản hybrid giống hệt bản tuần tự
trên nhiều cấu hình N/P, chứng minh tính đúng đắn ổn định.

```bash
./scripts/run_benchmark.sh verify
# Log đầy đủ: results/verify_<timestamp>.log
```

Script chạy `--verify` với 5 cấu hình (N nhỏ/lớn, P khác nhau).
Mỗi dòng phải có `PASSED ✓`, không có `FAILED ✗`.

---

### 8.5. Thí nghiệm 5.2 — Scan N (tìm kích thước chạy 2-3 phút)

**Mục tiêu:** Tìm N₀ sao cho thời gian BFS nằm trong khoảng 2-3 phút,
dùng P = TOTAL_CORES cố định.

```bash
./scripts/run_benchmark.sh scan-n
# CSV: results/scan_n_<timestamp>.csv
```

**Quan trọng — thời gian BFS vs thời gian gen:** Cột `gen_ms` (sinh/load
đồ thị) và `bfs_time_ms` (BFS thật sự) được tách riêng trong CSV. Mục 5.2
nhắm vào `bfs_time_ms` — đây là phần thuật toán đang đánh giá. Nếu dùng
`--graph` (đã gen-graph trước), `gen_ms` sẽ rất nhỏ (vài giây load file),
phản ánh đúng chi phí thực tế của BFS.

**Sau khi chạy xong:**
1. Mở file CSV, tìm N sao cho `bfs_time_ms` ≈ 120000–180000 ms (2-3 phút).
2. Nếu chưa đạt: thêm N lớn hơn vào `SCAN_N_LIST`, chạy `gen-graph` thêm
   cho N mới, rồi chạy `scan-n` lại.
3. Điền N₀ vào `GRANULARITY_N` trong script.

**Biểu đồ cần vẽ** (X = `num_vertices`, Y = thời gian ms):
- Đường 1: `bfs_time_ms` (có communication)
- Đường 2: `bfs_time_ms - comm_ms` (không có communication)

---

### 8.6. Thí nghiệm 5.3 — Granularity / Load Balancing

**Mục tiêu:** Với N₀ và P = TOTAL_CORES cố định, kiểm tra tải có cân bằng
không (ngưỡng đề bài: lệch > 25% = mất cân bằng).

```bash
./scripts/run_benchmark.sh granular
# CSV per-rank: results/granularity_ranks_<timestamp>.csv
# CSV summary:  results/granularity_summary_<timestamp>.csv
```

Kết quả in thẳng ra console bao gồm bảng per-rank:

```
[HYBRID] Per-rank breakdown:
[HYBRID]   rank      vertices          edges  compute(ms)  comm(ms)  total(ms)
[HYBRID]   0           131072       8764201        31.40      8.50      39.90
[HYBRID]   1           131072       2922800        12.10     26.80      38.90
[HYBRID]   2           131072       2924000        13.20     25.60      38.80
[HYBRID]   3           131072        869483        18.40     20.30      38.70
[HYBRID]   Load imbalance (max-min)/max = 2.8% (<= 25% -> tam on)
```

**Ý nghĩa các cột:**

| Cột | Ý nghĩa |
|---|---|
| `edges` | Tổng số cạnh của dải đỉnh rank đó sở hữu. Do R-MAT phân phối bậc lệch, đỉnh chỉ số nhỏ (rank 0) thường có nhiều cạnh hơn rõ rệt. |
| `compute_ms` | Thời gian rank đó thực tế tính toán (BFS steps + đếm frontier_edges cục bộ) qua mọi level. |
| `comm_ms` | Thời gian rank đó "ở trong" các lệnh `MPI_Allreduce` — bao gồm cả thời gian **chờ** rank chậm nhất tới điểm đồng bộ. Đây chính là thời gian rảnh do mất cân bằng tải. |
| `(max-min)/max` | % lệch tải tự động tính, so sánh với ngưỡng 25% của đề bài. |

**Biểu đồ cần vẽ:** Stacked bar chart — 1 cột = 1 rank, 2 màu xếp chồng
(`compute_ms` và `comm_ms`). Lọc file CSV theo `num_vertices=N₀`, `num_ranks=TOTAL_CORES`.

**Nếu lệch > 25%:** Partition hiện tại chia đều theo số đỉnh, không theo số
cạnh — trong khi R-MAT có bậc lệch mạnh (đỉnh chỉ số nhỏ bậc cao hơn).
Hướng cải tiến: chia theo số cạnh (edge-balanced partition) thay vì số đỉnh.

---

### 8.7. Thí nghiệm 5.4 — Speedup

**Mục tiêu:** Với N = 2×N₀ cố định, quét P = 1, 2, 4, ..., 2×TOTAL_CORES.
Vẽ biểu đồ thời gian và speedup so với đường lý tưởng Speedup = P.

```bash
./scripts/run_benchmark.sh speedup
# CSV: results/speedup_<timestamp>.csv
```

**Biểu đồ 1** (X = `num_ranks`, Y = thời gian ms):
- Đường 1: `bfs_time_ms` (có communication)
- Đường 2: `bfs_time_ms - comm_ms` (không có communication)

**Biểu đồ 2** (X = `num_ranks`, Y = speedup):
- `Speedup(P) = bfs_time_ms(P=1) / bfs_time_ms(P)` — tính từ CSV, dòng P=1 là mốc
- Đường lý tưởng: Speedup = P (linear speedup)

> Kỳ vọng: speedup không tuyến tính khi P lớn, vì `Allreduce(dist[])`
> truyền O(n) dữ liệu mỗi level bất kể P — đây là minh hoạ thực nghiệm
> của định luật Amdahl.

---

### 8.8. Cấu trúc CSV output

#### `--csv <file>` — 1 dòng / lần chạy (dùng cho 5.2, 5.4)

```
num_vertices, scale, seed, num_ranks, omp_threads, total_cores,
gen_ms, bfs_time_ms, compute_ms, comm_ms, num_levels, visited_edges,
mteps, seq_time_ms, speedup, verify_errors
```

- `gen_ms` = thời gian gen R-MAT (nếu gen) hoặc thời gian load từ file (nếu dùng `--graph`)
- `compute_ms`, `comm_ms` = của **rank 0** (đại diện)
- `seq_time_ms`, `speedup`, `verify_errors` = để trống nếu không chạy `--verify`
- File tự tạo header nếu chưa tồn tại, append nếu đã tồn tại

#### `--csv-ranks <file>` — 1 dòng / rank / lần chạy (dùng cho 5.3)

```
num_vertices, scale, seed, num_ranks, omp_threads, rank,
local_vertices, local_edges, compute_ms, comm_ms, total_ms
```

- Lọc theo `(num_vertices, num_ranks)` khi vẽ biểu đồ load-balancing
- `local_edges` = tổng số cạnh của dải đỉnh rank đó sở hữu (minh chứng R-MAT skew)

---

## 9. Kết quả mẫu

### BFS tuần tự — load từ file

```
[SEQ] Loading graph from 'graphs/g500k.bin' ...
[GRAPH] Loaded graph from 'graphs/g500k.bin' (n=524288, m=15480484)
[GRAPH] Vertices: 524288 | Edges: 15480484 | Avg degree: 29.52 | Min: 0 | Max: 67234
[SEQ] Graph loaded in 1.87 ms

[SEQ] BFS from source vertex 42
[SEQ] Graph: 524288 vertices, 15480484 edges
[SEQ] Visited: 335349 vertices, 15480274 edges
[SEQ] Time: 42.34 ms
[SEQ] MTEPS: 365.65
```

### BFS hybrid — load từ file + verify + per-rank breakdown

```
[HYBRID] Loading graph from 'graphs/g500k.bin' ...
[HYBRID] Graph loaded in 2.10 ms

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
# Tăng slots trong hostfile theo số vCPU thật
node01 slots=4

# Hoặc thêm --oversubscribe (chỉ dùng khi debug)
mpirun -np 8 --hostfile hostfile --oversubscribe ./bfs_hybrid ...
```

**`No such file or directory` khi chạy trên worker**
```bash
# NFS chưa mount — kiểm tra trên worker
ls ~/bfs-project/bfs_hybrid

# Mount lại NFS (bước 6.4)
sudo mount 192.168.1.101:/home/mpiuser /home/mpiuser
```

**`bad magic — không phải file đồ thị hợp lệ`**
```bash
# File .bin bị corrupt hoặc sinh từ phiên bản cũ — xóa và gen lại
rm graphs/g4M.bin
./scripts/run_benchmark.sh gen-graph
```

**`Failed to load graph` khi worker không thấy file**
```bash
# Kiểm tra NFS đã mount và file tồn tại trên worker
ls ~/bfs-project/graphs/

# Nếu không thấy: kiểm tra NFS, hoặc copy thủ công
scp graphs/g4M.bin mpiuser@node02:~/bfs-project/graphs/
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
    enp0s3:
      dhcp4: no
      addresses: [192.168.1.101/24]
      routes:
        - to: default
          via: 192.168.1.1
      nameservers:
        addresses: [8.8.8.8]
```
```bash
sudo netplan apply
```

**`Verify FAILED` — kết quả sai**
```bash
# Debug với graph nhỏ
OMP_NUM_THREADS=1 mpirun -np 2 ./bfs_hybrid 10000 8 42 --verify
```

**Script benchmark báo lỗi `cannot open csv file`**
```bash
mkdir -p results graphs
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
| `MPI_Wtime` | Đo thời gian wall-clock (đo compute/comm time per-rank) |