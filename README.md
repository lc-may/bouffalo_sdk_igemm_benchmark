# mac_int8

Integer GEMM (General Matrix Multiply) benchmark for Bouffalo SDK.

## Support CHIP

|      CHIP        | Remark |
|:----------------:|:------:|
|BL602/BL604       |        |
|BL702/BL704/BL706 |        |
|BL702L/BL704L     |        |
|BL616/BL618       |        |
|BL808             |        |
|BL628             |        |

## Compile

- BL602/BL604

```
make CHIP=bl602 BOARD=bl602dk
```

- BL702/BL704/BL706

```
make CHIP=bl702 BOARD=bl702dk
```

- BL702L/BL704L

```
make CHIP=bl702l BOARD=bl702ldk
```

- BL616/BL618

```
make CHIP=bl616 BOARD=bl616dk
```

- BL808

```
make CHIP=bl808 BOARD=bl808dk CPU_ID=m0
make CHIP=bl808 BOARD=bl808dk CPU_ID=d0
```

- BL628

```
make CHIP=bl628 BOARD=bl628dk
```

## Flash

```
make flash CHIP=chip_name COMX=xxx # xxx is your com name
```

## Shell Commands

### igemm - Integer GEMM Benchmark

Usage:
```
igemm <m> <n> <k> [iterations]
```

Parameters:
- `m`: Number of rows in matrix A and C
- `n`: Number of columns in matrix B and C
- `k`: Number of columns in matrix A / rows in matrix B
- `iterations`: Number of iterations for benchmark (default: 1)

Examples:
```
igemm 16 16 16           # Single run with 16x16 matrices
igemm 32 32 32 100       # 100 iterations for performance measurement
igemm 64 64 64 10        # 10 iterations with 64x64 matrices
```

Note: Timer precision is 1 microsecond. For accurate benchmarking, use sufficient iterations.

## Algorithm

Implements: C = alpha * A^T * B + beta * C

- Input: A (int8), B (int8)
- Output: C (int32)
- Layout: Column major
- Transpose: A is transposed, B is not transposed
