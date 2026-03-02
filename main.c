/*
 * Integer GEMM (General Matrix Multiply) Benchmark
 *
 * Implements: C = alpha * A^T * B + beta * C
 * Input matrices A (int8) and B (int8), output matrix C (int32).
 * Supported: ColMajor layout, A transposed, B not transposed.
 *
 * Timer precision: 1 microsecond (bflb_mtimer_get_time_us)
 */

#include "bflb_mtimer.h"
#include "shell.h"
#include "FreeRTOS.h"
#include "task.h"
#include "board.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ========== Type Definitions ========== */

/* Layout type - for function signature compatibility */
typedef enum {
    kUmdColMajor,
    kUmdRowMajor
} UmdLayout;

/* Transpose type - for function signature compatibility */
typedef enum {
    kUmdNoTrans,
    kUmdTrans
} UmdTranspose;

/* ========== Macros ========== */

/* Color output macros */
#define CLR_RESET   "\033[0m"
#define CLR_RED     "\033[31m"
#define CLR_GREEN   "\033[32m"
#define CLR_CYAN    "\033[36m"

/* ========== GEMM Implementation ========== */

/**
 * @brief Integer GEMM: C = alpha * A^T * B + beta * C
 *
 * @param layout   Matrix layout (only kUmdColMajor supported)
 * @param transa   Transpose flag for A (only kUmdTrans supported)
 * @param transb   Transpose flag for B (only kUmdNoTrans supported)
 * @param m        Rows in A^T and C
 * @param n        Columns in B and C
 * @param k        Columns in A^T / Rows in B
 * @param alpha    Scalar multiplier for A^T * B
 * @param a        Matrix A (int8)
 * @param lda      Leading dimension of A
 * @param b        Matrix B (int8)
 * @param ldb      Leading dimension of B
 * @param beta     Scalar multiplier for C
 * @param c        Matrix C (int32), input and output
 * @param ldc      Leading dimension of C
 */
void UmdIgemmS8S8S32(UmdLayout layout, UmdTranspose transa, UmdTranspose transb,
                     int32_t m, int32_t n, int32_t k, int32_t alpha,
                     const int8_t *a, int32_t lda, const int8_t *b, int32_t ldb,
                     int32_t beta, int32_t *c, int32_t ldc)
{
    int m_index, n_index, k_index;
    for (m_index = 0; m_index < m; ++m_index) {
        for (n_index = 0; n_index < n; ++n_index) {
            int32_t tmp_value = 0;
            for (k_index = 0; k_index < k; ++k_index) {
                tmp_value += alpha * a[m_index * lda + k_index] * b[n_index * ldb + k_index];
            }
            c[n_index * ldc + m_index] = beta * c[n_index * ldc + m_index] + tmp_value;
        }
    }
}

/* ========== V3: Register Tiling (TILE_N=4) + k×4 unroll ========== */

/**
 * @brief V3: Register tiling optimization.
 *        Processes TILE_N=4 output columns per outer mi iteration so that
 *        one a_row load serves 4 independent accumulators simultaneously.
 *        Works for any dimension n >= 4, not just 64.
 */
void UmdIgemmS8S8S32_v3(UmdLayout layout, UmdTranspose transa, UmdTranspose transb,
                         int32_t m, int32_t n, int32_t k, int32_t alpha,
                         const int8_t *a, int32_t lda, const int8_t *b, int32_t ldb,
                         int32_t beta, int32_t *c, int32_t ldc)
{
    int32_t n_block = (n / 4) * 4;

    for (int32_t mi = 0; mi < m; ++mi) {
        const int8_t *a_row = a + mi * k;

        /* Tiled loop: 4 columns of C at once, sharing one a_row scan */
        int32_t ni = 0;
        for (; ni < n_block; ni += 4) {
            const int8_t *b0 = b + (ni + 0) * k;
            const int8_t *b1 = b + (ni + 1) * k;
            const int8_t *b2 = b + (ni + 2) * k;
            const int8_t *b3 = b + (ni + 3) * k;
            int32_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            int32_t ki = 0;

            for (; ki <= k - 4; ki += 4) {
                int32_t a0 = a_row[ki], a1 = a_row[ki+1];
                int32_t a2 = a_row[ki+2], a3 = a_row[ki+3];
                s0 += a0*b0[ki] + a1*b0[ki+1] + a2*b0[ki+2] + a3*b0[ki+3];
                s1 += a0*b1[ki] + a1*b1[ki+1] + a2*b1[ki+2] + a3*b1[ki+3];
                s2 += a0*b2[ki] + a1*b2[ki+1] + a2*b2[ki+2] + a3*b2[ki+3];
                s3 += a0*b3[ki] + a1*b3[ki+1] + a2*b3[ki+2] + a3*b3[ki+3];
            }
            for (; ki < k; ++ki) {
                int32_t av = a_row[ki];
                s0 += av * b0[ki]; s1 += av * b1[ki];
                s2 += av * b2[ki]; s3 += av * b3[ki];
            }

            c[(ni+0) * m + mi] = beta * c[(ni+0) * m + mi] + alpha * s0;
            c[(ni+1) * m + mi] = beta * c[(ni+1) * m + mi] + alpha * s1;
            c[(ni+2) * m + mi] = beta * c[(ni+2) * m + mi] + alpha * s2;
            c[(ni+3) * m + mi] = beta * c[(ni+3) * m + mi] + alpha * s3;
        }

        /* Remainder columns (n not a multiple of 4) */
        for (; ni < n; ++ni) {
            const int8_t *b_row = b + ni * k;
            int32_t acc = 0;
            for (int32_t ki = 0; ki < k; ++ki)
                acc += (int32_t)a_row[ki] * b_row[ki];
            c[ni * m + mi] = beta * c[ni * m + mi] + alpha * acc;
        }
    }
}

/* ========== V4: P-extension smaqa inline asm (4×int8 MAC per instruction) ========== */

/**
 * @brief V4: Uses RISC-V P-extension smaqa instruction.
 *        smaqa rd,rs1,rs2: rd += rs1[7:0]*rs2[7:0] + rs1[15:8]*rs2[15:8]
 *                               + rs1[23:16]*rs2[23:16] + rs1[31:24]*rs2[31:24]
 *        BL616 -march=rv32imafcpzpsfoperand_xtheade already includes this.
 *        Mnemonic is "smaqa" (no th. prefix - that prefix is for XTheadE).
 */
void UmdIgemmS8S8S32_v4(UmdLayout layout, UmdTranspose transa, UmdTranspose transb,
                         int32_t m, int32_t n, int32_t k, int32_t alpha,
                         const int8_t *a, int32_t lda, const int8_t *b, int32_t ldb,
                         int32_t beta, int32_t *c, int32_t ldc)
{
    for (int32_t mi = 0; mi < m; ++mi) {
        const int8_t *a_row = a + mi * k;

        for (int32_t ni = 0; ni < n; ++ni) {
            const int8_t *b_row = b + ni * k;
            int32_t acc = 0;
            int32_t ki = 0;

            /* 4×int8 dot-product per smaqa instruction */
            for (; ki <= k - 4; ki += 4) {
                uint32_t ap, bp;
                __builtin_memcpy(&ap, a_row + ki, 4);
                __builtin_memcpy(&bp, b_row + ki, 4);
                __asm__ volatile("smaqa %0, %1, %2" : "+r"(acc) : "r"(ap), "r"(bp));
            }
            /* Scalar tail for k not a multiple of 4 */
            for (; ki < k; ++ki)
                acc += (int32_t)a_row[ki] * b_row[ki];

            c[ni * m + mi] = beta * c[ni * m + mi] + alpha * acc;
        }
    }
}

/* ========== V5: V3 register tiling + V4 smaqa (no volatile) ========== */

/**
 * @brief V5: Combines V3 register tiling (TILE_N=4) with V4 smaqa instruction.
 *        Key improvements over V4:
 *        - TILE_N=4 means one a_row packed word (ap) drives 4 smaqa calls,
 *          all with independent accumulators -> no RAW dependency between them.
 *        - No volatile on asm: compiler can schedule the 4 smaqa instructions
 *          around the 4 loads (bp0-bp3) to fill load latency slots.
 *        - alpha==1/beta==0 fast path eliminates writeback multiplications.
 */
void UmdIgemmS8S8S32_v5(UmdLayout layout, UmdTranspose transa, UmdTranspose transb,
                         int32_t m, int32_t n, int32_t k, int32_t alpha,
                         const int8_t *a, int32_t lda, const int8_t *b, int32_t ldb,
                         int32_t beta, int32_t *c, int32_t ldc)
{
    int32_t n_block = (n / 4) * 4;

    if (alpha == 1 && beta == 0) {
        for (int32_t mi = 0; mi < m; ++mi) {
            const int8_t *a_row = a + mi * k;

            int32_t ni = 0;
            for (; ni < n_block; ni += 4) {
                const int8_t *b0 = b + (ni + 0) * k;
                const int8_t *b1 = b + (ni + 1) * k;
                const int8_t *b2 = b + (ni + 2) * k;
                const int8_t *b3 = b + (ni + 3) * k;
                int32_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                int32_t ki = 0;

                for (; ki <= k - 4; ki += 4) {
                    uint32_t ap, bp0, bp1, bp2, bp3;
                    __builtin_memcpy(&ap,  a_row + ki, 4);
                    __builtin_memcpy(&bp0, b0 + ki, 4);
                    __builtin_memcpy(&bp1, b1 + ki, 4);
                    __builtin_memcpy(&bp2, b2 + ki, 4);
                    __builtin_memcpy(&bp3, b3 + ki, 4);
                    /* 4 independent smaqa: no data dependency -> schedulable */
                    __asm__("smaqa %0, %1, %2" : "+r"(s0) : "r"(ap), "r"(bp0));
                    __asm__("smaqa %0, %1, %2" : "+r"(s1) : "r"(ap), "r"(bp1));
                    __asm__("smaqa %0, %1, %2" : "+r"(s2) : "r"(ap), "r"(bp2));
                    __asm__("smaqa %0, %1, %2" : "+r"(s3) : "r"(ap), "r"(bp3));
                }
                for (; ki < k; ++ki) {
                    int32_t av = a_row[ki];
                    s0 += av * b0[ki]; s1 += av * b1[ki];
                    s2 += av * b2[ki]; s3 += av * b3[ki];
                }

                c[(ni+0) * m + mi] = s0;
                c[(ni+1) * m + mi] = s1;
                c[(ni+2) * m + mi] = s2;
                c[(ni+3) * m + mi] = s3;
            }

            for (; ni < n; ++ni) {
                const int8_t *b_row = b + ni * k;
                int32_t acc = 0;
                int32_t ki = 0;
                for (; ki <= k - 4; ki += 4) {
                    uint32_t ap, bp;
                    __builtin_memcpy(&ap, a_row + ki, 4);
                    __builtin_memcpy(&bp, b_row + ki, 4);
                    __asm__("smaqa %0, %1, %2" : "+r"(acc) : "r"(ap), "r"(bp));
                }
                for (; ki < k; ++ki)
                    acc += (int32_t)a_row[ki] * b_row[ki];
                c[ni * m + mi] = acc;
            }
        }
        return;
    }

    /* General path (alpha/beta not specialized) */
    for (int32_t mi = 0; mi < m; ++mi) {
        const int8_t *a_row = a + mi * k;

        int32_t ni = 0;
        for (; ni < n_block; ni += 4) {
            const int8_t *b0 = b + (ni + 0) * k;
            const int8_t *b1 = b + (ni + 1) * k;
            const int8_t *b2 = b + (ni + 2) * k;
            const int8_t *b3 = b + (ni + 3) * k;
            int32_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            int32_t ki = 0;

            for (; ki <= k - 4; ki += 4) {
                uint32_t ap, bp0, bp1, bp2, bp3;
                __builtin_memcpy(&ap,  a_row + ki, 4);
                __builtin_memcpy(&bp0, b0 + ki, 4);
                __builtin_memcpy(&bp1, b1 + ki, 4);
                __builtin_memcpy(&bp2, b2 + ki, 4);
                __builtin_memcpy(&bp3, b3 + ki, 4);
                __asm__("smaqa %0, %1, %2" : "+r"(s0) : "r"(ap), "r"(bp0));
                __asm__("smaqa %0, %1, %2" : "+r"(s1) : "r"(ap), "r"(bp1));
                __asm__("smaqa %0, %1, %2" : "+r"(s2) : "r"(ap), "r"(bp2));
                __asm__("smaqa %0, %1, %2" : "+r"(s3) : "r"(ap), "r"(bp3));
            }
            for (; ki < k; ++ki) {
                int32_t av = a_row[ki];
                s0 += av * b0[ki]; s1 += av * b1[ki];
                s2 += av * b2[ki]; s3 += av * b3[ki];
            }

            c[(ni+0) * m + mi] = beta * c[(ni+0) * m + mi] + alpha * s0;
            c[(ni+1) * m + mi] = beta * c[(ni+1) * m + mi] + alpha * s1;
            c[(ni+2) * m + mi] = beta * c[(ni+2) * m + mi] + alpha * s2;
            c[(ni+3) * m + mi] = beta * c[(ni+3) * m + mi] + alpha * s3;
        }

        for (; ni < n; ++ni) {
            const int8_t *b_row = b + ni * k;
            int32_t acc = 0;
            int32_t ki = 0;
            for (; ki <= k - 4; ki += 4) {
                uint32_t ap, bp;
                __builtin_memcpy(&ap, a_row + ki, 4);
                __builtin_memcpy(&bp, b_row + ki, 4);
                __asm__("smaqa %0, %1, %2" : "+r"(acc) : "r"(ap), "r"(bp));
            }
            for (; ki < k; ++ki)
                acc += (int32_t)a_row[ki] * b_row[ki];
            c[ni * m + mi] = beta * c[ni * m + mi] + alpha * acc;
        }
    }
}

/* ========== Shell Command Implementation ========== */

/**
 * @brief Shell command: igemm <m> <n> <k> [iterations]
 *
 * Runs Baseline / V3 / V4 GEMM benchmark and prints three-way comparison.
 * Uses bflb_mtimer_get_time_us() for timing (1us precision).
 */
static int cmd_igemm(int argc, char **argv)
{
    if (argc < 4) {
        printf(CLR_RED "Usage: igemm <m> <n> <k> [iterations]\r\n" CLR_RESET);
        printf("  m, n, k:      Matrix dimensions\r\n");
        printf("  iterations:   Number of runs (default: 1)\r\n");
        printf("\r\nExample:\r\n");
        printf("  igemm 16 16 16        # Single run\r\n");
        printf("  igemm 32 32 32 100    # 100 iterations benchmark\r\n");
        return -1;
    }

    int32_t m = atoi(argv[1]);
    int32_t n = atoi(argv[2]);
    int32_t k = atoi(argv[3]);
    int iterations = 1;

    if (argc > 4) {
        iterations = atoi(argv[4]);
        if (iterations < 1) iterations = 1;
        if (iterations > 10000) iterations = 10000;
    }

    if (m <= 0 || n <= 0 || k <= 0) {
        printf(CLR_RED "Error: Matrix dimensions must be positive\r\n" CLR_RESET);
        return -1;
    }

    #define MAX_DIM 64

    if (m > MAX_DIM || n > MAX_DIM || k > MAX_DIM) {
        printf(CLR_RED "Error: Max dimension is %d\r\n" CLR_RESET, MAX_DIM);
        return -1;
    }

    static int8_t  a[MAX_DIM * MAX_DIM];
    static int8_t  b[MAX_DIM * MAX_DIM];
    static int32_t c_ref[MAX_DIM * MAX_DIM];
    static int32_t c_v3[MAX_DIM * MAX_DIM];
    static int32_t c_v4[MAX_DIM * MAX_DIM];
    static int32_t c_v5[MAX_DIM * MAX_DIM];

    /* Initialize A and B with reproducible test data */
    for (int i = 0; i < m * k; i++) a[i] = (int8_t)(i % 128);
    for (int i = 0; i < k * n; i++) b[i] = (int8_t)((i + 1) % 128);

    printf(CLR_CYAN "========================================\r\n" CLR_RESET);
    printf(CLR_CYAN "  Integer GEMM Benchmark: Baseline / V3 / V4 / V5\r\n" CLR_RESET);
    printf(CLR_CYAN "========================================\r\n" CLR_RESET);
    printf("Matrix A: %d x %d (transposed)\r\n", k, m);
    printf("Matrix B: %d x %d\r\n", k, n);
    printf("Matrix C: %d x %d\r\n", m, n);
    printf("Iterations: %d\r\n", iterations);
    printf("----------------------------------------\r\n");

    uint64_t ops_per_gemm = 2ULL * m * n * k;
    uint64_t total_ops    = ops_per_gemm * iterations;
    uint64_t t0, t1, us_ref, us_v3, us_v4, us_v5;

    /* --- Baseline --- */
    memset(c_ref, 0, sizeof(int32_t) * m * n);
    UmdIgemmS8S8S32(kUmdColMajor, kUmdTrans, kUmdNoTrans,
                    m, n, k, 1, a, k, b, k, 0, c_ref, m);   /* warmup */
    memset(c_ref, 0, sizeof(int32_t) * m * n);

    t0 = bflb_mtimer_get_time_us();
    for (int iter = 0; iter < iterations; iter++) {
        UmdIgemmS8S8S32(kUmdColMajor, kUmdTrans, kUmdNoTrans,
                        m, n, k, 1, a, k, b, k, 0, c_ref, m);
    }
    t1 = bflb_mtimer_get_time_us();
    us_ref = t1 - t0;

    /* --- V3 (register tiling TILE_N=4 + k×4 unroll) --- */
    memset(c_v3, 0, sizeof(int32_t) * m * n);
    UmdIgemmS8S8S32_v3(kUmdColMajor, kUmdTrans, kUmdNoTrans,
                       m, n, k, 1, a, k, b, k, 0, c_v3, m);  /* warmup */
    memset(c_v3, 0, sizeof(int32_t) * m * n);

    t0 = bflb_mtimer_get_time_us();
    for (int iter = 0; iter < iterations; iter++) {
        UmdIgemmS8S8S32_v3(kUmdColMajor, kUmdTrans, kUmdNoTrans,
                           m, n, k, 1, a, k, b, k, 0, c_v3, m);
    }
    t1 = bflb_mtimer_get_time_us();
    us_v3 = t1 - t0;

    /* --- V4 (P-extension smaqa inline asm, no tiling) --- */
    memset(c_v4, 0, sizeof(int32_t) * m * n);
    UmdIgemmS8S8S32_v4(kUmdColMajor, kUmdTrans, kUmdNoTrans,
                       m, n, k, 1, a, k, b, k, 0, c_v4, m);  /* warmup */
    memset(c_v4, 0, sizeof(int32_t) * m * n);

    t0 = bflb_mtimer_get_time_us();
    for (int iter = 0; iter < iterations; iter++) {
        UmdIgemmS8S8S32_v4(kUmdColMajor, kUmdTrans, kUmdNoTrans,
                           m, n, k, 1, a, k, b, k, 0, c_v4, m);
    }
    t1 = bflb_mtimer_get_time_us();
    us_v4 = t1 - t0;

    /* --- V5 (reg tile 4x4 + smaqa, no volatile, alpha=1/beta=0 fast path) --- */
    memset(c_v5, 0, sizeof(int32_t) * m * n);
    UmdIgemmS8S8S32_v5(kUmdColMajor, kUmdTrans, kUmdNoTrans,
                       m, n, k, 1, a, k, b, k, 0, c_v5, m);  /* warmup */
    memset(c_v5, 0, sizeof(int32_t) * m * n);

    t0 = bflb_mtimer_get_time_us();
    for (int iter = 0; iter < iterations; iter++) {
        UmdIgemmS8S8S32_v5(kUmdColMajor, kUmdTrans, kUmdNoTrans,
                           m, n, k, 1, a, k, b, k, 0, c_v5, m);
    }
    t1 = bflb_mtimer_get_time_us();
    us_v5 = t1 - t0;

    /* --- Correctness checks --- */
    int mismatch_v3 = 0, mismatch_v4 = 0, mismatch_v5 = 0;
    for (int i = 0; i < m * n; i++) {
        if (c_ref[i] != c_v3[i]) mismatch_v3++;
        if (c_ref[i] != c_v4[i]) mismatch_v4++;
        if (c_ref[i] != c_v5[i]) mismatch_v5++;
    }

    /* --- Print results table --- */
    printf("  %-36s  %9s  %8s  %8s\r\n", "Version", "Total(us)", "Avg(us)", "MOPS");
    printf("  %-36s  %9llu  %8llu  %8.3f\r\n",
           "Baseline",
           (unsigned long long)us_ref,
           (unsigned long long)(us_ref / iterations),
           (us_ref > 0) ? (double)(total_ops * 1000000ULL / us_ref) / 1e6 : 0.0);
    printf("  %-36s  %9llu  %8llu  %8.3f\r\n",
           "V3 (reg tile 4x4 + k*4 unroll)",
           (unsigned long long)us_v3,
           (unsigned long long)(us_v3 / iterations),
           (us_v3 > 0) ? (double)(total_ops * 1000000ULL / us_v3) / 1e6 : 0.0);
    printf("  %-36s  %9llu  %8llu  %8.3f\r\n",
           "V4 (P-ext smaqa, no tile)",
           (unsigned long long)us_v4,
           (unsigned long long)(us_v4 / iterations),
           (us_v4 > 0) ? (double)(total_ops * 1000000ULL / us_v4) / 1e6 : 0.0);
    printf("  %-36s  %9llu  %8llu  %8.3f\r\n",
           "V5 (reg tile 4x4 + smaqa)",
           (unsigned long long)us_v5,
           (unsigned long long)(us_v5 / iterations),
           (us_v5 > 0) ? (double)(total_ops * 1000000ULL / us_v5) / 1e6 : 0.0);

    printf("----------------------------------------\r\n");
    if (us_ref > 0 && us_v3 > 0)
        printf("Speedup V3 vs Baseline: %.2fx\r\n", (double)us_ref / (double)us_v3);
    if (us_ref > 0 && us_v4 > 0)
        printf("Speedup V4 vs Baseline: %.2fx\r\n", (double)us_ref / (double)us_v4);
    if (us_ref > 0 && us_v5 > 0)
        printf("Speedup V5 vs Baseline: %.2fx\r\n", (double)us_ref / (double)us_v5);
    if (us_v3 > 0 && us_v5 > 0)
        printf("Speedup V5 vs V3:       %.2fx\r\n", (double)us_v3 / (double)us_v5);

    printf("----------------------------------------\r\n");
    if (mismatch_v3 == 0)
        printf(CLR_GREEN "Correctness V3: PASS\r\n" CLR_RESET);
    else
        printf(CLR_RED "Correctness V3: FAIL (%d mismatches)\r\n" CLR_RESET, mismatch_v3);
    if (mismatch_v4 == 0)
        printf(CLR_GREEN "Correctness V4: PASS\r\n" CLR_RESET);
    else
        printf(CLR_RED "Correctness V4: FAIL (%d mismatches)\r\n" CLR_RESET, mismatch_v4);
    if (mismatch_v5 == 0)
        printf(CLR_GREEN "Correctness V5: PASS\r\n" CLR_RESET);
    else
        printf(CLR_RED "Correctness V5: FAIL (%d mismatches)\r\n" CLR_RESET, mismatch_v5);

    printf("----------------------------------------\r\n");
    printf("Sample output C_ref / C_v3 / C_v4 / C_v5 [0..3]:\r\n");
    for (int i = 0; i < 4 && i < m * n; i++) {
        printf("  [%d] %d / %d / %d / %d\r\n", i, c_ref[i], c_v3[i], c_v4[i], c_v5[i]);
    }

    return 0;
}
SHELL_CMD_EXPORT_ALIAS(cmd_igemm, igemm, int8 GEMM benchmark);

/* ========== Main Entry Point ========== */

int main(void)
{
    board_init();

    configASSERT((configMAX_PRIORITIES > 4));

    /* Initialize shell with FreeRTOS task mode */
    struct bflb_device_s *uart0 = bflb_device_get_by_name("uart0");
    shell_init_with_task(uart0);

    /* Print welcome message */
    printf("\r\n");
    printf("========================================\r\n");
    printf("  mac_int8 - Integer GEMM Benchmark\r\n");
    printf("========================================\r\n");
    printf("Type 'igemm <m> <n> <k> [iterations]' to run benchmark\r\n");
    printf("Example: igemm 32 32 32 100\r\n");
    printf("\r\n");

    vTaskStartScheduler();

    while (1) {
        /* Should never reach here */
    }

    return 0;
}
