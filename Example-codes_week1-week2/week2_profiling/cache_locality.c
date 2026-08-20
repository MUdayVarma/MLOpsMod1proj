/* cache_locality.c  --  Week 2: "Why Data Layout Matters" slide.
 *
 * Same FLOPs, same result, ~one memory layout: we sum every element of an
 * N x N matrix stored row-major. Traversing it row-by-row (stride 1) uses
 * each 64-byte cache line fully; traversing it column-by-column (stride N)
 * touches a new cache line almost every element and thrashes the cache.
 *
 * The slide's claim: "Same math, same FLOPs -- but the first can be many
 * times faster." This program measures that difference so students see it.
 *
 * Build (no OpenMP needed -- plain C):
 *     clang -O2 cache_locality.c -o cache_locality      # macOS
 *     gcc   -O2 cache_locality.c -o cache_locality      # Linux/cluster
 * Run:
 *     ./cache_locality           # default N=4096
 *     ./cache_locality 8192      # bigger matrix, bigger gap
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    long n = (argc > 1) ? atol(argv[1]) : 4096;   /* matrix is n x n */
    double *a = malloc((size_t)n * n * sizeof(double));
    if (!a) { fprintf(stderr, "alloc failed for n=%ld\n", n); return 1; }

    /* First-touch initialise (also warms the allocator). */
    for (long i = 0; i < n * n; i++) a[i] = 1.0;

    /* --- ROW-MAJOR traversal: inner index j is contiguous (stride 1). --- */
    double t0 = now_sec();
    double sum_row = 0.0;
    for (long i = 0; i < n; i++)
        for (long j = 0; j < n; j++)
            sum_row += a[i * n + j];         /* good spatial locality */
    double t_row = now_sec() - t0;

    /* --- COLUMN-MAJOR traversal: inner index i jumps a whole row (stride n). */
    t0 = now_sec();
    double sum_col = 0.0;
    for (long j = 0; j < n; j++)
        for (long i = 0; i < n; i++)
            sum_col += a[i * n + j];         /* cache miss almost every step */
    double t_col = now_sec() - t0;

    printf("Matrix: %ld x %ld doubles  (%.1f MB)\n",
           n, n, (double)n * n * sizeof(double) / 1e6);
    printf("Row-major   (stride 1): %8.4f s   sum=%.1f\n", t_row, sum_row);
    printf("Column-major(stride N): %8.4f s   sum=%.1f\n", t_col, sum_col);
    printf("Slowdown from bad layout: %.2fx\n", t_col / t_row);
    printf("\nBoth loops did the SAME %ld additions. Only the memory\n"
           "access pattern changed. That is the whole lesson.\n", n * n);

    free(a);
    return 0;
}
