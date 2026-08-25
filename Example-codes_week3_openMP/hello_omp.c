/* hello_omp.c  --  Week 3: "Hello, Threads" & "The Fork-Join Model" slides.
 *
 * The single most important OpenMP fact: inside a parallel region EVERY thread
 * runs the code. This program is the slide verbatim, so you can show fork/join
 * live and prove that OMP_NUM_THREADS controls the team size.
 *
 * Build:  make hello_omp
 * Run:    OMP_NUM_THREADS=4 ./hello_omp
 *         OMP_NUM_THREADS=8 ./hello_omp     # watch the count change
 */
#include <stdio.h>
#ifdef _OPENMP
#include <omp.h>
#endif

int main(void) {
#ifndef _OPENMP
    printf("Built WITHOUT OpenMP -- runs serially as one thread.\n");
    printf("(The pragma below is simply ignored; the code is still correct.)\n");
#endif

    /* fork: the runtime spawns a team of threads for this region */
    #pragma omp parallel
    {
#ifdef _OPENMP
        int id = omp_get_thread_num();      /* who am I in the team? */
        int n  = omp_get_num_threads();     /* how big is the team?  */
#else
        int id = 0, n = 1;
#endif
        printf("Hello from thread %d of %d\n", id, n);
    }   /* join: threads rendezvous and only the master continues */

    return 0;
}
