#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define N 500

void init_matrix(double mat[N][N]) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            mat[i][j] = rand() % 10;
}

int main() {
    static double A[N][N], B[N][N], C[N][N];

    srand(time(NULL));

    init_matrix(A);
    init_matrix(B);

    printf("Matrix multiplication %dx%d\n", N, N);

    for (int t = 1; t <= 8; t *= 2) {

        // Reset C
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                C[i][j] = 0.0;

        double start = omp_get_wtime();

        #pragma omp parallel for num_threads(t) collapse(2) schedule(static)
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                double sum = 0.0;
                for (int k = 0; k < N; k++) {
                    sum += A[i][k] * B[k][j];
                }
                C[i][j] = sum;
            }
        }

        double end = omp_get_wtime();
        printf("Threads: %d | Time: %f seconds\n", t, end - start);
    }

    return 0;
}
