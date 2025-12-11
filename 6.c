#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char** argv) {
MPI_Init(&argc, &argv);
int rank, size;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);
int N = 100; // total array size (small example)
if (argc > 1) N = atoi(argv[1]);
double *data = NULL;
 if (rank == 0) {
data = malloc(sizeof(double) * N);
for (int i = 0; i < N; ++i) data[i] = 1.0; // fill with 1.0 for easy check
}
int chunk = N / size; // integer division
double *local = malloc(sizeof(double) * chunk);
// Scatter equal chunks of size 'chunk' to everyone
MPI_Scatter(data, chunk, MPI_DOUBLE, local, chunk, MPI_DOUBLE, 0, MPI_COMM_WORLD);
// local partial sum
double local_sum = 0.0;
for (int i = 0; i < chunk; ++i) local_sum += local[i];
// reduce to get global sum of scattered parts
double global_sum = 0.0;
MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
// root handles remainder elements (if any)
if (rank == 0) {
int start_rem = chunk * size;
for (int i = start_rem; i < N; ++i) global_sum += data[i];
printf("N=%d, size=%d -> Global sum = %.1f (expected ~%d)\n",
N, size, global_sum, N);
free(data);
}
free(local);
MPI_Finalize();
return 0;
}
