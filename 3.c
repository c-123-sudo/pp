#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
// Merge two sorted halves
void merge(int arr[], int l, int m, int r) {
	int n1 = m - l + 1;
	int n2 = r - m;
	int *L = (int*)malloc(n1 * sizeof(int));
	int *R = (int*)malloc(n2 * sizeof(int));
	for (int i = 0; i < n1; i++) L[i] = arr[l + i];
	for (int j = 0; j < n2; j++) R[j] = arr[m + 1 + j];
	int i = 0, j = 0, k = l;
	while (i < n1 && j < n2) {
		arr[k++] = (L[i] <= R[j]) ? L[i++] : R[j++];
	}
	while (i < n1) arr[k++] = L[i++];
	while (j < n2) arr[k++] = R[j++];
	free(L);
	free(R);
}
// Parallel recursive merge sort
void parallel_merge_sort(int arr[], int l, int r, int depth) {
	if (l < r) {
		int m = l + (r - l) / 2;
		if (depth > 0) {
#pragma omp parallel sections
			{
#pragma omp section
				parallel_merge_sort(arr, l, m, depth - 1);
#pragma omp section
				parallel_merge_sort(arr, m + 1, r, depth - 1);
			}
		} else {
			// fall back to sequential recursion when depth exhausted
			parallel_merge_sort(arr, l, m, 0);
			parallel_merge_sort(arr, m + 1, r, 0);
		}
		merge(arr, l, m, r);
	}
}
int main() {
	int n = 1000000;
	int *arr = (int*)malloc(n * sizeof(int));
	for (int i = 0; i < n; i++) arr[i] = rand() % 1000000;
	double start, end;
	// Sequential baseline
	int *arr_copy = (int*)malloc(n * sizeof(int));
	for (int i = 0; i < n; i++) arr_copy[i] = arr[i];
	start = omp_get_wtime();
	parallel_merge_sort(arr_copy, 0, n - 1, 0); // depth = 0 → sequential
	end = omp_get_wtime();
	printf("Sequential time: %f s\n", end - start);
	// Parallel
	for (int i = 0; i < n; i++) arr_copy[i] = arr[i];
	start = omp_get_wtime();
	parallel_merge_sort(arr_copy, 0, n - 1, 4); // depth controls parallelism
	end = omp_get_wtime();
	printf("Parallel time: %f s\n", end - start);
	free(arr);
	free(arr_copy);
	return 0;
}
