#include <stdio.h>
#include <omp.h>
#define N 100
#define CHUNK 10
void print_schedule_info(const char *schedule_type, int i, int chunk) {
	printf("Thread %d executing a chunk at iteration %d\n",
			omp_get_thread_num(), i);
}
int main() {
	int i;
	printf("\n--- OpenMP Thread Schedule Example ---\n");
	printf("\n## 1: Static Scheduling (chunk size %d)\n", CHUNK);
#pragma omp parallel for schedule(static, CHUNK)
	for (i = 0; i < N; i++) {
		print_schedule_info("Static", i, CHUNK);
	}
	printf("Static schedule complete\n");
	printf("\n## 2: Dynamic Scheduling (chunk size %d)\n", CHUNK);
#pragma omp parallel for schedule(dynamic, CHUNK)
	for (i = 0; i < N; i++) {
		print_schedule_info("Dynamic", i, CHUNK);
	}
	printf("Dynamic schedule complete\n");
	printf("\n## 3: Guided Scheduling (min chunk size %d)\n", CHUNK);
#pragma omp parallel for schedule(guided, CHUNK)
	for (i = 0; i < N; i++) {
		print_schedule_info("Guided", i, CHUNK);
	}
	printf("Guided schedule complete\n");
}
