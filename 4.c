    #include <stdio.h>
#include <omp.h>

/* * 1. Define a global variable.
 * 2. Apply threadprivate so each thread gets its own persistent copy.
 */
int my_persistent_var = 0;
#pragma omp threadprivate(my_persistent_var)

int main() {
    // Disable dynamic thread adjustment to ensure thread persistence behavior guarantees
   // omp_set_dynamic(0);
    
    // Set the master variable to a baseline value
    my_persistent_var = 100;
    
    printf("--- Initial Serial Region ---\n");
    printf("Master Thread: my_persistent_var set to %d\n\n", my_persistent_var);

    /*
     * PARALLEL REGION 1: Initialization and Modification
     * copyin(my_persistent_var): Copies 100 from Master to all threads.
     */
    printf("--- Parallel Region 1 (With copyin) ---\n");
    #pragma omp parallel copyin(my_persistent_var)
    {
        int tid = omp_get_thread_num();
        
        // Verify copyin worked (everyone should see 100)
        printf("Thread %d sees initial value: %d\n", tid, my_persistent_var);
        
        // Modify the variable (add thread ID to make it unique)
        my_persistent_var += tid; 
        
        printf("Thread %d changed value to:   %d\n", tid, my_persistent_var);
    }

    /*
     * SERIAL REGION
     * Back to master thread only. 
     * Master's copy was modified in Region 1 (100 + 0 = 100).
     */
    printf("\n--- Serial Region ---\n");
    printf("Master Thread sees its own copy: %d %d \n\n", my_persistent_var,omp_get_thread_num());

    /*
     * PARALLEL REGION 2: Persistence Check
     * No copyin here. 
     * Threads should remember the values they calculated in Region 1.
     */
    printf("--- Parallel Region 2 (Persistence Check) ---\n");
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        // If threadprivate works, Thread N should still see (100 + N)
        printf("Thread %d retrieves persistent value: %d\n", tid, my_persistent_var);
    }

    return 0;
}
