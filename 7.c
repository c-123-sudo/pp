#include <mpi.h>
#include <stdio.h>

#define ARRAY_SIZE 5

int main(int argc, char** argv) {
    int rank, size;
    int data_array[ARRAY_SIZE]; // The data structure to be broadcast
    int local_result;           // The specific result calculated by each process

    // 1. Initialize the MPI environment
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Ensure we have at least 2 processes for a meaningful P2P demo
    if (size < 2) {
        if (rank == 0) {
            printf("Please run with at least 2 processes (e.g., mpirun -np 4 ...)\n");
        }
        MPI_Finalize();
        return 0;
    }

    // --- PART 1: COLLECTIVE COMMUNICATION (MPI_Bcast) ---
    
    if (rank == 0) {
        // Master initializes the data array
        printf("[Master] Initializing array to broadcast...\n");
        for (int i = 0; i < ARRAY_SIZE; i++) {
            data_array[i] = (i + 1) * 10; // {10, 20, 30, 40, 50}
        }
    }

    // Broadcast the array from Root (0) to everyone else.
    // Syntax: (buffer, count, datatype, root, communicator)
    MPI_Bcast(data_array, ARRAY_SIZE, MPI_INT, 0, MPI_COMM_WORLD);

    // Verify: Every process now has the data
    printf("Process %d received broadcast data. Element[0] = %d\n", rank, data_array[0]);
    
    // Ensure printing is (mostly) synchronized before moving on
    MPI_Barrier(MPI_COMM_WORLD); 


    // --- PART 2: POINT-TO-POINT COMMUNICATION (MPI_Send / MPI_Recv) ---

    if (rank != 0) {
        // WORKER PROCESSES
        // Perform a calculation: result = (Rank) * (Value from array at index Rank)
        // We use modulo (%) to handle cases where rank >= ARRAY_SIZE
        int index = rank % ARRAY_SIZE;
        local_result = rank * data_array[index];

        printf("   -> Process %d sending result %d to Master\n", rank, local_result);
        
        // Send the result to Rank 0
        // Syntax: (buffer, count, datatype, dest, tag, communicator)
        MPI_Send(&local_result, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    
    } else {
        // MASTER PROCESS
        printf("\n[Master] Waiting to receive results from workers...\n");
        
        int received_val;
        for (int i = 1; i < size; i++) {
            // Receive from any source (i)
            // Syntax: (buffer, count, datatype, source, tag, comm, status)
            MPI_Recv(&received_val, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            printf("[Master] Received %d from Process %d\n", received_val, i);
        }
    }

    // 3. Finalize MPI environment
    MPI_Finalize();
    return 0;
}
