#include "../grid/grid.h"
#include "../matrix/matrix.h"

#include <fstream>
#include <sstream>

void print_matrix_to_stream(int *matrix, int block_size, std::ostream &out)
{
    for (int i = 0; i < block_size; ++i)
    {
        for (int j = 0; j < block_size; ++j)
        {
            out << matrix[i * block_size + j] << " ";
        }
        out << "\n";
    }
    out << "\n";
}

void fox_multiply(int n, grid_structure *grid, int *local_a, int *local_b, int *local_c)
{
    MPI_Status status;

    int block_size = n / grid->dim;
    int *temp_a = new int[block_size * block_size];

    if (grid->grid_comm == MPI_COMM_NULL)
    {
        std::cout << "Error: grid->grid_comm is null.\n";
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    // Create a process-specific log file
    std::ostringstream filename;
    filename << "process_" << grid->rank << ".log";
    std::ofstream log_file(filename.str());

    log_file << "Process " << grid->rank << " (Row: " << grid->row
             << ", Col: " << grid->col << ") starting Fox algorithm.\n";

    // Log local_a matrix
    log_file << "Initial local_a matrix:\n";
    print_matrix_to_stream(local_a, block_size, log_file);

    // Log local_b matrix
    log_file << "Initial local_b matrix:\n";
    print_matrix_to_stream(local_b, block_size, log_file);

    // Determine source and destination for matrix rotation
    int src, dst;
    MPI_Cart_shift(grid->grid_comm, 0, 1, &src, &dst);

    std::cout << "src: " << src << "dst " << dst << '\n';

    log_file << "src: " << src << "dst " << dst << '\n';

    // Initialize local_c to zero
    for (int i = 0; i < block_size * block_size; ++i)
    {
        local_c[i] = 0;
    }

    // Start the stages of the Fox algorithm
    for (int stage = 0; stage < grid->dim; ++stage)
    {
        int root = (grid->row + stage) % grid->dim;

        // Broadcast the relevant block of A
        if (grid->col == root)
        {
            std::copy(local_a, local_a + block_size * block_size, temp_a);
            log_file << "Process " << grid->rank << ": Root for stage " << stage
                     << " broadcasting matrix A.\n";
        }

        MPI_Bcast(temp_a, block_size * block_size, MPI_INT, root, grid->row_comm);

        // Log the broadcasted block
        log_file << "Process " << grid->rank << ": Received broadcasted matrix A:\n";
        print_matrix_to_stream(temp_a, block_size, log_file);

        // Perform local matrix multiplication
        multiply_matrices(temp_a, local_b, local_c, block_size);

        // Log intermediate result of local_c
        log_file << "Process " << grid->rank << ": Intermediate local_c after stage "
                 << stage << ":\n";
        print_matrix_to_stream(local_c, block_size, log_file);

        // Rotate local_b among column communicators
        MPI_Sendrecv_replace(local_b, block_size * block_size, MPI_INT, dst, 0, src, 0, grid->grid_comm, &status);

        // Log rotated local_b
        log_file << "Process " << grid->rank << ": Rotated matrix B after stage "
                 << stage << ":\n";
        print_matrix_to_stream(local_b, block_size, log_file);
    }

    log_file << "Process " << grid->rank << ": Completed Fox algorithm.\n";

    // Clean up
    delete[] temp_a;
    log_file.close();
}
