#include <mpi.h>
#include <iostream>
#include <cstdlib>
#include <iomanip>
#include <math.h>
#include "grid/grid.h"
#include "matrix/matrix.h"
#include "fox/fox.h"

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    grid_structure grid;
    grid_setup(&grid);
    MPI_Barrier(grid.grid_comm);

    const int MATRIX_SIZE = 30;
    int block_size = MATRIX_SIZE / grid.dim;

    // std::cout << "bsize: " << block_size << "\n";

    // Allocate matrices
    int *matrix_a = nullptr;
    int *matrix_b = nullptr;
    int *matrix_c = nullptr;

    if (grid.rank == 0)
    {
        matrix_a = generate_matrix(MATRIX_SIZE);
        matrix_b = generate_matrix(MATRIX_SIZE);
        matrix_c = new int[MATRIX_SIZE * MATRIX_SIZE];

        std::cout << "Matrix A:\n";
        // print_matrix(matrix_a, MATRIX_SIZE);
        std::cout << "Matrix B:\n";
        // print_matrix(matrix_b, MATRIX_SIZE);
    }

    std::cout << "got here\n";

    int *local_a = new int[block_size * block_size];
    int *local_b = new int[block_size * block_size];
    int *local_c = new int[block_size * block_size];

    MPI_Datatype block_type;
    MPI_Type_vector(block_size, block_size, MATRIX_SIZE, MPI_INT, &block_type);
    MPI_Type_commit(&block_type);

    for (int i = 0; i < grid.dim; ++i)
    {
        for (int j = 0; j < grid.dim; ++j)
        {
            int rank = i * grid.dim + j; // Rank in the grid
            if (grid.rank == 0)
            {
                int offset = i * MATRIX_SIZE * block_size + j * block_size;
                MPI_Send(matrix_a + offset, 1, block_type, rank, 0, grid.grid_comm);
            }
            if (grid.rank == rank)
            {
                MPI_Recv(local_a, block_size * block_size, MPI_INT, 0, 0, grid.grid_comm, MPI_STATUS_IGNORE);
            }
        }
    }

    std::cout << "got here1\n";

    for (int i = 0; i < grid.dim; ++i)
    {
        for (int j = 0; j < grid.dim; ++j)
        {
            int rank = i * grid.dim + j; // Rank in the grid
            if (grid.rank == 0)
            {
                int offset = i * MATRIX_SIZE * block_size + j * block_size;
                MPI_Send(matrix_b + offset, 1, block_type, rank, 1, grid.grid_comm);
            }
            if (grid.rank == rank)
            {
                MPI_Recv(local_b, block_size * block_size, MPI_INT, 0, 1, grid.grid_comm, MPI_STATUS_IGNORE);
            }
        }
    }

    fox_multiply(MATRIX_SIZE, &grid, local_a, local_b, local_c);

    for (int i = 0; i < grid.dim; ++i)
    {
        for (int j = 0; j < grid.dim; ++j)
        {
            int rank = i * grid.dim + j;
            if (grid.rank == rank)
            {
                MPI_Send(local_c, block_size * block_size, MPI_INT, 0, 2, grid.grid_comm);
            }
            if (grid.rank == 0)
            {
                int offset = i * MATRIX_SIZE * block_size + j * block_size;
                MPI_Recv(matrix_c + offset, 1, block_type, rank, 2, grid.grid_comm, MPI_STATUS_IGNORE);
            }
        }
    }

    if (grid.rank == 0)
    {
        int *serial_result = new int[MATRIX_SIZE * MATRIX_SIZE]();
        multiply_matrices(matrix_a, matrix_b, serial_result, MATRIX_SIZE);

        bool correct = true;
        for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; ++i)
        {
            if (matrix_c[i] != serial_result[i])
            {
                correct = false;
                std::cout << "Mismatch at index " << i << ": "
                          << "Expected " << serial_result[i] << ", Got " << matrix_c[i] << "\n";
            }
        }

        if (correct)
        {
            std::cout << "Verification passed: The distributed result matches the serial result.\n";
        }
        else
        {
            std::cout << "Verification failed: The distributed result does not match the serial result.\n";
        }

        delete[] serial_result;
        delete[] matrix_a;
        delete[] matrix_b;
        delete[] matrix_c;
    }

    delete[] local_a;
    delete[] local_b;

    MPI_Finalize();
    return 0;
}