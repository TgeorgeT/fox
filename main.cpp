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

    const int MATRIX_SIZE = 1024;
    int block_size = MATRIX_SIZE / grid.dim;

    // std::cout << "bsize: " << block_size << "\n";

    // Allocate matrices
    int *matrix_a = nullptr;
    int *matrix_b = nullptr;
    int *matrix_c = nullptr;

    // std::cout << "block size " << block_size << "\n";

    if (grid.rank == 0)
    {
        matrix_a = generate_matrix(MATRIX_SIZE);
        matrix_b = generate_matrix(MATRIX_SIZE);
        matrix_c = new int[MATRIX_SIZE * MATRIX_SIZE];

        // std::cout << "Matrix A:\n";
        // print_matrix(matrix_a, MATRIX_SIZE);
        // std::cout << "Matrix B:\n";
        // print_matrix(matrix_b, MATRIX_SIZE);
    }

    // std::cout << "got here\n";

    int *local_a = new int[block_size * block_size];
    int *local_b = new int[block_size * block_size];
    int *local_c = new int[block_size * block_size];

    MPI_Datatype block_type;
    // MPI_Type_vector(block_size, block_size, MATRIX_SIZE, MPI_INT, &block_type);
    MPI_Type_commit(&block_type);

    int sizes[2] = {MATRIX_SIZE, MATRIX_SIZE};
    int subsizes[2] = {block_size, block_size};
    int starts[2] = {0, 0};

    MPI_Datatype subarray_type, resized_type;
    MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPI_INT, &subarray_type);
    MPI_Type_create_resized(subarray_type, 0, block_size * sizeof(int), &resized_type);
    MPI_Type_commit(&resized_type);

    int *sendcounts = nullptr;
    int *displs = nullptr;

    if (grid.rank == 0)
    {
        displs = new int[grid.dim * grid.dim];
        sendcounts = new int[grid.dim * grid.dim];
        std::cout << "grid dim: " << grid.dim << "\n";
        for (int i = 0; i < grid.dim * grid.dim; i++)
            sendcounts[i] = 1;
        int disp = 0;
        for (int i = 0; i < grid.dim; i++)
        {
            for (int j = 0; j < grid.dim; j++)
            {
                displs[i * grid.dim + j] = disp;
                disp += 1;
            }
            disp += (block_size - 1) * grid.dim;
        }

        for (int i = 0; i < grid.dim * grid.dim; i++)
        {
            std::cout << displs[i] << " ";
        }
        std::cout << "\n";
    }
    MPI_Scatterv(matrix_a, sendcounts, displs, resized_type, local_a, block_size * block_size, MPI_INT, 0, grid.grid_comm);

    MPI_Scatterv(matrix_b, sendcounts, displs, resized_type, local_b, block_size * block_size, MPI_INT, 0, grid.grid_comm);

    // std::cout << "rank " << grid.rank << " received: \n";
    // for (int i = 0; i < block_size; ++i)
    // {
    //     for (int j = 0; j < block_size; ++j)
    //     {
    //         std::cout << local_a[i * block_size + j] << " ";
    //     }
    //     std::cout << "\n";
    // }

    fox_multiply(MATRIX_SIZE, &grid, local_a, local_b, local_c);

    MPI_Gatherv(local_c, block_size * block_size, MPI_INT, matrix_c, sendcounts, displs, resized_type, 0, grid.grid_comm);
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