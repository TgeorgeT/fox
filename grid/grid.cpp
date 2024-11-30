#include "grid.h"
#include <math.h>

void grid_setup(grid_structure *grid)
{
    int dimensions[2];
    int wrap_around[2];
    int coordinates[2];
    int free_coords[2];
    int world_rank;

    MPI_Comm_size(MPI_COMM_WORLD, &(grid->proc_count));
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    grid->dim = static_cast<int>(sqrt(grid->proc_count));
    dimensions[0] = dimensions[1] = grid->dim;

    wrap_around[0] = 1;
    wrap_around[1] = 0;

    MPI_Cart_create(MPI_COMM_WORLD, 2, dimensions, wrap_around, 1, &(grid->grid_comm));
    MPI_Comm_rank(grid->grid_comm, &(grid->rank));
    MPI_Cart_coords(grid->grid_comm, grid->rank, 2, coordinates);
    grid->row = coordinates[0];
    grid->col = coordinates[1];

    free_coords[0] = 0;
    free_coords[1] = 1;
    MPI_Cart_sub(grid->grid_comm, free_coords, &(grid->row_comm));

    free_coords[0] = 1;
    free_coords[1] = 0;
    MPI_Cart_sub(grid->grid_comm, free_coords, &(grid->col_comm));

    if (world_rank == 0)
    {
        std::cout << "Grid setup complete: " << grid->dim << "x" << grid->dim << "\n";
    }
    std::cout << "Process " << grid->rank << " at coordinates ("
              << grid->row << ", " << grid->col << ")\n";
}