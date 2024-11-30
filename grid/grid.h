#include <mpi.h>

struct grid_structure
{
    int proc_count;
    int dim;
    int row;
    int col;
    int rank;
    MPI_Comm grid_comm;
    MPI_Comm row_comm;
    MPI_Comm col_comm;
};

void grid_setup(grid_structure *grid);