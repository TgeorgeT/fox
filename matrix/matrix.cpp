#include "matrix.h"
#include <iostream>
#include <iomanip>

void print_matrix(int *matrix, int size)
{
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            std::cout << std::setw(6) << matrix[i * size + j] << " ";
        }
        std::cout << "\n";
    }
}

// generate matrix of size x size
int *generate_matrix(int size)
{
    int *matrix = new int[size * size];
    for (int i = 0; i < size * size; ++i)
    {
        matrix[i] = rand() % 100;
    }
    return matrix;
}

void multiply_matrices(int *matrix_a, int *matrix_b, int *result, int size)
{
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            for (int k = 0; k < size; ++k)
            {
                // result[i][j] = a[i][k] + b[k][j]
                result[i * size + j] += matrix_a[i * size + k] * matrix_b[k * size + j];
            }
        }
    }
}