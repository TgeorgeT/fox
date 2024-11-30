mpicxx main.cpp -o main fox/fox.cpp grid/grid.cpp matrix/matrix.cpp
mpirun -np 4 ./main 