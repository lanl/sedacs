To run:
1. module load nvhpc
2. nvcc -arch=sm_70 diagonalizeJosh.cu -lcusolver -lcublas -o diagonalize 
