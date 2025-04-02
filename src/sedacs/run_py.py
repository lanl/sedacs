import numpy as np
import your_module  # Assuming your C++ module is compiled to Python as 'your_module'

# Generate random matrix (M, N)
M, N = 3, 4
matrix = np.random.rand(M, N)

# Call C++ function to sum the elements of the matrix
result = your_module.sum_matrix(matrix)
print(f"Sum of matrix elements: {result}")

