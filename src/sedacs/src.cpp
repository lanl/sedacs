#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>


// Python equivalent of import numpy as np, so that we can 
// access numpy functions like np.random. Here in C++ this takes
// the form of np::random.
namespace py = pybind11;

double sum_matrix(py::array_t<double> marix) {

    // Access the data buffer of the NumPy array
    py::buffer_info buf_info = matrix.request();

    // Get a pointer to the 
    double* ptr = static_cast<double*>(buf_info.ptr);

    // Get shape of the matrix
    size_t M = buf_info.shape[0];  // Rows
    size_t N = buf_info.shape[1];  // Columns

    // Loop through and sum the elements
    double sum = 0.0;
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            sum += ptr[i * N + j];  // Access element at (i, j)
        }
    }
    return sum;
}

PYBIND11_MODULE(your_module, m) {
    m.def("sum_matrix", &sum_matrix, "Sum all elements of a matrix");
}

