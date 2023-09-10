#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE

    // Matrix Multiplication and Exponentiation
    for (size_t i = 0; i < m; i += batch){ // Loop 1 for batch
        size_t start = i * n;
        float *Z = new float[batch * k];
        for (size_t j = 0; j < batch; j++){ // Loop 2 for row in batch
            for (size_t l = 0; l < k; l++){ // Loop 3 for columns in theta
                float sum = 0.0;
                for (size_t m = 0; m < n; m++){ // Loop 4 for dimensions
                    // Matrix multiplication
                    sum += X[start + j * n + m] * theta[m * k + l];
                }
                Z[j * k + l] = exp(sum); // exponentiation
            }
        }

        // Normalisation
        float *Z_sum = new float[batch];
        for (size_t j = 0; j < batch; j++) {
            float sum = 0.0;
            for (size_t l=0; l < k; l++){
                sum += Z[j * k + l];
            }
            Z_sum[j] = sum;
        }
        for (size_t j = 0; j < batch; j++){
            for (size_t l = 0; l < k; l++){
                Z[j * k + l] /= Z_sum[j];
            }
        }

        // Z -= I
        for (size_t j = 0; j < batch; j++){
            Z[j * k + y[i + j]] -= 1.0;
        }

        // X.T @ Z
        for (size_t j = 0; j < n; j++) {
            for (size_t l = 0; l < k; l++) {
                float sum = 0.0;
                for (size_t m = 0; m < batch; m++){
                    sum += X[start + m * n + j] * Z[m * k + l];
                }
                theta[j * k + l] -= lr / batch * sum;
            }
        }
    }

    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
