
/**
*    ICRAR - International Centre for Radio Astronomy Research
*    (c) UWA - The University of Western Australia
*    Copyright by UWA (in the framework of the ICRAR)
*    All rights reserved
*
*    This library is free software; you can redistribute it and/or
*    modify it under the terms of the GNU Lesser General Public
*    License as published by the Free Software Foundation; either
*    version 2.1 of the License, or (at your option) any later version.
*
*    This library is distributed in the hope that it will be useful,
*    but WITHOUT ANY WARRANTY; without even the implied warranty of
*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
*    Lesser General Public License for more details.
*
*    You should have received a copy of the GNU Lesser General Public
*    License along with this library; if not, write to the Free Software
*    Foundation, Inc., 59 Temple Place, Suite 330, Boston,
*    MA 02111-1307  USA
*/

#include <icrar/leap-accelerate/math/cuda/matrix_multiply.h>

#include <cublasLt.h>

namespace icrar
{
namespace cuda
{
    template<typename T>
    __global__ void g_matrix_multiply_vector(cublasHandle_t handle, const size_t m, const size_t n, const T* mat, const T* vec, T* out)
    {

    }

    template<typename T>
    __host__ void matrix_multiply_vector(cublasHandle_t handle, const size_t m, const size_t n, const T* mat, const T* vec, T* out)
    {
        g_matrix_multiply_vector<<<1,1>>>(handle,m,n,mat,vec,out);
    }

    __host__ void matrix_multiply_vector(cublasHandle_t handle, const size_t m, const size_t n, const double* mat, const double* vec, double* out) { matrix_multiply_vector<double>(handle,m,n,mat,vec,out); }
    __host__ void matrix_multiply_vector(cublasHandle_t handle, const size_t m, const size_t n, const float* mat, const float* vec, float* out) { matrix_multiply_vector<float>(handle,m,n,mat,vec,out); }
    __host__ void matrix_multiply_vector(cublasHandle_t handle, const size_t m, const size_t n, const int* mat, const int* vec, int* out) { matrix_multiply_vector<int>(handle,m,n,mat,vec,out); }

    template<typename T>
    __global__ void g_matrix_multiply_matrix(cublasHandle_t handle, const size_t m, const size_t n, const size_t k, const T* left, const T* right, T* out)
    {
        //TODO: cublasLtMatlmul();
    }

    template<typename T>
    __host__ void matrix_multiply_matrix(cublasHandle_t handle, const size_t m, const size_t n, const size_t k, const T* left, const T* right, T* out)
    {
        g_matrix_multiply_matrix<<<1,1>>>(handle,m,n,k,left,right,out);
    }

    __host__ void matrix_multiply_matrix(cublasHandle_t handle, const size_t m, const size_t n, const size_t k, const double* left, const double* right, double* out) {}
    __host__ void matrix_multiply_matrix(cublasHandle_t handle, const size_t m, const size_t n, const size_t k, const float* left, const float* right, float* out) {}
    __host__ void matrix_multiply_matrix(cublasHandle_t handle, const size_t m, const size_t n, const size_t k, const int* left, const int* right, int* out) {}
} // namespace cuda
} // namespace icrar
