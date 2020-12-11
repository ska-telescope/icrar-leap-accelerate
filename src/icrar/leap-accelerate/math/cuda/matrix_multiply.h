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

#pragma once

#ifdef CUDA_ENABLED

#include <icrar/leap-accelerate/cuda/device_vector.h>
#include <icrar/leap-accelerate/cuda/device_matrix.h>
#include <icrar/leap-accelerate/exception/exception.h>

#include <cublasLt.h>

// C++ Style interface (templates not supported when linking to nvcc compiled sources)
namespace icrar
{
namespace cuda
{
    // Matrix Multiply Column Vector
    //    --N--       1       1
    // | [     ]   | [ ]   | [ ]
    // M [     ] x N [ ] = M [ ]
    // | [     ]   | [ ]   | [ ]
    // | [     ]           | [ ]
    //

    __host__ void matrix_multiply_vector(cublasLtHandle_t handle, const size_t m, const size_t n, const double* mat, const double* vec, double* out);
    __host__ void matrix_multiply_vector(cublasLtHandle_t handle, const size_t m, const size_t n, const float* mat, const float* vec, float* out);
    __host__ void matrix_multiply_vector(cublasLtHandle_t handle, const size_t m, const size_t n, const int* mat, const int* vec, int* out);

    template<typename T>
    __host__ void multiply(cublasLtHandle_t handle, const device_matrix<T>& left, const device_vector<T>& right, device_vector<T>& result)
    {
        if(left.GetCols() != right.GetSize())
        {
            throw invalid_argument_exception("left columns does not match right rows", "right", __FILE__, __LINE__);
        }
        if(left.GetRows() != result.GetSize())
        {
            throw invalid_argument_exception("result matrix has invalid dimensions", "result", __FILE__, __LINE__);
        }
        matrix_multiply_vector(handle, left.GetRows(), left.GetCols(), left.Get(), right.Get(), result.Get());
    }

    // Matrix Multiply Matrix
    //    --N--       -k-       -k-
    // | [     ]   | [   ]   | [   ]
    // M [     ] x N [   ] = M [   ]
    // | [     ]   | [   ]   | [   ]
    // | [     ]             | [   ]
    //

    __host__ void matrix_multiply_matrix(cublasLtHandle_t handle, const size_t m, const size_t n, const size_t k, const double* left, const double* right, double* out);
    __host__ void matrix_multiply_matrix(cublasLtHandle_t handle, const size_t m, const size_t n, const size_t k, const float* left, const float* right, float* out);
    __host__ void matrix_multiply_matrix(cublasLtHandle_t handle, const size_t m, const size_t n, const size_t k, const int* left, const int* right, int* out);

    template<typename T>
    __host__ void multiply(cublasLtHandle_t handle, const device_matrix<T>& left, const device_matrix<T>& right, device_matrix<T>& result)
    {
        if(left.GetCols() != right.GetRows())
        {
            throw invalid_argument_exception("left columns does not match right rows", "right", __FILE__, __LINE__);
        }
        if(left.GetRows() != result.GetRows() || right.GetCols() != result.GetCols())
        {
            throw invalid_argument_exception("result matrix has invalid dimensions", "result", __FILE__, __LINE__);
        }
        matrix_multiply_vector(handle, left.GetRows(), left.GetCols(), right.GetCols(), left.Get(), right.Get(), result.Get());
    }
} // namespace cuda
} // namespace icrar
#endif
