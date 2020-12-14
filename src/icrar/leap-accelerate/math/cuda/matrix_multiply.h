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

#include <icrar/leap-accelerate/cuda/helper_cuda.cuh>
#include <cublasLt.h>

// C++ Style interface (templates not supported when linking to nvcc compiled sources)
namespace icrar
{
namespace cuda
{
    // Matrix Multiply Matrix
    //    --N--       -k-       -k-
    // | [     ]   | [   ]   | [   ]
    // M [     ] x N [   ] = M [   ]
    // | [     ]   | [   ]   | [   ]
    // | [     ]             | [   ]

    __host__ void mat_mul(cublasHandle_t handle, const size_t m, const size_t n, const size_t k, const double* left, const double* right, double* out);
    __host__ void mat_mul(cublasHandle_t handle, const size_t m, const size_t n, const size_t k, const float* left, const float* right, float* out);
    __host__ void mat_mul(cublasHandle_t handle, const size_t m, const size_t n, const size_t k, const int* left, const int* right, int* out);

    __host__ void mat_mul(cublasLtHandle_t handle, const size_t m, const size_t n, const size_t k, const double* left, const double* right, double* out);
    __host__ void mat_mul(cublasLtHandle_t handle, const size_t m, const size_t n, const size_t k, const float* left, const float* right, float* out);
    __host__ void mat_mul(cublasLtHandle_t handle, const size_t m, const size_t n, const size_t k, const int* left, const int* right, int* out);

    // Matrix Multiply Matrix Add
    //    --N--       -k-       -k-       -k-
    // | [     ]   | [   ]   | [   ]   | [   ]
    // M [     ] x N [   ] + M [   ] = M [   ]
    // | [     ]   | [   ]   | [   ]   | [   ]
    // | [     ]             | [   ]   | [   ]
    __host__ void mat_mul_add(cublasHandle_t handle, const size_t m, const size_t n, const size_t k, const double* a, const double* b, double* c);
    __host__ void mat_mul_add(cublasHandle_t handle, const size_t m, const size_t n, const size_t k, const float* a, const float* b, float* c);
    __host__ void mat_mul_add(cublasHandle_t handle, const size_t m, const size_t n, const size_t k, const int* a, const int* b, int* c);

    __host__ void mat_mul_add(cublasLtHandle_t handle, const size_t m, const size_t n, const size_t k, const double* a, const double* b, const double* c, double* d);
    __host__ void mat_mul_add(cublasLtHandle_t handle, const size_t m, const size_t n, const size_t k, const float* a, const float* b, const float* c, float* d);
    __host__ void mat_mul_add(cublasLtHandle_t handle, const size_t m, const size_t n, const size_t k, const int* a, const int* b, const int* c, int* d);

    template<typename T>
    __host__ void multiply(cublasLtHandle_t handle, const device_matrix<T>& left, const device_vector<T>& right, device_vector<T>& result)
    {
        if(left.GetCols() != right.GetRows())
        {
            throw invalid_argument_exception("left columns does not match right rows", "right", __FILE__, __LINE__);
        }
        if(left.GetRows() != result.GetRows())
        {
            throw invalid_argument_exception("result matrix has invalid dimensions", "result", __FILE__, __LINE__);
        }
        mat_mul(handle, left.GetRows(), left.GetCols(), 1, left.Get(), right.Get(), result.Get());
    }

    template<typename T>
    __host__ void multiply_add(cublasLtHandle_t handle, const device_matrix<T>& a, const device_vector<T>& b, const device_vector<T>& c, device_vector<T>& d)
    {
        if(a.GetCols() != b.GetRows())
        {
            throw invalid_argument_exception("left columns does not match right rows", "b", __FILE__, __LINE__);
        }
        if(a.GetRows() != c.GetRows())
        {
            throw invalid_argument_exception("c matrix has invalid dimensions", "c", __FILE__, __LINE__);
        }
        if(c.GetRows() != d.GetRows())
        {
            throw invalid_argument_exception("c and d matrix not equal shape", "d", __FILE__, __LINE__);
        }
        mat_mul_add(handle, a.GetRows(), a.GetCols(), 1, a.Get(), b.Get(), c.Get(), d.Get());
    }

    template<typename T>
    __host__ void multiply_add(cublasHandle_t handle, const device_matrix<T>& a, const device_matrix<T>& b, device_matrix<T>& c)
    {
        if(a.GetCols() != b.GetRows())
        {
            throw invalid_argument_exception("left columns does not match right rows", "b", __FILE__, __LINE__);
        }
        if(a.GetRows() != c.GetRows() || b.GetCols() != c.GetCols())
        {
            throw invalid_argument_exception("c matrix has invalid dimensions", "c", __FILE__, __LINE__);
        }
        mat_mul_add(handle, a.GetRows(), a.GetCols(), b.GetCols(), a.Get(), b.Get(), c.Get());
    }

    template<typename T>
    __host__ void multiply_add(cublasHandle_t handle, const device_matrix<T>& a, const device_vector<T>& b, device_vector<T>& c)
    {
        if(a.GetCols() != b.GetRows())
        {
            throw invalid_argument_exception("left columns does not match right rows", "b", __FILE__, __LINE__);
        }
        if(a.GetRows() != c.GetRows())
        {
            throw invalid_argument_exception("c matrix has invalid dimensions", "c", __FILE__, __LINE__);
        }
        mat_mul_add(handle, a.GetRows(), a.GetCols(), 1, a.Get(), b.Get(), c.Get());
    }

    template<typename T>
    __host__ void multiply(cublasHandle_t handle, const device_matrix<T>& left, const device_matrix<T>& right, device_matrix<T>& result)
    {
        if(left.GetCols() != right.GetRows())
        {
            throw invalid_argument_exception("left columns does not match right rows", "right", __FILE__, __LINE__);
        }
        if(left.GetRows() != result.GetRows() || right.GetCols() != result.GetCols())
        {
            throw invalid_argument_exception("result matrix has invalid dimensions", "result", __FILE__, __LINE__);
        }
        mat_mul(handle, left.GetRows(), left.GetCols(), right.GetCols(), left.Get(), right.Get(), result.Get());
    }

    template<typename T>
    __host__ void multiply_add(cublasLtHandle_t handle, const device_matrix<T>& a, const device_matrix<T>& b, const device_matrix<T>& c, device_matrix<T>& d)
    {
        if(a.GetCols() != b.GetRows())
        {
            throw invalid_argument_exception("left columns does not match right rows", "b", __FILE__, __LINE__);
        }
        if(a.GetRows() != c.GetRows() || b.GetCols() != c.GetCols())
        {
            throw invalid_argument_exception("c matrix has invalid dimensions", "c", __FILE__, __LINE__);
        }
        if(c.GetRows() != d.GetRows() || c.GetCols() != d.GetCols())
        {
            throw invalid_argument_exception("c and d matrix not equal shape", "d", __FILE__, __LINE__);
        }
        mat_mul_add(handle, a.GetRows(), a.GetCols(), b.GetCols(), a.Get(), b.Get(), c.Get(), d.Get());
    }
} // namespace cuda
} // namespace icrar
#endif
