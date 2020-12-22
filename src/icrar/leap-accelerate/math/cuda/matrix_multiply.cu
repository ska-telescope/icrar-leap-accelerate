
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

#include <icrar/leap-accelerate/cuda/helper_cuda.cuh>
#include <cublas_v2.h>
#include <cublasLt.h>

#include <type_traits>

template<typename T>
struct is_cublas_supported : public std::false_type {};
template<>
struct is_cublas_supported<double> : public std::true_type {};
template<>
struct is_cublas_supported<float> : public std::true_type {};
template<>
struct is_cublas_supported<int32_t> : public std::true_type {};

namespace icrar
{
namespace cuda
{
    /**
     * @brief Performs matrix multiplcation with offset of the form C = A * B
     */
    template<typename T, typename=std::enable_if_t<is_cublas_supported<T>::value>>
    __host__ void mat_mul(cublasHandle_t handle, const size_t m, const size_t n, const size_t k, const T* A, const T* B, T* C)
    {
        const double alpha = 1.0;
        const double beta = 0.0;
        cublasOperation_t transa = cublasOperation_t::CUBLAS_OP_N;
        cublasOperation_t transb = cublasOperation_t::CUBLAS_OP_N;

        int lda = m;
        int ldb = k;
        int ldc = m;

        cublasComputeType_t computeType;
        cudaDataType_t dataType;
        if(std::is_same<T, double>::value)
        {
            computeType = CUBLAS_COMPUTE_64F;
            dataType = CUDA_R_64F;
        }
        else if(std::is_same<T, float>::value)
        {
            computeType = CUBLAS_COMPUTE_32F;
            dataType = CUDA_R_32F;
        }
        else if(std::is_same<T, std::int32_t>::value)
        {
            computeType = CUBLAS_COMPUTE_32I;
            dataType = CUDA_R_32I;
        }
        else
        {
            throw invalid_argument_exception("invalid template", "T", __FILE__, __LINE__);
        }

        checkCudaErrors(cublasGemmEx(
            handle,
            transa, 
            transb,
            m, n, k,
            &alpha,
            A, dataType, lda,
            B, dataType, ldb,
            &beta,
            C, dataType, ldc,
            computeType,
            cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    /**
     * @brief Performs matrix multiplcation with offset of the form C = (A * B) + C
     */
    template<typename T, typename=std::enable_if_t<is_cublas_supported<T>::value>>
    __host__ void mat_mul_add(cublasHandle_t handle, const size_t m, const size_t n, const size_t k, const T* A, const T* B, T* C)
    {
        const double alpha = 1.0;
        const double beta = 1.0;
        cublasOperation_t transa = cublasOperation_t::CUBLAS_OP_N;
        cublasOperation_t transb = cublasOperation_t::CUBLAS_OP_N;

        int lda = m;
        int ldb = k;
        int ldc = m;

        cublasComputeType_t computeType;
        cudaDataType_t dataType;
        if(std::is_same<T, double>::value)
        {
            computeType = CUBLAS_COMPUTE_64F;
            dataType = CUDA_R_64F;
        }
        else if(std::is_same<T, float>::value)
        {
            computeType = CUBLAS_COMPUTE_32F;
            dataType = CUDA_R_32F;
        }
        else if(std::is_same<T, std::int32_t>::value)
        {
            computeType = CUBLAS_COMPUTE_32I;
            dataType = CUDA_R_32I;
        }
        else
        {
            throw invalid_argument_exception("invalid template", "T", __FILE__, __LINE__);
        }

        checkCudaErrors(cublasGemmEx(
            handle,
            transa, 
            transb,
            m, n, k,
            &alpha,
            A, dataType, lda,
            B, dataType, ldb,
            &beta,
            C, dataType, ldc,
            computeType,
            cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    template<typename T>
    __host__ void mat_mul(cublasLtHandle_t handle, const size_t m, const size_t n, const size_t k, const T* A, const T* B, T* C)
    {
        cublasOperation_t transa = cublasOperation_t::CUBLAS_OP_N;
        cublasOperation_t transb = cublasOperation_t::CUBLAS_OP_N;

        size_t lda = m;
        size_t ldb = k;
        size_t ldc = m;

        const double alpha = 1.0;
        const double beta = 1.0;

        cublasLtMatmulDescOpaque_t operationDesc = {};
        cublasLtMatrixLayoutOpaque_t Adesc = {}, Bdesc = {}, Cdesc = {};
        cublasLtMatmulAlgo_t algo = {};

        const int32_t algoId = 10;
        const cublasLtMatmulTile_t tileId = CUBLASLT_MATMUL_TILE_16x16;
        const cublasLtReductionScheme_t reductionMode = CUBLASLT_REDUCTION_SCHEME_INPLACE;
        const int32_t splitKFactor = 256;

        // create operation desciriptor; see cublasLtMatmulDescAttributes_t for details about defaults; here we just need to
        // set the transforms for A and B

        cublasComputeType_t computeType;
        cudaDataType_t dataType;
        if(std::is_same<T, double>::value)
        {
            computeType = CUBLAS_COMPUTE_64F;
            dataType = CUDA_R_64F;
        }
        else if(std::is_same<T, float>::value)
        {
            computeType = CUBLAS_COMPUTE_32F;
            dataType = CUDA_R_32F;
        }
        else if(std::is_same<T, std::int32_t>::value)
        {
            computeType = CUBLAS_COMPUTE_32I;
            dataType = CUDA_R_32I;
        }
        else
        {
            throw invalid_argument_exception("invalid template", "T", __FILE__, __LINE__);
        }

        checkCudaErrors(cublasLtMatmulDescInit(&operationDesc, computeType, dataType));
        checkCudaErrors(cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
        checkCudaErrors(cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transa)));

        // create matrix descriptors, we are good with the details here so no need to set any extra attributes
        checkCudaErrors(cublasLtMatrixLayoutInit(&Adesc, dataType, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
        checkCudaErrors(cublasLtMatrixLayoutInit(&Bdesc, dataType, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
        checkCudaErrors(cublasLtMatrixLayoutInit(&Cdesc, dataType, m, n, ldc));

        checkCudaErrors(cublasLtMatmulAlgoInit(
            handle,
            computeType, // compute
            dataType, //scale
            dataType, // A
            dataType, // B
            dataType, // C
            dataType, // D
            algoId,
            &algo));

        checkCudaErrors(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tileId, sizeof(tileId)));
        checkCudaErrors(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reductionMode, sizeof(reductionMode)));
        checkCudaErrors(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitKFactor, sizeof(splitKFactor)));

        size_t workspaceSize = 4 * 1024 * 1024;
        void *workspace = nullptr;
        checkCudaErrors(cudaMalloc(&workspace, workspaceSize));

        cudaStream_t stream = nullptr;

        checkCudaErrors(cublasLtMatmul(
            handle,
            &operationDesc,
            &alpha,
            (void*)A,
            &Adesc,
            (void*)B,
            &Bdesc,
            &beta,
            (void*)C,
            &Cdesc,
            (void*)C,
            &Cdesc,
            &algo,
            (void*)workspace,
            workspaceSize,
            stream));

        checkCudaErrors(cudaFree(workspace));
    }

    /**
     * @brief Performs matrix multiplcation with offset of the form D = (A * B) + C 
     * 
     * @tparam T 
     * @param handle 
     * @param m 
     * @param n 
     * @param k 
     * @param A 
     * @param B 
     * @param C 
     * @param D 
     * @return __host__ 
     */
    template<typename T>
    __host__ void mat_mul_add(cublasLtHandle_t handle, const size_t m, const size_t n, const size_t k, const T* A, const T* B, const T* C, T* D)
    {
        cublasOperation_t transa = cublasOperation_t::CUBLAS_OP_N;
        cublasOperation_t transb = cublasOperation_t::CUBLAS_OP_N;

        size_t lda = m;
        size_t ldb = k;
        size_t ldc = m;

        const double alpha = 1.0;
        const double beta = 1.0;

        cublasLtMatmulDescOpaque_t operationDesc = {};
        cublasLtMatrixLayoutOpaque_t Adesc = {}, Bdesc = {}, Cdesc = {};
        cublasLtMatmulAlgo_t algo = {};

        const int32_t algoId = 10;
        const cublasLtMatmulTile_t tileId = CUBLASLT_MATMUL_TILE_16x16;
        const cublasLtReductionScheme_t reductionMode = CUBLASLT_REDUCTION_SCHEME_INPLACE;
        const int32_t splitKFactor = 256;

        // create operation desciriptor; see cublasLtMatmulDescAttributes_t for details about defaults; here we just need to
        // set the transforms for A and B

        cublasComputeType_t computeType;
        cudaDataType_t dataType;
        if(std::is_same<T, double>::value)
        {
            computeType = CUBLAS_COMPUTE_64F;
            dataType = CUDA_R_64F;
        }
        else if(std::is_same<T, float>::value)
        {
            computeType = CUBLAS_COMPUTE_32F;
            dataType = CUDA_R_32F;
        }
        else if(std::is_same<T, std::int32_t>::value)
        {
            computeType = CUBLAS_COMPUTE_32I;
            dataType = CUDA_R_32I;
        }
        else
        {
            throw invalid_argument_exception("invalid template", "T", __FILE__, __LINE__);
        }

        //LtSgemm

        checkCudaErrors(cublasLtMatmulDescInit(&operationDesc, computeType, dataType));
        checkCudaErrors(cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
        checkCudaErrors(cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transa)));

        // create matrix descriptors, we are good with the details here so no need to set any extra attributes
        checkCudaErrors(cublasLtMatrixLayoutInit(&Adesc, dataType, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
        checkCudaErrors(cublasLtMatrixLayoutInit(&Bdesc, dataType, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
        checkCudaErrors(cublasLtMatrixLayoutInit(&Cdesc, dataType, m, n, ldc));

        checkCudaErrors(cublasLtMatmulAlgoInit(
            handle,
            computeType, // compute
            dataType, //scale
            dataType, // A
            dataType, // B
            dataType, // C
            dataType, // D
            algoId,
            &algo));

        checkCudaErrors(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tileId, sizeof(tileId)));
        checkCudaErrors(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reductionMode, sizeof(reductionMode)));
        checkCudaErrors(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitKFactor, sizeof(splitKFactor)));

        size_t workspaceSize = 4 * 1024 * 1024;
        void *workspace = nullptr;
        checkCudaErrors(cudaMalloc(&workspace, workspaceSize));

        cudaStream_t stream = nullptr;

        checkCudaErrors(cublasLtMatmul(
            handle,
            &operationDesc,
            &alpha,
            (void*)A,
            &Adesc,
            (void*)B,
            &Bdesc,
            &beta,
            (void*)C,
            &Cdesc,
            (void*)D,
            &Cdesc,
            &algo,
            (void*)workspace,
            workspaceSize,
            stream));

        checkCudaErrors(cudaFree(workspace));
    }

    __host__ void mat_mul(cublasHandle_t handle, const size_t m, const size_t n, const size_t k, const double* A, const double* B, double* C)
    {
        mat_mul<double>(handle, m, n, k, A, B, C);
    }
    __host__ void mat_mul(cublasHandle_t handle, const size_t m, const size_t n, const size_t k, const float* A, const float* B, float* C)
    {
        mat_mul<float>(handle, m, n, k, A, B, C);
    }
    __host__ void mat_mul(cublasHandle_t handle, const size_t m, const size_t n, const size_t k, const int* A, const int* B, int* C)
    {
        mat_mul<int>(handle, m, n, k, A, B, C);
    }

    __host__ void mat_mul(cublasLtHandle_t handle, const size_t m, const size_t n, const size_t k, const double* A, const double* B, double* C)
    {
        mat_mul<double>(handle, m, n, k, A, B, C);
    }
    __host__ void mat_mul(cublasLtHandle_t handle, const size_t m, const size_t n, const size_t k, const float* A, const float* B, float* C)
    {
        mat_mul<float>(handle, m, n, k, A, B, C);
    }
    __host__ void mat_mul(cublasLtHandle_t handle, const size_t m, const size_t n, const size_t k, const int* A, const int* B, int* C)
    {
        mat_mul<int>(handle, m, n, k, A, B, C);
    }

    __host__ void mat_mul_add(cublasHandle_t handle, const size_t m, const size_t n, const size_t k, const double* A, const double* B, double* C)
    {
        mat_mul_add<double>(handle, m, n, k, A, B, C);
    }
    __host__ void mat_mul_add(cublasHandle_t handle, const size_t m, const size_t n, const size_t k, const float* A, const float* B, float* C)
    {
        mat_mul_add<float>(handle, m, n, k, A, B, C);
    }
    __host__ void mat_mul_add(cublasHandle_t handle, const size_t m, const size_t n, const size_t k, const int* A, const int* B, int* C)
    {
        mat_mul_add<int>(handle, m, n, k, A, B, C);
    }

    __host__ void mat_mul_add(cublasLtHandle_t handle, const size_t m, const size_t n, const size_t k, const double* A, const double* B, const double* C, double* D)
    {
        mat_mul_add<double>(handle, m, n, k, A, B, C, D);
    }
    __host__ void mat_mul_add(cublasLtHandle_t handle, const size_t m, const size_t n, const size_t k, const float* A, const float* B, const float* C, float* D)
    {
        mat_mul_add<float>(handle, m, n, k, A, B, C, D);
    }
    __host__ void mat_mul_add(cublasLtHandle_t handle, const size_t m, const size_t n, const size_t k, const int* A, const int* B, const int* C, int* D)
    {
        mat_mul_add<int>(handle, m, n, k, A, B, C, D);
    }
} // namespace cuda
} // namespace icrar
