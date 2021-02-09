/**
 * ICRAR - International Centre for Radio Astronomy Research
 * (c) UWA - The University of Western Australia
 * Copyright by UWA(in the framework of the ICRAR)
 * All rights reserved
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston,
 * MA 02111 - 1307  USA
 */

#include <icrar/leap-accelerate/cuda/helper_cuda.cuh>
#include <icrar/leap-accelerate/math/cuda/matrix.h>
#include <icrar/leap-accelerate/tests/math/eigen_helper.h>

#include <cuda_runtime.h>

#include <gtest/gtest.h>

#include <stdio.h>
#include <array>

namespace icrar
{
    class CudaMatrixTests : public testing::Test
    {
        const double TOLERANCE = 0.0001;
        cublasHandle_t m_cublasContext;

    public:
        void SetUp() override
        {
            // See this page: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html
            int deviceCount = 0;
            checkCudaErrors(cudaGetDeviceCount(&deviceCount));
            ASSERT_EQ(1, deviceCount);

            checkCudaErrors(cublasCreate(&m_cublasContext));
        }

        void TearDown() override
        {
            checkCudaErrors(cublasCreate(&m_cublasContext));
            checkCudaErrors(cudaDeviceReset());
        }

        template<typename T>
        void TestMatrixMatrixMultiply()
        {
            using MatrixXT = Eigen::Matrix<T, -1, -1>;

            MatrixXT a = 2 * MatrixXT::Ones(16,16);
            MatrixXT b = 3 * MatrixXT::Ones(16,16);
            MatrixXT c = MatrixXT::Zero(16,16);

            auto ad = cuda::device_matrix<T>(a);
            auto bd = cuda::device_matrix<T>(b);
            auto cd = cuda::device_matrix<T>(c);

            cuda::multiply<T>(m_cublasContext, ad, bd, cd);
            cd.ToHost(c);

            MatrixXT expected = (a * b);
            ASSERT_MEQD(expected, c, TOLERANCE);
        }

        template<typename T>
        void TestMatrixMatrixMultiplyAdd()
        {
            using MatrixXT = Eigen::Matrix<T, -1, -1>;

            MatrixXT a = 2 * MatrixXT::Identity(16,16);
            MatrixXT b = 3 * MatrixXT::Identity(16,16);
            MatrixXT c = MatrixXT::Identity(16,16);
            MatrixXT d = MatrixXT::Zero(16,16);

            auto ad = cuda::device_matrix<T>(a);
            auto bd = cuda::device_matrix<T>(b);
            auto cd = cuda::device_matrix<T>(c);
            cuda::multiply_add<T>(m_cublasContext, ad, bd, cd);
            cd.ToHost(d);

            MatrixXT expected = (a * b) + c;
            ASSERT_MEQD(expected, d, TOLERANCE);
        }

        template<typename T>
        void TestMatrixMatrixMultiply32()
        {
            using MatrixXT = Eigen::Matrix<T, -1, -1>;

            auto a = MatrixXT(2,2);
            a << 1, 2,
                3, 4;

            auto b = MatrixXT(2,3);
            b << 5, 6, 7,
                8, 9, 10;

            auto c = MatrixXT(2,3);

            auto ad = cuda::device_matrix<T>(a);
            auto bd = cuda::device_matrix<T>(b);
            auto cd = cuda::device_matrix<T>(c);
            icrar::cuda::multiply(m_cublasContext, ad, bd, cd);
            cd.ToHost(c);

            MatrixXT expected = a * b;
            ASSERT_MEQD(expected, c, TOLERANCE);
        }

        template<typename T>
        void TestMatrixVectorMultiply33()
        {
            using MatrixXT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

            auto a = MatrixXT(3,3);
            a << 1, 2, 3,
                4, 5, 6,
                7, 8, 9;

            auto b = Eigen::Matrix<T, Eigen::Dynamic, 1>(3, 1);
            b << 1, 2, 3;

            auto c = Eigen::Matrix<T, Eigen::Dynamic, 1>(3, 1); 

            auto ad = cuda::device_matrix<T>(a);
            auto bd = cuda::device_vector<T>(b);
            auto cd = cuda::device_vector<T>(c);
            icrar::cuda::multiply(m_cublasContext, ad, bd, cd);
            cd.ToHost(c);

            MatrixXT expected = a * b;
            ASSERT_EQ(c, expected);
        }

        template<typename T>
        void TestScalearMatrixMultiply()
        {

        }
    };

    TEST_F(CudaMatrixTests, TestMatrixMatrixMultiply) { TestMatrixMatrixMultiply<double>(); }
    TEST_F(CudaMatrixTests, TestMatrixMatrixMultiplyAdd) { TestMatrixMatrixMultiplyAdd<double>(); }
    TEST_F(CudaMatrixTests, TestMatrixMatrixMultiply32) { TestMatrixMatrixMultiply32<double>(); }
    TEST_F(CudaMatrixTests, TestMatrixVectorMultiply33) { TestMatrixVectorMultiply33<double>(); }
    TEST_F(CudaMatrixTests, DISABLED_TestScalearMatrixMultiply) { TestScalearMatrixMultiply<double>(); }
} // namespace icrar
