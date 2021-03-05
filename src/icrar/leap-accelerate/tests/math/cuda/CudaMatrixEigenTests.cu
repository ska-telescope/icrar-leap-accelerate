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

#include <cuda_runtime.h>

#include <icrar/leap-accelerate/tests/test_helper.h>
#include <icrar/leap-accelerate/tests/math/eigen_helper.h>
#include <icrar/leap-accelerate/cuda/helper_cuda.cuh>
#include <icrar/leap-accelerate/math/cuda/vector_eigen.cuh>
#include <icrar/leap-accelerate/math/cuda/matrix.h>

#include <icrar/leap-accelerate/math/cpu/matrix_invert.h>

#include <icrar/leap-accelerate/core/log/logging.h>
#include <icrar/leap-accelerate/core/compute_implementation.h>

#include <Eigen/Core>
#include <gtest/gtest.h>
#include <stdio.h>
#include <array>

namespace icrar
{
    class CudaMatrixEigenTests : public testing::Test
    {
        double TOLERANCE = 0.1;
        cublasHandle_t m_cublasContext;
        cusolverDnHandle_t m_cusolverDnContext;

    public:
        void SetUp() override
        {
            // See this page: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html
            int deviceCount = 0;
            checkCudaErrors(cudaGetDeviceCount(&deviceCount));
            ASSERT_EQ(1, deviceCount);

            checkCudaErrors(cublasCreate(&m_cublasContext));
            checkCudaErrors(cusolverDnCreate(&m_cusolverDnContext));
        }

        void TearDown() override
        {
            checkCudaErrors(cusolverDnDestroy(m_cusolverDnContext));
            checkCudaErrors(cublasDestroy(m_cublasContext));
            checkCudaErrors(cudaDeviceReset());
        }

        void TestVectorAdd()
        {
            constexpr int N = 10;
            auto a = Eigen::Matrix<double, N, 1>();
            a << 6,6,6,6,6, 6,6,6,6,6;

            auto b = Eigen::Matrix<double, N, 1>();
            b << 10,10,10,10,10, 10,10,10,10,10;

            auto c = Eigen::Matrix<double, N, 1>();

            icrar::cuda::h_add<double, N>(a, b, c);

            auto expected = Eigen::Matrix<double, N, 1>();
            expected << 16,16,16,16,16, 16,16,16,16,16;
            ASSERT_EQ(c, expected);
        }

        void TestPseudoInverse23(cuda::JobType jobType)
        {
            constexpr int M = 2;
            constexpr int N = 3;

            auto m1 = Eigen::MatrixXd(M, N);
            m1 <<
            1, 3, 5,
            2, 4, 6;

            auto m1d = icrar::cuda::PseudoInverse(m_cusolverDnContext, m_cublasContext, m1, jobType);
            ASSERT_MEQD(m1, m1 * m1d * m1, TOLERANCE);
            ASSERT_MEQD(Eigen::MatrixXd::Identity(2,2), m1 * m1d, TOLERANCE);
        }

        void TestPseudoInverse32Degenerate()
        {
            constexpr int M = 3;
            constexpr int N = 2;

            auto m1 = Eigen::MatrixXd(M, N);
            m1 <<
            0.5, 0.5,
            -1, -1,
            -0.5, -0.5;

            auto m1d = icrar::cuda::PseudoInverse(m_cusolverDnContext, m_cublasContext, m1);

            auto expected_m1d = Eigen::MatrixXd(N, M);
            expected_m1d <<
            0.166667, -0.333333, -0.166667,
            0.166667, -0.333333, -0.166667;

            ASSERT_MEQD(expected_m1d, m1d, TOLERANCE);
            ASSERT_MEQD(m1, m1 * m1d * m1, TOLERANCE);
        }

        void TestPseudoInverse33(cuda::JobType jobType)
        {
            constexpr int M = 3;
            constexpr int N = 3;

            auto m1 = Eigen::MatrixXd(M, N);
            m1 <<
            1, 2, 3,
            4, 5, 6,
            7, 8, 9;

            ASSERT_THROW(icrar::cuda::PseudoInverse(m_cusolverDnContext, m_cublasContext, m1, jobType), icrar::invalid_argument_exception);
            //auto m1d = icrar::cuda::PseudoInverse(m_cusolverDnContext, m1, jobType);
            //ASSERT_MEQD(m1, m1 * m1d * m1, TOLERANCE);
            //ASSERT_MEQD(Eigen::MatrixXd::Identity(3,3), m1 * m1d, TOLERANCE);
        }

        void TestPseudoInverse32(cuda::JobType jobType)
        {
            constexpr int M = 3;
            constexpr int N = 2;

            auto m1 = Eigen::MatrixXd(M, N);
            m1 <<
            1, 2,
            3, 4,
            5, 6;

            Eigen::MatrixXd m1d = icrar::cuda::PseudoInverse(m_cusolverDnContext, m_cublasContext, m1, jobType);
            ASSERT_MEQD(m1, m1 * (m1d * m1), TOLERANCE);
            ASSERT_MEQD(Eigen::MatrixXd::Identity(2,2), m1d * m1, TOLERANCE);
        }

        void TestPseudoInverse42(cuda::JobType jobType)
        {
            constexpr int M = 4;
            constexpr int N = 2;

            auto m1 = Eigen::MatrixXd(M, N);
            m1 <<
            1, 2,
            3, 4,
            5, 6,
            7, 8;

            Eigen::MatrixXd m1d = icrar::cuda::PseudoInverse(m_cusolverDnContext, m_cublasContext, m1, jobType);
            ASSERT_MEQD(m1, m1 * (m1d * m1), TOLERANCE);
            ASSERT_MEQD(Eigen::MatrixXd::Identity(2,2), m1d * m1, TOLERANCE);
        }

        void TestPseudoInverseMWA(cuda::JobType jobType)
        {
            constexpr int M = 8001;
            constexpr int N = 128;

            Eigen::MatrixXd m1 = Eigen::MatrixXd::Random(M, N);
            Eigen::MatrixXd m1d = icrar::cuda::PseudoInverse(m_cusolverDnContext, m_cublasContext, m1, jobType);

            ASSERT_MEQD(m1, m1 * (m1d * m1), TOLERANCE);
            ASSERT_MEQD(Eigen::MatrixXd::Identity(N,N), m1d * m1, TOLERANCE);
        }

        void TestPseudoInverseLarge(ComputeImplementation impl)
        {
            constexpr int M = 70817;
            constexpr int N = 512;

            Eigen::MatrixXd m1 = Eigen::MatrixXd::Random(M, N);
            Eigen::MatrixXd m1d;
            if(impl == ComputeImplementation::cuda)
            {
                m1d = icrar::cuda::PseudoInverse(m_cusolverDnContext, m_cublasContext, m1, cuda::JobType::S);
            }
            else
            {
                m1d = icrar::cpu::PseudoInverse(m1);
            }
            
            //Note: for large matrices the smaller intermediate matrix is required to avoid std::bad_alloc issues
            Eigen::MatrixXd calculatedm1 = m1 * (m1d * m1);
            ASSERT_MEQD(m1, calculatedm1, TOLERANCE);
            ASSERT_MEQD(Eigen::MatrixXd::Identity(N,N), m1d * m1, TOLERANCE);
        }

        void TestPseudoInverseSKA(ComputeImplementation impl)
        {
            constexpr int M = 130817; // TODO(calgray): ska - cudamalloc error
            constexpr int N = 512;

            Eigen::MatrixXd m1 = Eigen::MatrixXd::Random(M, N);
            Eigen::MatrixXd m1d;
            if(impl == ComputeImplementation::cuda)
            {
                m1d = icrar::cuda::PseudoInverse(m_cusolverDnContext, m_cublasContext, m1, cuda::JobType::S);
            }
            else
            {
                m1d = icrar::cpu::PseudoInverse(m1);
            }
            
            //Note: for large matrices the smaller intermediate matrix is required to avoid memory issues
            Eigen::MatrixXd calculatedm1 = m1 * (m1d * m1);
            ASSERT_MEQD(m1, calculatedm1, TOLERANCE);
            ASSERT_MEQD(Eigen::MatrixXd::Identity(N,N), m1d * m1, TOLERANCE);
        }
    };

    TEST_F(CudaMatrixEigenTests, TestCudaVectorAdd10) { TestVectorAdd(); }
    TEST_F(CudaMatrixEigenTests, DISABLED_TestCudaPseudoInverse23A) { TestPseudoInverse23(cuda::JobType::A); }
    TEST_F(CudaMatrixEigenTests, DISABLED_TestCudaPseudoInverse23S) { TestPseudoInverse23(cuda::JobType::S); }
    TEST_F(CudaMatrixEigenTests, DISABLED_TestCudaPseudoInverse32Degenerate) { TestPseudoInverse32Degenerate(); }
    TEST_F(CudaMatrixEigenTests, TestCudaPseudoInverse32A) { TestPseudoInverse32(cuda::JobType::A); }
    TEST_F(CudaMatrixEigenTests, TestCudaPseudoInverse32S) { TestPseudoInverse32(cuda::JobType::S); }
    TEST_F(CudaMatrixEigenTests, TestCudaPseudoInverse33A) { TestPseudoInverse33(cuda::JobType::A); }
    TEST_F(CudaMatrixEigenTests, TestCudaPseudoInverse33S) { TestPseudoInverse33(cuda::JobType::S); }
    TEST_F(CudaMatrixEigenTests, TestCudaPseudoInverse42A) { TestPseudoInverse42(cuda::JobType::A); }
    TEST_F(CudaMatrixEigenTests, TestCudaPseudoInverse42S) { TestPseudoInverse42(cuda::JobType::S); }

    TEST_F(CudaMatrixEigenTests, TestCudaPseudoInverseMWA) { TestPseudoInverseMWA(cuda::JobType::S); }
    TEST_F(CudaMatrixEigenTests, TestPseudoInverseLarge) { TestPseudoInverseLarge(ComputeImplementation::cpu); }
    TEST_F(CudaMatrixEigenTests, TestCudaPseudoInverseLarge) { TestPseudoInverseLarge(ComputeImplementation::cuda); }
    TEST_F(CudaMatrixEigenTests, TestPseudoInverseSKA) { TestPseudoInverseSKA(ComputeImplementation::cpu); }
    TEST_F(CudaMatrixEigenTests, DISABLED_TestCudaPseudoInverseSKA) { TestPseudoInverseSKA(ComputeImplementation::cuda); }
}
