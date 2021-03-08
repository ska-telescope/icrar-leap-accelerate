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

#include "icrar/leap-accelerate/math/cuda/matrix_invert.h"

#include <cusolver_common.h>
#include <cusolverDn.h>
#include <icrar/leap-accelerate/cuda/helper_cuda.cuh>
#include <icrar/leap-accelerate/exception/exception.h>
#include <icrar/leap-accelerate/core/ioutils.h>

#include <icrar/leap-accelerate/cuda/device_matrix.h>
#include <icrar/leap-accelerate/cuda/device_vector.h>

#include <icrar/leap-accelerate/math/cuda/matrix_multiply.h>

#include <Eigen/Dense>
#include <Eigen/LU>

#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <complex>
#include <queue>
#include <limits>

#include <icrar/leap-accelerate/core/log/logging.h>
#include <icrar/leap-accelerate/common/eigen_stringutils.h>

namespace icrar
{
namespace cuda
{
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> PseudoInverse(
        cusolverDnHandle_t cusolverHandle,
        cublasHandle_t cublasHandle,
        const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& matrix,
        const JobType jobType)
    {
        size_t m = matrix.rows();
        size_t n = matrix.cols();
        size_t k = std::min(m, n);
        if(m <= n)
        {
            throw invalid_argument_exception("m<=n not supported", "matrix", __FILE__, __LINE__);
        }

        signed char jobu = static_cast<std::underlying_type<decltype(jobType)>::type>(jobType);
        signed char jobvt = static_cast<std::underlying_type<decltype(jobType)>::type>(jobType);
        
        int ldu = m;
        int lda = m;
        int ldvt = n;

        Eigen::MatrixXd U;
        if(jobType == JobType::A)
        {
            U = Eigen::MatrixXd::Zero(ldu, m);
        }
        else if(jobType == JobType::S)
        {
            U = Eigen::MatrixXd::Zero(ldu, k);
        }
        else
        {
            throw invalid_argument_exception("Unsupported argument", "jobu", __FILE__, __LINE__);
        }


        Eigen::MatrixXd Vt;
        if(jobType == JobType::A)
        {
            Vt = Eigen::MatrixXd::Zero(ldvt, n);
        }
        else if(jobType == JobType::S)
        {
            ldvt = k;
            Vt = Eigen::MatrixXd::Zero(ldvt, k);
        }
        else
        {
            throw invalid_argument_exception("Unsupported argument", "jobvt", __FILE__, __LINE__);
        }
        Eigen::VectorXd S = Eigen::VectorXd::Zero(k);

        size_t free;
        size_t total;
        checkCudaErrors(cudaMemGetInfo(&free, &total));
        LOG(info) << "free memory: " << memory_amount(free) << "/" << memory_amount(total); 
        LOG(info) << "cuda svd allocation (" << m << ", " << n << "): "
        << memory_amount((matrix.size() + U.size() + Vt.size() + S.size()) * sizeof(double));


        auto d_U = device_matrix<double>(U.rows(), U.cols());
        auto d_Vt = device_matrix<double>(Vt.rows(), Vt.cols());

        // Solve U, S, Vt with A
        // https://stackoverflow.com/questions/17401765/parallel-implementation-for-multiple-svds-using-cuda
        {
            auto d_A = device_matrix<double>(matrix);
            auto d_S = device_vector<double>(S.size());
            
            //gesvdjInfo_t gesvdjParams = nullptr;
            //cusolveSafeCall(cusolverDnCreateGesvdjInfo(&gesvdjParams));

            int* d_devInfo;
            size_t d_devInfoSize = sizeof(std::remove_pointer_t<decltype(d_devInfo)>());
            checkCudaErrors(cudaMalloc(&d_devInfo, d_devInfoSize));

            // --- Set the computation tolerance, since the default tolerance is machine precision
            //cusolveSafeCall(cusolverDnXgesvdjSetTolerance(gesvdj_params, tol));

            // --- Set the maximum number of sweeps, since the default value of max. sweeps is 100
            //cusolveSafeCall(cusolverDnXgesvdjSetMaxSweeps(gesvdj_params, maxSweeps));

            int workSize = 0;
            checkCudaErrors(cusolverDnDgesvd_bufferSize(cusolverHandle, m, n, &workSize));
            LOG(info) << "inverse matrix cuda worksize: " << memory_amount(workSize * sizeof(double));
            double* d_work; checkCudaErrors(cudaMalloc(&d_work, workSize * sizeof(double)));

            LOG(info) << "inverse matrix cuda rworksize: " << memory_amount((m-1) * sizeof(double));
            double* d_rwork; checkCudaErrors(cudaMalloc(&d_rwork, (m-1) * sizeof(double)));

            int h_devInfo = 0;
            checkCudaErrors(cusolverDnDgesvd(
                cusolverHandle,
                jobu, jobvt,
                m, n,
                d_A.Get(),
                lda,
                d_S.Get(),
                d_U.Get(),
                ldu,
                d_Vt.Get(),
                ldvt,
                d_work,
                workSize,
                d_rwork,
                d_devInfo));
            checkCudaErrors(cudaMemcpy(&h_devInfo, d_devInfo, d_devInfoSize, cudaMemcpyDeviceToHost));

            if(h_devInfo != 0)
            {
                std::stringstream ss;
                ss << "devInfo: " << h_devInfo;
                throw icrar::exception(ss.str(), __FILE__, __LINE__);
            }

            d_S.ToHostAsync(S.data()); // Sigma currently converted to matrix on cpu
            checkCudaErrors(cudaFree(d_devInfo));

            //cusolverDnDestroyGesvdjInfo(&);
        }

        cudaThreadSynchronize();
        double epsilon = std::numeric_limits<typename Eigen::MatrixXd::Scalar>::epsilon();
        double tolerance = epsilon * std::max(matrix.cols(), matrix.rows()) * S.array().abs()(0);

        Eigen::MatrixXd Sd;
        if(jobType == JobType::A)
        {
            Sd = Eigen::MatrixXd::Zero(n, m);
        }
        else if(jobType == JobType::S)
        {
            Sd = Eigen::MatrixXd::Zero(n, k);
        }
        else
        {
            throw invalid_argument_exception("Unsupported argument", "jobType", __FILE__, __LINE__);
        }
        Sd.topLeftCorner(k, k) = (S.array().abs() > tolerance).select(S.array().inverse(), 0).matrix().asDiagonal();


        auto d_Sd = device_matrix<double>(Sd);
        auto d_result = device_matrix<double>(Sd.rows(), U.rows());
        
        // result = V * (S * Ut)
        icrar::cuda::multiply(cublasHandle, d_Sd, d_U, d_result, MatrixOp::normal, MatrixOp::hermitian);
        icrar::cuda::multiply(cublasHandle, d_Vt, d_result, d_result, MatrixOp::transpose, MatrixOp::normal);

        auto VSUt = Eigen::MatrixXd(matrix.cols(), matrix.rows());
        d_result.ToHostAsync(VSUt.data());

        //Inverse = V * Sd * Ut
        return VSUt;
    }
    
    device_matrix<double> PseudoInverse(
        cusolverDnHandle_t cusolverHandle,
        cublasHandle_t cublasHandle,
        const device_matrix<double>& d_A,
        const JobType jobType)
    {
        size_t m = d_A.GetRows();
        size_t n = d_A.GetCols();
        size_t k = std::min(m, n);
        if(m <= n)
        {
            std::stringstream ss;
            ss << "matrix inverse (" << m << "," << n << ") " << "m<=n not supported";
            throw invalid_argument_exception(ss.str(), "d_A", __FILE__, __LINE__);
        }

        signed char jobu = static_cast<std::underlying_type<decltype(jobType)>::type>(jobType);
        signed char jobvt = static_cast<std::underlying_type<decltype(jobType)>::type>(jobType);
        
        int ldu = m;
        int lda = m;
        int ldvt = n;

        Eigen::MatrixXd U;
        if(jobType == JobType::A)
        {
            U = Eigen::MatrixXd::Zero(ldu, m);
        }
        else if(jobType == JobType::S)
        {
            U = Eigen::MatrixXd::Zero(ldu, k);
        }
        else
        {
            throw invalid_argument_exception("Unsupported argument", "jobu", __FILE__, __LINE__);
        }


        Eigen::MatrixXd Vt;
        if(jobType == JobType::A)
        {
            Vt = Eigen::MatrixXd::Zero(ldvt, n);
        }
        else if(jobType == JobType::S)
        {
            ldvt = k;
            Vt = Eigen::MatrixXd::Zero(ldvt, k);
        }
        else
        {
            throw invalid_argument_exception("Unsupported argument", "jobvt", __FILE__, __LINE__);
        }
        Eigen::VectorXd S = Eigen::VectorXd::Zero(k);

        size_t free;
        size_t total;
        checkCudaErrors(cudaMemGetInfo(&free, &total));
        LOG(info) << "free memory: " << memory_amount(free) << "/" << memory_amount(total); 
        LOG(info) << "cuda svd allocation (" << m << ", " << n << "): "
        << memory_amount((U.size() + Vt.size() + S.size()) * sizeof(double));

        auto d_U = device_matrix<double>(U.rows(), U.cols());
        auto d_Vt = device_matrix<double>(Vt.rows(), Vt.cols());

        // Solve U, S, Vt with A
        // https://stackoverflow.com/questions/17401765/parallel-implementation-for-multiple-svds-using-cuda
        {
            auto d_S = device_vector<double>(S.size());
            
            //gesvdjInfo_t gesvdjParams = nullptr;
            //cusolveSafeCall(cusolverDnCreateGesvdjInfo(&gesvdjParams));

            int* d_devInfo;
            size_t d_devInfoSize = sizeof(std::remove_pointer_t<decltype(d_devInfo)>());
            checkCudaErrors(cudaMalloc(&d_devInfo, d_devInfoSize));

            // --- Set the computation tolerance, since the default tolerance is machine precision
            //cusolveSafeCall(cusolverDnXgesvdjSetTolerance(gesvdj_params, tol));

            // --- Set the maximum number of sweeps, since the default value of max. sweeps is 100
            //cusolveSafeCall(cusolverDnXgesvdjSetMaxSweeps(gesvdj_params, maxSweeps));

            int workSize = 0;
            checkCudaErrors(cusolverDnDgesvd_bufferSize(cusolverHandle, m, n, &workSize));
            LOG(info) << "inverse matrix cuda worksize: " << memory_amount(workSize * sizeof(double));
            double* d_work; checkCudaErrors(cudaMalloc(&d_work, workSize * sizeof(double)));

            LOG(info) << "inverse matrix cuda rworksize: " << memory_amount((m-1) * sizeof(double));
            double* d_rwork; checkCudaErrors(cudaMalloc(&d_rwork, (m-1) * sizeof(double)));

            int h_devInfo = 0;
            checkCudaErrors(cusolverDnDgesvd(
                cusolverHandle,
                jobu, jobvt,
                m, n,
                (double*)d_A.Get(), // TODO cast be be deprecated in newer function
                lda,
                d_S.Get(),
                d_U.Get(),
                ldu,
                d_Vt.Get(),
                ldvt,
                d_work,
                workSize,
                d_rwork,
                d_devInfo));
            checkCudaErrors(cudaMemcpy(&h_devInfo, d_devInfo, d_devInfoSize, cudaMemcpyDeviceToHost));

            if(h_devInfo != 0)
            {
                std::stringstream ss;
                ss << "devInfo: " << h_devInfo;
                throw icrar::exception(ss.str(), __FILE__, __LINE__);
            }

            d_S.ToHostAsync(S.data()); // Sigma currently converted to matrix on cpu
            checkCudaErrors(cudaFree(d_devInfo));

            //cusolverDnDestroyGesvdjInfo(&);
        }

        cudaThreadSynchronize();
        double epsilon = std::numeric_limits<typename Eigen::MatrixXd::Scalar>::epsilon();
        double tolerance = epsilon * std::max(d_A.GetCols(), d_A.GetRows()) * S.array().abs()(0);

        Eigen::MatrixXd Sd;
        if(jobType == JobType::A)
        {
            Sd = Eigen::MatrixXd::Zero(n, m);
        }
        else if(jobType == JobType::S)
        {
            Sd = Eigen::MatrixXd::Zero(n, k);
        }
        else
        {
            throw invalid_argument_exception("Unsupported argument", "jobType", __FILE__, __LINE__);
        }
        Sd.topLeftCorner(k, k) = (S.array().abs() > tolerance).select(S.array().inverse(), 0).matrix().asDiagonal();


        auto d_Sd = device_matrix<double>(Sd);
        auto d_result = device_matrix<double>(Sd.rows(), U.rows());
        
        // result = V * (S * Ut)
        icrar::cuda::multiply(cublasHandle, d_Sd, d_U, d_result, MatrixOp::normal, MatrixOp::hermitian);
        icrar::cuda::multiply(cublasHandle, d_Vt, d_result, d_result, MatrixOp::transpose, MatrixOp::normal);
        return d_result;
    }
} // namespace cuda
} // namespace icrar
