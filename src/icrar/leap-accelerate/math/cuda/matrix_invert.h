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

#pragma once

#include <icrar/leap-accelerate/cuda/device_vector.h>
#include <icrar/leap-accelerate/cuda/device_matrix.h>

#include <cusolverDn.h>
#include <Eigen/Dense>

namespace icrar
{
namespace cuda
{
    /**
     * @brief Corresponds to job types of CusolverDn API (e.g. cusolverDnDgesvd)
     * 
     */
    enum class JobType : signed char
    {
        A = 'A', ///< All - Entire dense matrix is used
        S = 'S' ///< Slim/Thin - Minimal matrix dimensions
        // T = 'T' Truncated
    };

    /**
     * @brief Computes the moore penrose pseudo inverse where A'A = I (left inverse)
     * 
     * @param cusolverHandle 
     * @param cublasHandle
     * @param a 
     * @param jobtype SVD matrix dimension type
     * @return Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> 
     */
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> pseudo_inverse(
        cusolverDnHandle_t cusolverHandle,
        cublasHandle_t cublasHandle,
        const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& a,
        const JobType jobtype = JobType::S);

    /**
     * @brief Computes the U, S and Vt values of matrix singular value decomposition
     * 
     * @param cusolverHandle 
     * @param deviceA 
     * @param jobType 
     * @return std::tuple<device_matrix<double>, device_vector<double>, device_matrix<double>> 
     */
    std::tuple<device_matrix<double>, device_vector<double>, device_matrix<double>> svd(
        cusolverDnHandle_t cusolverHandle,
        const device_matrix<double>& deviceA,
        const JobType jobType);

    /**
     * @brief Performs matrix inversion using cusolver and cublas 
     * 
     * @param cusolverHandle 
     * @param cublasHandle 
     * @param matrix 
     * @param jobType 
     * @return device_matrix<double> 
     */
    device_matrix<double> pseudo_inverse(
        cusolverDnHandle_t cusolverHandle,
        cublasHandle_t cublasHandle,
        const device_matrix<double>& matrix,
        const JobType jobType = JobType::S);

    // template<typename T>
    // device_matrix<T> pseudo_inverse(
    //     cusolverDnHandle_t cusolverHandle,
    //     cublasHandle_t cublasHandle,
    //     const device_matrix<T>& matrix,
    //     const JobType jobType = JobType::S);
} // namespace cuda
} // namespace icrar
