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

#include <gsl/gsl_linalg.h>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/SVD>

#include <utility>

namespace icrar
{
namespace cpu
{
    /**
     * @brief Calculates the singular value decomposition values u, eigen values, v
     * 
     * @param mat 
     * @return std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::MatrixXd> s sigma v
     */
    std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::MatrixXd> SVD(const Eigen::MatrixXd& mat);

    /**
     * @brief * @brief SVD returning U, Sigma, V
     * 
     * @param mat 
     * @return std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> 
     */
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> SVDSigma(const Eigen::MatrixXd& mat);

    /**
     * @brief SVD returning U, Sigma Psuedoinverse, V
     * 
     * @param mat 
     * @return std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> 
     */
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> SVDSigmaP(const Eigen::MatrixXd& mat);

    std::tuple<Eigen::MatrixXd, Eigen::SparseMatrix<double>, Eigen::MatrixXd> SVDSparse(const Eigen::MatrixXd& mat);

    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> SVD_gsl(Eigen::MatrixXd& mat);
}
}