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

#include <icrar/leap-accelerate/config.h>
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>


#ifdef CUDA_ENABLED
#include <thrust/complex.h>
#else
namespace thrust
{
    template<typename Scalar>
    using complex = std::complex<Scalar>;
} // namespace thrust
#endif // CUDA_ENABLED

namespace Eigen
{
    using MatrixXb = Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorXb = Eigen::Vector<bool, Eigen::Dynamic>;

    template<typename Scalar>
    auto EIGEN_DEVICE_FUNC ToMatrix(const Eigen::Tensor<Scalar, 2>& tensor)
    {
        return Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>(
            tensor.data(),
            tensor.dimension(0),
            tensor.dimension(2)
        );
    }

    template<typename Scalar>
    auto EIGEN_DEVICE_FUNC ToVector(const Eigen::Tensor<Scalar, 1>& tensor)
    {
        return Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>(
            tensor.data(),
            tensor.dimension(0)
        );
    }

    namespace internal
    {
        template<>
        inline EIGEN_DEVICE_FUNC std::complex<double> cast(const std::complex<float>& x)
        {
            return thrust::complex<double>(x);
        }
    } // namespace internal
} // namespace Eigen
