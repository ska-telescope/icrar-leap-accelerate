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

#ifdef CUDA_ENABLED

#include <icrar/leap-accelerate/model/cpu/MVuvw.h>
#include <icrar/leap-accelerate/common/SphericalDirection.h>

#include <icrar/leap-accelerate/common/constants.h>
#include <icrar/leap-accelerate/model/cpu/MetaData.h>

#include <icrar/leap-accelerate/cuda/device_vector.h>
#include <icrar/leap-accelerate/cuda/device_matrix.h>

#include <casacore/measures/Measures/MDirection.h>
#include <casacore/casa/Quanta/MVuvw.h>

#include <casacore/casa/Arrays/Matrix.h>
#include <casacore/casa/Arrays/Vector.h>

#include <Eigen/Core>

#include <boost/optional.hpp>

#include <memory>
#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <complex>

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuComplex.h>

namespace icrar
{
namespace cuda
{
    /**
     * Container of uniform gpu buffers available to all cuda
     * threads that are const/immutable per calibration.
     */
    class ConstantBuffer
    {
        icrar::cpu::Constants m_constants;
        
        device_matrix<double> m_A;
        device_vector<int> m_I;
        device_matrix<double> m_Ad;

        device_matrix<double> m_A1;
        device_vector<int> m_I1;
        device_matrix<double> m_Ad1;

    public:
        ConstantBuffer(
            const icrar::cpu::Constants& constants,
            const Eigen::MatrixXd& A,
            const Eigen::VectorXi& I,
            const Eigen::MatrixXd& Ad,
            const Eigen::MatrixXd& A1,
            const Eigen::VectorXi& I1,
            const Eigen::MatrixXd& Ad1);
        
        /**
         * @brief Construct a new Constant Buffer object and compute inverse matrices
         * 
         * @param constants 
         * @param A 
         * @param I 
         * @param A1 
         * @param I1 
         */
        ConstantBuffer(
            cusolverDnHandle_t& cusolverHandle,
            cublasHandle_t& cublasHandle,
            const icrar::cpu::Constants& constants,
            const Eigen::MatrixXd& A,
            const Eigen::VectorXi& I,
            const Eigen::MatrixXd& A1,
            const Eigen::VectorXi& I1);

        const icrar::cpu::Constants& GetConstants() const { return m_constants; }
        const device_matrix<double>& GetA() const { return m_A; } 
        const device_vector<int>& GetI() const { return m_I; }
        const device_matrix<double>& GetAd() const { return m_Ad; }
        const device_matrix<double>& GetA1() const { return m_A1; }
        const device_vector<int>& GetI1() const { return m_I1; }
        const device_matrix<double>& GetAd1() const { return m_Ad1; }

        void ToHost(icrar::cpu::MetaData& host) const;
    };

    /**
     * @brief MetaData variables allocated per solution interval 
     * 
     */
    class SolutionIntervalBuffer
    {
        device_vector<icrar::MVuvw> m_UVW;
    public:
        explicit SolutionIntervalBuffer(const std::vector<icrar::MVuvw>& UVW);
        explicit SolutionIntervalBuffer(size_t baselines);
        
        const device_vector<icrar::MVuvw>& GetUVW() const { return m_UVW; }

        void SetUVW(std::vector<icrar::MVuvw> UVW)
        {
            assert(UVW.size() == m_UVW.GetCount());
            m_UVW.SetDataAsync(UVW.data());
        }
    };

    /**
     * @brief MetaData Variables allocated per direction
     * 
     */
    class DirectionBuffer
    {
        SphericalDirection m_direction;
        Eigen::Matrix3d m_dd;

        device_vector<icrar::MVuvw> m_rotatedUVW;
        device_matrix<std::complex<double>> m_avgData;

    public:
        /**
         * @brief Constructs a new Direction Buffer object initializing all memory
         * 
         * @param uvw 
         * @param direction 
         * @param dd 
         * @param avgData 
         */
        DirectionBuffer(
            const SphericalDirection& direction,
            const Eigen::Matrix3d& dd,
            const std::vector<icrar::MVuvw>& rotatedUVW,
            const Eigen::MatrixXcd& avgData);

        /**
         * @brief Constructs a new Direction Buffer object for late initialization
         * 
         * @param uvwSize 
         * @param avgDataRows 
         * @param avgDataCols 
         */
        DirectionBuffer(
            int uvwSize,
            int avgDataRows,
            int avgDataCols);

        const SphericalDirection& GetDirection() const { return m_direction; }
        const device_vector<icrar::MVuvw>& GetRotatedUVW() const { return m_rotatedUVW; }
        const Eigen::Matrix3d& GetDD() const { return m_dd; }

        device_vector<icrar::MVuvw>& GetRotatedUVW() { return m_rotatedUVW; }
        device_matrix<std::complex<double>>& GetAvgData() { return m_avgData; }

        void SetDirection(const SphericalDirection& direction);
        void SetDD(const Eigen::Matrix3d& dd);
    };

    /**
     * Represents the complete collection of MetaData that
     * resides on the GPU for leap-calibration
     */
    class DeviceMetaData
    {
        std::shared_ptr<ConstantBuffer> m_constantBuffer; // Constant buffer, never null
        std::shared_ptr<SolutionIntervalBuffer> m_solutionIntervalBuffer;
        std::shared_ptr<DirectionBuffer> m_directionBuffer;

    public:
        DeviceMetaData(DeviceMetaData&& other) noexcept = default;
        DeviceMetaData& operator=(DeviceMetaData&& other) noexcept = default;

        /**
         * @brief Construct a new Device MetaData object from the equivalent object on CPU memory. This copies to
         * all device buffers
         * 
         * @param metadata 
         */
        explicit DeviceMetaData(const icrar::cpu::MetaData& metadata);
        
        /**
         * @brief Construct a new Device MetaData object from the equivalent object on CPU memory. This copies to
         * all device buffers
         * 
         * @param constantBuffer 
         * @param SolutionIntervalBuffer 
         * @param directionBuffer 
         */
        DeviceMetaData(
            std::shared_ptr<ConstantBuffer> constantBuffer,
            std::shared_ptr<SolutionIntervalBuffer> SolutionIntervalBuffer,
            std::shared_ptr<DirectionBuffer> directionBuffer);


        const icrar::cpu::Constants& GetConstants() const;

        const device_vector<icrar::MVuvw>& GetUVW() const { return m_solutionIntervalBuffer->GetUVW(); }
        const device_vector<icrar::MVuvw>& GetRotatedUVW() const { return m_directionBuffer->GetRotatedUVW(); }
        const SphericalDirection& GetDirection() const { return m_directionBuffer->GetDirection(); }
        const Eigen::Matrix3d& GetDD() const { return m_directionBuffer->GetDD(); }
        
        const ConstantBuffer& GetConstantBuffer() const { return *m_constantBuffer; }
        const device_matrix<std::complex<double>>& GetAvgData() { return m_directionBuffer->GetAvgData(); };

        void SetAvgData(int v);

        void ToHost(icrar::cpu::MetaData& host) const;
        icrar::cpu::MetaData ToHost() const;
        void ToHostAsync(icrar::cpu::MetaData& host) const;
    };
}
}

#endif // CUDA_ENABLED