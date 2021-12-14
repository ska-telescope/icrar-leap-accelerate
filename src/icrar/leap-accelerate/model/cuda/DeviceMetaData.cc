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

#if CUDA_ENABLED

#include "DeviceMetaData.h"
#include <icrar/leap-accelerate/math/vector_extensions.h>
#include <icrar/leap-accelerate/math/casacore_helper.h>

#include <icrar/leap-accelerate/exception/exception.h>
#include <icrar/leap-accelerate/math/cpu/matrix_invert.h>
#include <icrar/leap-accelerate/math/cuda/matrix_invert.h>

namespace icrar
{
namespace cuda
{
    ConstantBuffer::ConstantBuffer(
            const icrar::cpu::Constants& constants,
            device_matrix<double>&& A,
            device_vector<int>&& I,
            device_matrix<double>&& Ad,
            device_matrix<double>&& A1,
            device_vector<int>&& I1,
            device_matrix<double>&& Ad1)
        : m_constants(constants)
        , m_A(std::move(A))
        , m_I(std::move(I))
        , m_Ad(std::move(Ad))
        , m_A1(std::move(A1))
        , m_I1(std::move(I1))
        , m_Ad1(std::move(Ad1))
        { }

    void ConstantBuffer::ToHost(icrar::cpu::MetaData& host) const
    {
        host.m_constants = m_constants;

        m_A.ToHost(host.m_A);
        m_I.ToHost(host.m_I);
        m_Ad.ToHost(host.m_Ad);
        m_A1.ToHost(host.m_A1);
        m_I1.ToHost(host.m_I1);
        m_Ad1.ToHost(host.m_Ad1);
    }

    void ConstantBuffer::ToHostAsync(icrar::cpu::MetaData& host) const
    {
        host.m_constants = m_constants;

        m_A.ToHostAsync(host.m_A);
        m_I.ToHostAsync(host.m_I);
        m_Ad.ToHostAsync(host.m_Ad);
        m_A1.ToHostAsync(host.m_A1);
        m_I1.ToHostAsync(host.m_I1);
        m_Ad1.ToHostAsync(host.m_Ad1);
    }

    DirectionBuffer::DirectionBuffer(
        const SphericalDirection& direction,
        const Eigen::Matrix3d& dd,
        const Eigen::MatrixXcd& avgData)
    : m_direction(direction)
    , m_dd(dd)
    , m_avgData(avgData)
    {}

    DirectionBuffer::DirectionBuffer(
        int avgDataRows,
        int avgDataCols)
    : m_avgData(avgDataRows, avgDataCols)
    {}

    void DirectionBuffer::SetDirection(const SphericalDirection& direction)
    {
        m_direction = direction;
    }

    void DirectionBuffer::SetDD(const Eigen::Matrix3d& dd)
    {
        m_dd = dd;
    }

    DeviceMetaData::DeviceMetaData(const cpu::MetaData& metadata)
    : m_constantBuffer(std::make_shared<ConstantBuffer>(
        metadata.GetConstants(),
        device_matrix<double>(metadata.GetA()),
        device_vector<int>(metadata.GetI()),
        device_matrix<double>(metadata.GetAd()),
        device_matrix<double>(metadata.GetA1()),
        device_vector<int>(metadata.GetI1()),
        device_matrix<double>(metadata.GetAd1())))
    , m_directionBuffer(std::make_shared<DirectionBuffer>(
        metadata.GetDirection(),
        metadata.GetDD(),
        metadata.GetAvgData()))
    {}

    DeviceMetaData::DeviceMetaData(
        std::shared_ptr<ConstantBuffer> constantBuffer,
        std::shared_ptr<DirectionBuffer> directionBuffer)
    : m_constantBuffer(std::move(constantBuffer))
    , m_directionBuffer(std::move(directionBuffer))
    {}

    const icrar::cpu::Constants& DeviceMetaData::GetConstants() const
    {
        return m_constantBuffer->GetConstants();
    }

    void DeviceMetaData::SetAvgData(int v)
    {
        cudaMemset(m_directionBuffer->GetAvgData().Get(), v, m_directionBuffer->GetAvgData().GetSize());
    }

    void DeviceMetaData::ToHost(cpu::MetaData& metadata) const
    {
        m_constantBuffer->ToHost(metadata);
        metadata.m_direction = m_directionBuffer->GetDirection();
        metadata.m_dd = m_directionBuffer->GetDD();
        m_directionBuffer->GetAvgData().ToHost(metadata.m_avgData);
    }

    cpu::MetaData DeviceMetaData::ToHost() const
    {
        cpu::MetaData result = cpu::MetaData();
        ToHost(result);
        return result;
    }

    void DeviceMetaData::ToHostAsync(cpu::MetaData& metadata) const
    {
        m_constantBuffer->ToHostAsync(metadata);

        metadata.m_direction = m_directionBuffer->GetDirection();
        metadata.m_dd = m_directionBuffer->GetDD();
        m_directionBuffer->GetAvgData().ToHostVectorAsync(metadata.m_avgData);
    }
} // namespace cuda
} // namespace icrar
#endif // CUDA_ENABLED
