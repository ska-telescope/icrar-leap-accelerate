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

#include "DeviceMetaData.h"
#include <icrar/leap-accelerate/math/vector_extensions.h>
#include <icrar/leap-accelerate/math/casacore_helper.h>

#include <icrar/leap-accelerate/exception/exception.h>

namespace icrar
{
namespace cuda
{
    ConstantBuffer::ConstantBuffer(
            const icrar::cpu::Constants& constants,
            const Eigen::MatrixXd& A,
            const Eigen::VectorXi& I,
            const Eigen::MatrixXd& Ad,
            const Eigen::MatrixXd& A1,
            const Eigen::VectorXi& I1,
            const Eigen::MatrixXd& Ad1)
        : m_constants(constants)
        , m_A(A)
        , m_I(I)
        , m_Ad(Ad)
        , m_A1(A1)
        , m_I1(I1)
        , m_Ad1(Ad1) { }

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

    SolutionIntervalBuffer::SolutionIntervalBuffer(const std::vector<icrar::MVuvw>& uvw)
    : m_UVW(uvw)
    {}

    SolutionIntervalBuffer::SolutionIntervalBuffer(size_t baselines)
    : m_UVW(baselines, nullptr)
    {}

    DirectionBuffer::DirectionBuffer(
        const SphericalDirection& direction,
        const Eigen::Matrix3d& dd,
        const std::vector<icrar::MVuvw>& rotatedUVW,
        const Eigen::MatrixXcd& avgData)
    : m_direction(direction)
    , m_dd(dd)
    , m_rotatedUVW(rotatedUVW)
    , m_avgData(avgData)
    {}

    DirectionBuffer::DirectionBuffer(
        int uvwSize,
        int avgDataRows,
        int avgDataCols)
    : m_rotatedUVW(uvwSize)
    , m_avgData(avgDataRows, avgDataCols)
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
        metadata.GetA(),
        metadata.GetI(),
        metadata.GetAd(),
        metadata.GetA1(),
        metadata.GetI1(),
        metadata.GetAd1()))
    , m_solutionIntervalBuffer(std::make_shared<SolutionIntervalBuffer>(
        metadata.GetUVW()))
    , m_directionBuffer(std::make_shared<DirectionBuffer>(
        metadata.GetDirection(),
        metadata.GetDD(),
        metadata.GetRotatedUVW(),
        metadata.GetAvgData()))
    {}

    DeviceMetaData::DeviceMetaData(
        std::shared_ptr<ConstantBuffer> constantBuffer,
        std::shared_ptr<SolutionIntervalBuffer> SolutionIntervalBuffer,
        std::shared_ptr<DirectionBuffer> directionBuffer)
    : m_constantBuffer(std::move(constantBuffer))
    , m_solutionIntervalBuffer(std::move(SolutionIntervalBuffer))
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

        m_solutionIntervalBuffer->GetUVW().ToHost(metadata.m_UVW);
        m_directionBuffer->GetRotatedUVW().ToHost(metadata.m_rotatedUVW);
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

    void DeviceMetaData::ToHostAsync(cpu::MetaData& host) const
    {
        throw std::runtime_error("not implemented");
    }
}
}