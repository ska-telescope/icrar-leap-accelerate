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

#include <casacore/ms/MeasurementSets.h>

#include <eigen3/Eigen/Core>

#include <string>
#include <memory>
#include <vector>
#include <complex>
#include <queue>

namespace casacore
{
    class MDirection;
    class MVDirection;
    class MVuvw;
}

namespace icrar
{
    class Integration;
    class IntegrationResult;
    class CalibrationResult;
    
    namespace cuda
    {
        struct MetaDataCudaHost;
    }
}

namespace icrar
{
namespace cpu
{
    /**
     * @brief 
     * 
     * @param metadata 
     * @param directions 
     */
    void RemoteCalibration(
        cuda::MetaDataCudaHost& metadata,
        const Eigen::Matrix<casacore::MVDirection, Eigen::Dynamic, 1>& directions);

    /**
     * @brief 
     * 
     * @param metadata 
     * @param directions 
     * @param input 
     */
    void PhaseRotate(
        cuda::MetaDataCudaHost& metadata,
        const casacore::MVDirection& directions,
        std::queue<Integration>& input,
        std::queue<IntegrationResult>& output_integrations,
        std::queue<CalibrationResult>& output_calibrations);

    /**
     * @brief 
     * 
     * @param integration 
     * @param metadata 
     * @param direction 
     */
    void RotateVisibilities(
        Integration& integration,
        cuda::MetaDataCudaHost& metadata,
        const casacore::MVDirection& direction);

    /**
     * @brief Form Phase Matrix
     * Given the antenna lists from MS and (optionally) RefAnt & Map:
     * If non-negative RefAnt is provided it only forms the matrix for baselines with that antenna.
     * If True Map is provided it returns the index map for the matrix (only useful if RefAnt set).
     *
     * This function generates and returns the linear matrix for the phase calibration (only)
     * @param a1 
     * @param a2 
     * @param refAnt the reference antenna (0, 1), -1 
     * @param map 
     * @return std::pair<Matrixd, Matrixi> 
     */
    std::pair<Eigen::MatrixXd, Eigen::VectorXi> PhaseMatrixFunction(
        const Eigen::VectorXi& a1,
        const Eigen::VectorXi& a2,
        int refAnt=-1,
        bool map=false);
}
}