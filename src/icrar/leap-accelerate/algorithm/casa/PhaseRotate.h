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

#include <icrar/leap-accelerate/model/casa/CalibrateResult.h>

#include <casacore/casa/Arrays/Vector.h>
#include <casacore/casa/Arrays/Matrix.h>
#include <casacore/ms/MeasurementSets.h>

#include <casacore/casa/Quanta/MVDirection.h>
#include <casacore/casa/Quanta/MVuvw.h>

#include <boost/optional.hpp>

#include <string>
#include <memory>
#include <vector>
#include <complex>
#include <queue>

namespace icrar
{
    class MeasurementSet;

    namespace casalib
    {
        class Integration;
        class IntegrationResult;
        class CalibrationResult;
    }
}

namespace icrar
{
namespace casalib
{
    struct MetaData;

    /**
     * @copydoc icrar::cpu::Calibrate
     * 
     */
    CalibrateResult Calibrate(
        const icrar::MeasurementSet& ms,
        const std::vector<casacore::MVDirection>& directions,
        double minimumBaselineThreshold);

    /**
     * @copydoc icrar::cpu::PhaseRotate
     * 
     */
    void PhaseRotate(
        MetaData& metadata,
        const casacore::MVDirection& directions,
        std::queue<Integration>& input,
        std::queue<IntegrationResult>& output_integrations,
        std::queue<CalibrationResult>& output_calibrations);

    /**
     * @copydoc icrar::cpu::RotateVisibilities
     * 
     * @param integration 
     * @param metadata 
     * @param direction 
     */
    void RotateVisibilities(Integration& integration, MetaData& metadata, const casacore::MVDirection& direction);
}
}
