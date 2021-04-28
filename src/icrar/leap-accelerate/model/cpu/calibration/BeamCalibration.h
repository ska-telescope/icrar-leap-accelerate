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

#include <icrar/leap-accelerate/common/SphericalDirection.h>
#include <icrar/leap-accelerate/exception/exception.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>

#include <boost/optional.hpp>
#include <boost/noncopyable.hpp>

namespace icrar
{
namespace cpu
{
    /**
     * @brief Contains the results of leap calibration for a single direction
     * 
     */
    class BeamCalibration
    {
        SphericalDirection m_direction;
        Eigen::MatrixXd m_calibration;

    public:
        /**
         * @brief Construct a new Direction Calibration object
         * 
         * @param direction direciton of calibration
         * @param calibration calibration of each antenna for the given direction 
         */
        BeamCalibration(SphericalDirection direction, Eigen::MatrixXd calibration);

        BeamCalibration(const std::pair<SphericalDirection, Eigen::MatrixXd>& beamCalibration);

        bool IsApprox(const BeamCalibration& beamCalibration, double threshold);

        /**
         * @brief Gets the calibration direction
         * 
         * @return const SphericalDirection 
         */
        const SphericalDirection& GetDirection() const;

        /**
         * @brief Get the phase calibration Vector for the antenna array in the specified direction
         * 
         * @return const Eigen::MatrixXd 
         */
        const Eigen::MatrixXd& GetPhaseCalibration() const;

        /**
         * @brief Serializes the beam calibration to JSON format
         * 
         * @param os JSON output stream
         */
        void Serialize(std::ostream& os, bool pretty = false) const;

        void Write(rapidjson::Writer<rapidjson::StringBuffer>& writer) const;

        static BeamCalibration Parse(const rapidjson::Value& doc);
    };
}
}
