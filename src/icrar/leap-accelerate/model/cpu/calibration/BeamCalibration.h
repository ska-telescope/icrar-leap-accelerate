/**
 * ICRAR - International Centre for Radio Astronomy Research
 * (c) UWA - The University of Western Australia
 * Copyright by UWA(in the framework of the ICRAR)
 * All rights reserved
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#pragma once

#include <icrar/leap-accelerate/common/SphericalDirection.h>
#include <icrar/leap-accelerate/exception/exception.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#ifndef __NVCC__
#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>
#endif // __NVCC__

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
        Eigen::MatrixXd m_calibration; //TODO(calgray) refactor to Eigen::VectorXd

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

        template<typename Writer>
        void Write(Writer& writer) const
        {
            assert(m_calibration.cols() == 1);

            writer.StartObject();
            writer.String("direction");
            writer.StartArray();

            for(auto& v : m_direction)
            {
                writer.Double(v);
            }
            writer.EndArray();

            writer.String("beamCalibration");
            writer.StartArray();
            for(int i = 0; i < m_calibration.rows(); ++i)
            {
                writer.Double(m_calibration(i));
            }
            writer.EndArray();
            writer.EndObject();
        }

#ifndef __NVCC__
        static BeamCalibration Parse(const rapidjson::Value& doc);
    };
#endif // __NVCC__
}
}
