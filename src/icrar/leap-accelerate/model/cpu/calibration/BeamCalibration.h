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

#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include <rapidjson/writer.h>
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
        BeamCalibration(
            SphericalDirection direction,
            Eigen::MatrixXd calibration)
            : m_direction(std::move(direction))
            , m_calibration(std::move(calibration))
        {
        }

        /**
         * @brief Gets the calibration direction
         * 
         * @return const SphericalDirection 
         */
        const SphericalDirection GetDirection() const { return m_direction; }

        /**
         * @brief Get the calibration Vector for the antenna array in the specified direction
         * 
         * @return const Eigen::MatrixXd 
         */
        const Eigen::MatrixXd& GetCalibration() const { return m_calibration; }

        void Serialize(std::ostream& os) const
        {
            constexpr uint32_t PRECISION = 15;
            os.precision(PRECISION);
            os.setf(std::ios::fixed);

            rapidjson::StringBuffer s;

#ifdef PRETTY_WRITER
            rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(s);
#else
            rapidjson::Writer<rapidjson::StringBuffer> writer(s);
#endif
            Write(writer);
            os << s.GetString() << std::endl;
        }

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

            writer.String("calibration");
            writer.StartArray();
            for(int i = 0; i < m_calibration.rows(); ++i)
            {
                writer.Double(m_calibration(i,0));
            }
            writer.EndArray();

            writer.EndObject();
        }
    };
}
}
