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
 * MA  02110-1301  USA
 */

#pragma once

#include <icrar/leap-accelerate/model/cpu/calibration/Calibration.h>

#include <rapidjson/writer.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>

#include <vector>

namespace icrar
{
namespace cpu
{
    /**
     * @brief Contains a collection of calibrations
     * 
     */
    class CalibrationCollection
    {
        std::vector<cpu::Calibration> m_calibrations;
    public:
        CalibrationCollection(std::vector<cpu::Calibration>&& calibrations)
        {
            m_calibrations = std::move(calibrations);
        }

        const std::vector<cpu::Calibration>& GetCalibrations() const
        {
            return m_calibrations;
        }

        void Serialize(std::ostream& os, bool pretty = false) const
        {
            constexpr uint32_t PRECISION = 15;
            os.precision(PRECISION);
            os.setf(std::ios::fixed);

            rapidjson::StringBuffer s;
            if(pretty)
            {
                 rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(s);
                 Write(writer);
            }
            else
            {
                //rapidjson::Writer<rapidjson::StringBuffer> writer(s);
                //Write(writer);
            }

            os << s.GetString() << std::endl;
        }

        void Write(rapidjson::PrettyWriter<rapidjson::StringBuffer>& writer) const
        {
            writer.StartArray();
            for(const auto& calibration : m_calibrations)
            {
                calibration.Write(writer);
            }
            writer.EndArray();
        }
    };
}
}
