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

#include <icrar/leap-accelerate/model/cpu/calibration/BeamCalibration.h>
#include <icrar/leap-accelerate/math/math_conversion.h>

#include <rapidjson/writer.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>

#include <vector>

namespace icrar
{
namespace cpu
{
    /**
     * @brief Contains a single calibration solution.
     * 
     */
    class Calibration
    {
        double m_startEpoch;
        double m_endEpoch;
        std::vector<BeamCalibration> m_beamCalibrations;

    public:
        /**
         * @brief Creates an empty calibration
         * 
         * @param startEpoch 
         * @param endEpoch 
         */
        Calibration(double startEpoch, double endEpoch)
        : m_startEpoch(startEpoch)
        , m_endEpoch(endEpoch)
        {}


        Calibration(double startEpoch, double endEpoch, std::vector<cpu::BeamCalibration>&& beamCalibrations)
        : m_startEpoch(startEpoch)
        , m_endEpoch(endEpoch)
        , m_beamCalibrations(std::move(beamCalibrations))
        {
        }

        double GetStartEpoch() const { return m_startEpoch; }
        
        double GetEndEpoch() const { return m_endEpoch; }

        const std::vector<BeamCalibration>& GetBeamCalibrations() const
        {
            return m_beamCalibrations;
        }

        std::vector<BeamCalibration>& GetBeamCalibrations()
        {
            return m_beamCalibrations;
        }

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
            writer.StartObject();
                writer.String("epoch"); writer.StartObject();
                    writer.String("start"); writer.Double(m_startEpoch);
                    writer.String("end"); writer.Double(m_endEpoch);
                writer.EndObject(); 
                writer.String("calibration"); writer.StartArray();
                    for(auto& calibration : m_beamCalibrations)
                    {
                        calibration.Write(writer);
                    }
                writer.EndArray();
            writer.EndObject();
        }
    };
}
}
