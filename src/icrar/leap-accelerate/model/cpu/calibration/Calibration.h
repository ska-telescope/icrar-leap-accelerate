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
        std::vector<BeamCalibration> m_beamCalibrations;
        //double m_startEpoch;
        //double m_endEpoch;

    public:
        Calibration(const std::vector<cpu::BeamCalibration>& beamCalibrations)
        : m_beamCalibrations(beamCalibrations)
        {
        }

        const std::vector<BeamCalibration> GetBeamCalibrations() const
        {
            return m_beamCalibrations;
        }

        template<typename Writer>
        void Write(Writer& writer) const
        {
            writer.StartObject();
            //write.String("epoch");
            //write.Double();
            writer.String("calibration");
            for(auto& calibration : m_beamCalibrations)
            {
                calibration.Write(writer);
            }
            writer.EndObject();
        }
    };
}
}
