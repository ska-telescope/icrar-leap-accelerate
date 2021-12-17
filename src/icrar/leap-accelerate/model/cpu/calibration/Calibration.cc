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

#include "Calibration.h"

namespace icrar
{
namespace cpu
{
    Calibration::Calibration(double startEpoch, double endEpoch)
    : m_startEpoch(startEpoch)
    , m_endEpoch(endEpoch)
    {}


    Calibration::Calibration(double startEpoch, double endEpoch, std::vector<cpu::BeamCalibration>&& beamCalibrations)
    : m_startEpoch(startEpoch)
    , m_endEpoch(endEpoch)
    , m_beamCalibrations(std::move(beamCalibrations))
    {
    }

    double Calibration::GetStartEpoch() const { return m_startEpoch; }
    
    double Calibration::GetEndEpoch() const { return m_endEpoch; }

    bool Calibration::IsApprox(const Calibration& calibration, double threshold)
    {
        bool equal = m_startEpoch == calibration.m_startEpoch
        && m_endEpoch == calibration.m_endEpoch
        && GetBeamCalibrations().size() == calibration.GetBeamCalibrations().size(); 
        if(equal)
        {
            for(size_t i = 0; i < GetBeamCalibrations().size();  i++)
            {
                equal &= GetBeamCalibrations()[i].IsApprox(calibration.GetBeamCalibrations()[i], threshold);
                if(!equal)
                {
                    break;
                }
            }
        }
        return equal;
    }

    const std::vector<BeamCalibration>& Calibration::GetBeamCalibrations() const
    {
        return m_beamCalibrations;
    }

    std::vector<BeamCalibration>& Calibration::GetBeamCalibrations()
    {
        return m_beamCalibrations;
    }

    void Calibration::Serialize(std::ostream& os, bool pretty) const
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
            rapidjson::Writer<rapidjson::StringBuffer> writer(s);
            Write(writer);
        }

        os << s.GetString() << std::endl;
    }

    Calibration Calibration::Parse(const std::string& json)
    {
        rapidjson::Document doc;
        doc.Parse(json.c_str());
        return Parse(doc);
    }

    Calibration Calibration::Parse(const rapidjson::Value& doc)
    {
        if(!doc.IsObject())
        {
            throw icrar::exception("expected an object", __FILE__, __LINE__);
        }

        if(!doc.HasMember("epoch"))
        {
            throw icrar::exception("expected an epoch key", __FILE__, __LINE__);
        }
        const rapidjson::Value& epoch = doc["epoch"];
        if(!epoch.IsObject())
        {
            throw icrar::exception("expected an epoch object", __FILE__, __LINE__);
        }
        double startEpoch = epoch["start"].GetDouble();
        double endEpoch = epoch["end"].GetDouble();

        if(!doc.HasMember("calibration"))
        {
            throw icrar::exception("expected a calibration key", __FILE__, __LINE__);
        }
        const rapidjson::Value& calibration = doc["calibration"];
        if(!calibration.IsArray())
        {
            throw icrar::exception("expected a calibration array", __FILE__, __LINE__);
        }
        std::vector<cpu::BeamCalibration> beamCalibrations;
        for(auto it = calibration.Begin(); it != calibration.End(); it++)
        {
            beamCalibrations.push_back(BeamCalibration::Parse(*it));
        }
        return Calibration(startEpoch, endEpoch, std::move(beamCalibrations));
    }
} // namespace cpu
} // namespace icrar