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

#include "BeamCalibration.h"

namespace icrar
{
namespace cpu
{
    BeamCalibration::BeamCalibration(
        SphericalDirection direction,
        Eigen::MatrixXd calibration)
        : m_direction(std::move(direction))
        , m_calibration(std::move(calibration))
    {
    }

    BeamCalibration::BeamCalibration(const std::pair<SphericalDirection, Eigen::MatrixXd>& beamCalibration)
    {
        std::tie(m_direction, m_calibration) = beamCalibration;
    }

    bool BeamCalibration::IsApprox(const BeamCalibration& beamCalibration, double threshold)
    {
        bool equal = m_direction == beamCalibration.m_direction
        && m_calibration.rows() == beamCalibration.m_calibration.rows()
        && m_calibration.cols() == beamCalibration.m_calibration.cols()
        && m_calibration.isApprox(beamCalibration.m_calibration, threshold);
        if(!equal)
        {
            std::cout << "beamcal not equal" << std::endl;
            std::cout << std::setprecision(15);
            std::cout << beamCalibration.m_direction << std::endl;
            std::cout << std::endl;
            std::cout << beamCalibration.m_calibration << std::endl;
            std::cout << std::endl;
            std::cout << m_calibration << std::endl;
        }
        return equal;
    }

    const SphericalDirection& BeamCalibration::GetDirection() const
    {
        return m_direction;
    }

    const Eigen::MatrixXd& BeamCalibration::GetPhaseCalibration() const
    {
        return m_calibration;
    }

    void BeamCalibration::Serialize(std::ostream& os, bool pretty) const
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

    BeamCalibration BeamCalibration::Parse(const rapidjson::Value& doc)
    {
        if(!doc.IsObject())
        {
            throw icrar::exception("expected a beam calibration", __FILE__, __LINE__);
        }
        const auto& direction = doc["direction"];
        const auto sphericalDirection = SphericalDirection(direction[0].GetDouble(), direction[1].GetDouble());

        const auto& calibrationJson = doc["beamCalibration"];
        if(!calibrationJson.IsArray())
        {
            throw icrar::exception("expected an array", __FILE__, __LINE__);
        }
        Eigen::VectorXd calibrationVector(calibrationJson.Size());
        std::transform(calibrationJson.Begin(), calibrationJson.End(), calibrationVector.begin(),
        [](const rapidjson::Value& v){ return v.GetDouble(); });
        return BeamCalibration(sphericalDirection, calibrationVector); 
    }
} // namespace cpu
} // namespace icrar
