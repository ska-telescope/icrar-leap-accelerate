
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

#include "PhaseRotate.h"
#include "icrar/leap-accelerate/wsclean/chgcentre.h"

#include <icrar/leap-accelerate/utils.h>
#include <icrar/leap-accelerate/MetaData.h>
#include <icrar/leap-accelerate/math/Integration.h>


#include <casacore/ms/MeasurementSets/MeasurementSet.h>
#include <casacore/measures/Measures/MDirection.h>
#include <casacore/ms/MeasurementSets/MSAntenna.h>

#include <casacore/casa/Quanta/MVDirection.h>
#include <casacore/casa/Quanta/MVuvw.h>

#include <casacore/casa/Arrays/Matrix.h>
#include <casacore/casa/Arrays/Vector.h>

#include <boost/math/constants/constants.hpp>

#include <istream>
#include <iostream>
#include <iterator>
#include <string>
#include <filesystem>
#include <optional>
#include <exception>
#include <memory>

using namespace casacore;
using Arrayd = Array<double>;
using Matrixd = Matrix<double>;
using Matrixi = Matrix<int>;

namespace icrar
{
    // HandleRemoteMS
    void PhaseRotate(casacore::MeasurementSet& ms, std::vector<casacore::MDirection> directions)
    {
        MetaData metadata = {};
        metadata.init = True;
        metadata.channels;// = channels;
        metadata.num_pols;
        metadata.rows;

        metadata.freq_start_hz;// = freq_start_hz;
    }

    void RotateVisibilities(Integration& integration, MetaData& metadata, const MVDirection& direction)
    {
        using namespace std::complex_literals;
        auto& data = integration.data;
        auto& uvw = integration.uvw;
        auto parameters = integration.parameters;

        if(metadata.init)
        {
            //metadata['nbaseline']=metadata['stations']*(metadata['stations']-1)/2
            SetDD(metadata, direction);
            SetWv(metadata);
            // Zero a vector for averaging in time and freq
            metadata.avg_data = Matrix<DComplex>(integration.baselines, metadata.num_pols);
            metadata.init = false;
        }
        CalcUVW(uvw, metadata);

        // loop over baselines
        for(int baseline = 0; baseline < integration.baselines; ++baseline)
        {
            // For baseline
            //Needs checking/completing ...
            const double pi = boost::math::constants::pi<double>();
            double shiftFactor = -2 * pi * uvw[baseline].get()[2] - metadata.oldUVW[baseline].get()[2]; // check these are correct
            shiftFactor = shiftFactor + 2 * pi * (metadata.phase_centre_ra_rad*metadata.oldUVW[baseline].get()[0]);
            shiftFactor = shiftFactor -2 * pi * (direction.get()[0] * uvw[baseline].get()[0] - direction.get()[1] * uvw[baseline].get()[1]);

            if(baseline % 1000 == 1)
            {
                std::cout << "ShiftFactor for baseline " << baseline << " is " << shiftFactor << std::endl;
            }

            // Loop over channels
            for(int channel = 0; channel < metadata.channels; channel++)
            {
                double shiftRad = shiftFactor / metadata.channel_wavelength[channel];
                double rs = sin(shiftRad);
                double rc = cos(shiftRad);
                std::complex<double> v = data[channel][baseline];

                data[channel][baseline] = v * exp(1i * shiftRad);
                if(data[channel][baseline].real() == NAN
                || data[channel][baseline].imag() == NAN)
                {
                    metadata.avg_data(IPosition(baseline)) += data[channel][baseline];
                }
            }
        }
    }

    /**
     * Form Phase Matrix
     * Given the antenna lists from MS and (optionally) RefAnt & Map:
     * If non-negative RefAnt is provided it only forms the matrix for baselines with that antenna.
     * If True Map is provided it returns the index map for the matrix (only useful if RefAnt set).
     *
     * This function generates and returns the linear matrix for the phase calibration (only)
     */
    std::pair<Matrixd, Matrixi> PhaseMatrixFuntion(const Arrayd& a1, const Arrayd& a2, int refAnt=-1, bool map=false)
    {
        //TODO array equal, see invert.py
        //int nAnt = 1 + (a1 == a2) ? 1 : 0;
        int nAnt = 2;
        if(refAnt >= nAnt - 1)
        {
            throw std::invalid_argument("RefAnt out of bounds");
        }

        Matrixd A = Matrixd(a1.size() + 1, ArrayMax<double>(a1));
        for(auto v : A)
        {
            v = 0;
        }

        Matrixi I = Matrixi(a1.size() + 1, a1.size() + 1);
        for(auto v : I)
        {
            v = 1;
        }

        int k = 0;

        for(int n = 0; n < a1.size(); n++)
        {
            if(a1(IPosition(n)) != a2(IPosition(n)))
            {
                if((refAnt < 0) | ((refAnt >= 0) & ((a1(IPosition(n))==refAnt) | (a2(IPosition(n)) == refAnt))))
                {
                    A(IPosition(k, a1(IPosition(n)))) = 1;
                    A(IPosition(k, a2(IPosition(n)))) = -1;
                    I(IPosition(k)) = n;
                    k++;
                }
            }
        }
        if(refAnt < 0)
        {
            refAnt = 0;
            A(IPosition(k,refAnt)) = 1;
            k++;
            
            throw std::runtime_error("matrix slice needs implementation");
            //A = A[:k];
            //I = I[:k];
        }

        return std::make_pair(A, I);
    }
    
    void CalcUVW(std::vector<MVuvw>& uvw, MetaData& metadata)
    {
        metadata.oldUVW = uvw;
        for(int n = 0; n < uvw.size(); n++)
        {
            throw std::runtime_error("not implemented");
            //uvw(IPosition(n)) = uvw(IPosition(n));
        }
    }

    void SetDD(MetaData& metadata, const MVDirection& direction)
    {
        metadata.dlm_ra = direction.get()[0] - metadata.phase_centre_ra_rad;
        metadata.dlm_dec = direction.get()[1] - metadata.phase_centre_dec_rad;

        metadata.dd(IPosition(0,0)) = cos(metadata.dlm_ra) * cos(metadata.dlm_dec);
        metadata.dd(IPosition(0,1)) = -sin(metadata.dlm_ra);
        metadata.dd(IPosition(0,2)) = cos(metadata.dlm_ra) * sin(metadata.dlm_dec);
        
        metadata.dd(IPosition(1,0)) = sin(metadata.dlm_ra) * cos(metadata.dlm_dec);
        metadata.dd(IPosition(1,1)) = cos(metadata.dlm_ra);
        metadata.dd(IPosition(1,2)) = sin(metadata.dlm_ra) * sin(metadata.dlm_dec);

        metadata.dd(IPosition(2,0)) = -sin(metadata.dlm_dec);
        metadata.dd(IPosition(2,1)) = 0;
        metadata.dd(IPosition(2,2)) = cos(metadata.dlm_dec);
    }

    /**
     * @brief Set the wavelength from meta data
     * 
     * @param metadata 
     */
    void SetWv(MetaData& metadata)
    {
        double speed_of_light = 299792458.0;
        metadata.channel_wavelength = range(
            metadata.freq_start_hz,
            metadata.freq_inc_hz,
            metadata.freq_start_hz + metadata.freq_inc_hz * metadata.channels);
        for(double& v : metadata.channel_wavelength)
        {
            v = speed_of_light / v;
        }
    }
}
