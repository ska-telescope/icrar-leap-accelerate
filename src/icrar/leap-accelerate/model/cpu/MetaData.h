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

#if CUDA_ENABLED
#include <cuda_runtime.h>
#else
#ifndef __host__
#define __host__
#endif // __host__
#ifndef __device__
#define __device__
#endif // __device__
#endif // CUDA_ENABLED

#include <icrar/leap-accelerate/model/cpu/MVuvw.h>
#include <icrar/leap-accelerate/common/SphericalDirection.h>
#include <icrar/leap-accelerate/common/constants.h>

#include <icrar/leap-accelerate/cuda/device_vector.h>
#include <icrar/leap-accelerate/cuda/device_matrix.h>

#include <casacore/measures/Measures/MDirection.h>
#include <casacore/casa/Quanta/MVuvw.h>

#include <casacore/casa/Arrays/Matrix.h>
#include <casacore/casa/Arrays/Vector.h>

#include <Eigen/Core>

#include <boost/optional.hpp>

#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <complex>

namespace icrar
{
    class MeasurementSet;
    namespace cuda
    {
        class DeviceMetaData;
        class ConstantBuffer;
    } // namespace cuda
} // namespace icrar

namespace icrar
{
namespace cpu
{
    /**
     * @brief Container of fixed sized variables that do not change during calibration
     */
    struct Constants
    {
        uint32_t nbaselines; //the total number station pairs (excluding self cycles) 

        uint32_t referenceAntenna;

        uint32_t channels; // The number of channels of the current observation
        uint32_t num_pols; // The number of polarizations used by the current observation
        uint32_t stations; // The number of stations used by the current observation
        uint32_t rows;

        double freq_start_hz; // The frequency of the first channel, in Hz
        double freq_inc_hz; // The frequency incrmeent between channels, in Hz

        double phase_centre_ra_rad;
        double phase_centre_dec_rad;
        double dlm_ra;
        double dlm_dec;

        __host__ __device__ double GetChannelWavelength(int i) const
        {
            return constants::speed_of_light / (freq_start_hz + i * freq_inc_hz);
        }

        bool operator==(const Constants& rhs) const;
    };

    /**
     * @brief container of phaserotation constants and variables for calibrating a single beam.
     * Can be mutated to calibrate for multiple directions.
     */
    class MetaData
    {
        MetaData() = default;

    protected:
        Constants m_constants;
        double m_minimumBaselineThreshold;
        bool m_useCache;

        Eigen::MatrixXd m_A;
        Eigen::VectorXi m_I; // The flagged indexes of A
        Eigen::MatrixXd m_Ad; // The pseudo-inverse of m_A

        Eigen::MatrixXd m_A1;
        Eigen::VectorXi m_I1;
        Eigen::MatrixXd m_Ad1;

        std::vector<icrar::MVuvw> m_UVW;
    
        SphericalDirection m_direction; // calibration direction, late initialized
        Eigen::Matrix3d m_dd; // direction dependant matrix, late initialized
        Eigen::MatrixXcd m_avgData; // matrix of size (baselines, polarizations), late initialized
    
    public:
        
        /**
         * @brief Construct a new MetaData object. SetUVW() and SetDirection() must be called after construction
         * 
         * @param ms 
         * @param minimumBaselineThreshold
         * @param useCache
         */
        MetaData(const icrar::MeasurementSet& ms, boost::optional<unsigned int> refAnt = boost::none, double minimumBaselineThreshold = 0.0, bool computeInverse = true, bool useCache = true);


        /**
         * @brief Construct a new MetaData object. SetDirection() must be called after construction
         * 
         * @param ms measurement set to read observations from
         * @param uvws uvw coordinates of stations
         * @param refAnt the reference antenna index, default is the last index
         * @param minimumBaselineThreshold baseline lengths less that the minimum in meters are flagged
         * @param useCache whether to load Ad matrix from cache
         */
        MetaData(const icrar::MeasurementSet& ms, const std::vector<icrar::MVuvw>& uvws, boost::optional<unsigned int> refAnt = boost::none, double minimumBaselineThreshold = 0.0, bool computeInverse = true, bool useCache = true);

        /**
         * @brief Construct a new MetaData object.
         * 
         * @param ms measurement set to read observations from
         * @param direction the direction of the beam to calibrate for
         * @param uvws uvw coordinates of stations
         * @param refAnt the reference antenna index, default is the last index
         * @param minimumBaselineThreshold baseline lengths less that the minimum in meters are flagged
         * @param useCache whether to load Ad matrix from cache
         */
        MetaData(const icrar::MeasurementSet& ms, const SphericalDirection& direction, const std::vector<icrar::MVuvw>& uvws, boost::optional<unsigned int> refAnt = boost::none, double minimumBaselineThreshold = 0.0, bool computeInverse = true, bool useCache = true);

        const Constants& GetConstants() const;

        /**
         * @brief Matrix of baseline pairs of shape [baselines, stations] 
         */
        const Eigen::MatrixXd& GetA() const;

        /**
         * @brief Vector of indexes of the stations that are not flagged, shape [stations]
         */
        const Eigen::VectorXi& GetI() const;

        /**
         * @brief The pseudoinverse of A with shape [stations, baselines]
         */
        const Eigen::MatrixXd& GetAd() const;

        /**
         * @brief Matrix of baselines using the reference antenna of shape [stations+1, stations]
         * the last row represents the reference antenna
         */
        const Eigen::MatrixXd& GetA1() const;
        const Eigen::VectorXi& GetI1() const;
        const Eigen::MatrixXd& GetAd1() const;

        const std::vector<icrar::MVuvw>& GetUVW() const { return m_UVW; }

        const SphericalDirection& GetDirection() const { return m_direction; }
        const Eigen::Matrix3d& GetDD() const { return m_dd; }
        void SetDirection(const SphericalDirection& direction);

        void SetUVW(const std::vector<icrar::MVuvw>& uvws);

        /**
         * @brief Computes the A and A1 inverse matrices 
         * 
         */
        void ComputeInverse();

        /**
         * @brief Updates the rotated UVW vector using the DD matrix
         * @pre DD is set, UVW is set
         */
        void CalcUVW();

        /**
         * @brief Utility method to generate a direction matrix using the
         * configured zenith direction
         * 
         * @param direction 
         * @return Eigen::Matrix3d 
         */
        Eigen::Matrix3d GenerateDDMatrix(const SphericalDirection& direction) const;

        const Eigen::MatrixXcd& GetAvgData() const { return m_avgData; }
        Eigen::MatrixXcd& GetAvgData() { return m_avgData; }

        bool operator==(const MetaData& rhs) const;
        bool operator!=(const MetaData& rhs) const { return !(*this == rhs); }

        friend class icrar::cuda::DeviceMetaData;
        friend class icrar::cuda::ConstantBuffer;
    };
} // namespace cpu
} // namespace icrar
