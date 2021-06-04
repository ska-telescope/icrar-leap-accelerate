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

#include <icrar/leap-accelerate/exception/exception.h>

#include <casacore/ms/MeasurementSets.h>
#include <casacore/ms/MeasurementSets/MeasurementSet.h>
#include <casacore/ms/MeasurementSets/MSColumns.h>

#include <casacore/measures/Measures/MDirection.h>
#include <casacore/casa/Arrays/Matrix.h>
#include <casacore/casa/Arrays.h>

#include <boost/numeric/conversion/cast.hpp>

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include <icrar/leap-accelerate/math/cpu/eigen_extensions.h>

#include <iterator>
#include <sstream>
#include <string>
#include <exception>
#include <memory>
#include <vector>

namespace icrar
{
    //See https://github.com/OxfordSKA/OSKAR/blob/f018c03bb34c16dcf8fb985b46b3e9dc1cf0812c/oskar/ms/src/oskar_ms_read.cpp
    template<typename T>
    void ms_read_coords(
        const casacore::MeasurementSet& ms,
        uint32_t start_row,
        uint32_t num_baselines,
        T* uu,
        T* vv,
        T* ww)
    {
        auto rms = casacore::MeasurementSet(ms);
        auto msmc = std::make_unique<casacore::MSMainColumns>(rms);

        uint32_t total_rows = boost::numeric_cast<uint32_t>(ms.nrow());
        if(start_row >= total_rows)
        {
            std::stringstream ss;
            ss << "ms out of range " << start_row << " >= " << total_rows; 
            throw icrar::exception(ss.str(), __FILE__, __LINE__);
        }

        // reduce selection if selecting out of range
        if(start_row + num_baselines > total_rows)
        {
            num_baselines = total_rows - start_row;
        }

        // Read the coordinate data and copy it into the supplied arrays.
        casacore::Slice slice(start_row, num_baselines, 1);
        casacore::Array<double> columnRange = msmc->uvw().getColumnRange(slice);
        casacore::Matrix<double> matrix;
        matrix.reference(columnRange);
        for (uint32_t i = 0; i < num_baselines; ++i)
        {
            uu[i] = matrix(0, i); // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
            vv[i] = matrix(1, i); // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
            ww[i] = matrix(2, i); // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        }
    }

    template<typename T>
    Eigen::Matrix<T, -1, 3> ms_read_coords1(const casacore::MeasurementSet& ms, uint32_t start_row, uint32_t num_rows)
    {
        Eigen::MatrixX3d matrix = Eigen::MatrixX3d::Zero(num_rows, 3);
        icrar::ms_read_coords(
            ms,
            start_row,
            num_rows,
            matrix.col(0).data(),
            matrix.col(1).data(),
            matrix.col(2).data());
        return matrix;
    }

    template<typename T>
    Eigen::Matrix<T, -1, 3> ms_read_coords2(const casacore::MeasurementSet& ms, uint32_t start_row, uint32_t num_rows)
    {
        Eigen::Matrix3d sd;
        auto rms = casacore::MeasurementSet(ms);
        auto msmc = std::make_unique<casacore::MSMainColumns>(rms);

        uint32_t total_rows = boost::numeric_cast<uint32_t>(ms.nrow());
        if(start_row >= total_rows)
        {
            std::stringstream ss;
            ss << "ms out of range " << start_row << " >= " << total_rows; 
            throw icrar::exception(ss.str(), __FILE__, __LINE__);
        }

        // reduce selection if selecting out of range
        if(start_row + num_rows > total_rows)
        {
            std::stringstream ss;
            ss << "ms out of range " << start_row + num_rows << " >= " << total_rows; 
            throw icrar::exception(ss.str(), __FILE__, __LINE__);
        }

        casacore::Slice slice(start_row, num_rows, 1);
        return Eigen::Map<Eigen::Matrix<double, -1, 3, Eigen::RowMajorBit>>(
            msmc->uvw().getColumnRange(slice).data(), num_rows, 3);
    }

    //See https://github.com/OxfordSKA/OSKAR/blob/f018c03bb34c16dcf8fb985b46b3e9dc1cf0812c/oskar/ms/src/oskar_ms_read.cpp
    template<typename T>
    void ms_read_vis(
        const casacore::MeasurementSet& ms,
        unsigned int start_baseline,
        unsigned int start_channel,
        unsigned int num_channels,
        unsigned int num_baselines,
        unsigned int num_pols,
        const char* column,
        T* vis)
    {
        if(!ms.tableDesc().isColumn(column))
        {
            throw icrar::exception("ms column not found", __FILE__, __LINE__);
        }

        if(strcmp(column, "DATA")
        && strcmp(column, "CORRECTED_DATA")
        && strcmp(column, "MODEL_DATA"))
        {
            throw icrar::exception("ms column not found", __FILE__, __LINE__);
        }

        uint32_t total_rows = boost::numeric_cast<uint32_t>(ms.nrow());
        if (start_baseline >= total_rows)
        {
            std::stringstream ss;
            ss << "ms out of range " << start_baseline << " >= " << total_rows; 
            throw icrar::exception(ss.str(), __FILE__, __LINE__);
        }

        // clamp num_baselines
        if (start_baseline + num_baselines > total_rows)
        {
            std::stringstream ss;
            ss << "row selection [" << start_baseline << "," << start_baseline + num_baselines << "] exceeds total range [" << 0 << "," << total_rows << "]";
            throw icrar::exception(ss.str(), __FILE__, __LINE__);
        }

        // Create the slicers for the column.
        casacore::IPosition start1(1, start_baseline);
        casacore::IPosition length1(1, num_baselines);
        casacore::Slicer row_range(start1, length1);
        casacore::IPosition start2(2, 0, start_channel);
        casacore::IPosition length2(2, num_pols, num_channels);
        casacore::Slicer array_section(start2, length2);

        // Read the data.
        casacore::ArrayColumn<std::complex<float>> ac(ms, column);
        casacore::Array<std::complex<float>> column_range = ac.getColumnRange(row_range, array_section);

        // Copy the visibility data into the supplied array,
        // swapping baseline and channel dimensions.
        auto in = reinterpret_cast<const float*>(column_range.data()); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
        for (unsigned int c = 0; c < num_channels; ++c)
        {
            for (unsigned int b = 0; b < num_baselines; ++b)
            {
                for (unsigned int p = 0; p < num_pols; ++p)
                {
                    unsigned int i = (num_pols * (b * num_channels + c) + p) << 1;
                    unsigned int j = (num_pols * (c * num_baselines + b) + p) << 1;
                    vis[j]     = static_cast<T>(in[i]); // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
                    vis[j + 1] = static_cast<T>(in[i + 1]); // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
                }
            }
        }
    }

    template<typename T>
    auto ms_read_vis1(
        const casacore::MeasurementSet& ms,
        unsigned int start_timestep,
        unsigned int interval_timesteps,
        unsigned int num_timesteps,
        unsigned int num_baselines,
        unsigned int num_channels,
        unsigned int num_pols,
        const char* column)
    {
        const unsigned int start_row = start_timestep * num_baselines;
        const unsigned int rows = interval_timesteps * num_baselines;
        const unsigned int total_rows = num_timesteps * num_baselines;
        //uint32_t total_rows = boost::numeric_cast<uint32_t>(ms.nrow());
        const unsigned int out_pols = num_pols; //std::min(num_pols, 2u);

        if(!ms.tableDesc().isColumn(column))
        {
            throw icrar::exception("ms column not found", __FILE__, __LINE__);
        }

        if(strcmp(column, "DATA")
        && strcmp(column, "CORRECTED_DATA")
        && strcmp(column, "MODEL_DATA"))
        {
            throw icrar::exception("expected a data column", __FILE__, __LINE__);
        }

        if (start_row >= total_rows)
        {
            std::stringstream ss;
            ss << "ms out of range " << start_row << " >= " << total_rows; 
            throw icrar::exception(ss.str(), __FILE__, __LINE__);
        }

        // clamp num_baselines
        if (start_row + rows > total_rows)
        {
            std::stringstream ss;
            ss << "row selection [" << start_row << "," << start_row + rows << "] exceeds total range [" << 0 << "," << total_rows << "]";
            throw icrar::exception(ss.str(), __FILE__, __LINE__);
        }

        // Create slicers for table DATA
        // Slicer for table rows: array[baselines,timesteps]
        casacore::IPosition start1(1, start_row);
        casacore::IPosition length1(1, rows);
        casacore::Slicer row_range(start1, length1);

        // Slicer for row entries: matrix[polarizations,channels]
        casacore::IPosition start2(2, 0, 0);
        casacore::IPosition length2(2, num_pols, num_channels);
        casacore::Slicer array_section(start2, length2);

        // Read the data.
        casacore::ArrayColumn<std::complex<float>> ac(ms, column);
        casacore::Array<std::complex<float>> column_range = ac.getColumnRange(row_range, array_section);

        Eigen::TensorMap<Eigen::Tensor<std::complex<float>, 3>> view(column_range.data(), num_pols, num_channels, num_baselines * interval_timesteps);

        //TODO: Converting ICD format from [pol, channels, baselines*timesteps] to [pol, baselines*timesteps, channels]
        const Eigen::array<Eigen::DenseIndex, 3> shuffle = { 0, 2, 1 };
        //const Eigen::array<Eigen::DenseIndex, 3> strides = { std::max(1u, num_pols-1), 1u, 1u }; // select XX and YY polarizations
        const Eigen::array<Eigen::DenseIndex, 3> strides = { 1u, 1u, 1u }; // select XX and YY polarizations
        Eigen::Tensor<T, 3> output = view.stride(strides).shuffle(shuffle).cast<T>();
        return output;
    }

    template<typename T>
    auto ms_read_vis2(
        const casacore::MeasurementSet& ms,
        unsigned int start_timestep,
        unsigned int interval_timesteps,
        unsigned int num_timesteps,
        unsigned int num_baselines,
        unsigned int num_channels,
        unsigned int num_pols,
        const char* columnName)
    {
        const unsigned int start_row = start_timestep * num_baselines;
        const unsigned int rows = interval_timesteps * num_baselines;
        const unsigned int total_rows = num_timesteps * num_baselines;
        //uint32_t total_rows = boost::numeric_cast<uint32_t>(ms.nrow());
        const unsigned int out_pols = std::min(num_pols, 2u);

        if(!ms.tableDesc().isColumn(columnName))
        {
            throw icrar::exception("ms column not found", __FILE__, __LINE__);
        }

        if(strcmp(columnName, "DATA") && strcmp(columnName, "CORRECTED_DATA") && strcmp(columnName, "MODEL_DATA"))
        {
            throw icrar::exception("expected a data column", __FILE__, __LINE__);
        }

        if (start_row >= total_rows)
        {
            std::stringstream ss;
            ss << "ms out of range " << start_row << " >= " << total_rows; 
            throw icrar::exception(ss.str(), __FILE__, __LINE__);
        }

        // clamp num_baselines
        if (start_row + rows > total_rows)
        {
            std::stringstream ss;
            ss << "row selection [" << start_row << "," << start_row + rows << "] exceeds total range [" << 0 << "," << total_rows << "]";
            throw icrar::exception(ss.str(), __FILE__, __LINE__);
        }

        // Create slicers for table DATA
        // Slicer for table rows: array[baselines,timesteps]
        casacore::IPosition start1(1, start_row);
        casacore::IPosition length1(1, rows);
        casacore::Slicer row_range(start1, length1);

        // Slicer for row entries: matrix[polarizations,channels]
        casacore::IPosition start2(2, 0, 0);
        casacore::IPosition length2(2, out_pols, num_channels);
        casacore::IPosition stride2(2, std::max(1u, num_pols-1), 1u); // select XX and YY polarizations
        casacore::Slicer array_section(start2, length2, stride2);

        // Read the data
        casacore::ArrayColumn<std::complex<float>> ac(ms, columnName);
        casacore::Array<std::complex<float>> column_range = ac.getColumnRange(row_range, array_section);

        Eigen::TensorMap<Eigen::Tensor<std::complex<float>, 4>> view(column_range.data(), out_pols, num_channels, num_baselines, num_timesteps);
        Eigen::Tensor<T, 4> output = view.cast<T>();
        return output;
    }
} // namespace icrar