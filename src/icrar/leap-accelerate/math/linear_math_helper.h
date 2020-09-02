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

#include <icrar/leap-accelerate/common/MVuvw.h>

#include <casacore/casa/Arrays/Matrix.h>
#include <casacore/casa/Arrays/Vector.h>
#include <casacore/casa/Quanta/MVuvw.h>

#include <icrar/leap-accelerate/common/eigen_3_3_beta_1_2_support.h>
#include <eigen3/Eigen/Core>
#include <vector>

namespace icrar
{
    template<typename T>
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> ConvertMatrix(const casacore::Matrix<T>& value)
    {
        auto shape = value.shape();
        auto output = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(shape[0], shape[1]);
        memcpy(output.data(), value.data(), shape[0] * shape[1] * sizeof(T));
        return output;
    }

    template<typename T, int R, int C>
    Eigen::Matrix<T, R, C> ConvertMatrix(const casacore::Matrix<T>& value)
    {
        auto shape = value.shape();
        if(shape[0] != R || shape[1] != C)
        {
            throw std::invalid_argument("matrix shape does not match template");
        }

        auto output = Eigen::Matrix<T, R, C>();
        memcpy(output.data(), value.data(), R * C * sizeof(T));
        return output;
    }

    template<typename T, int R, int C>
    casacore::Matrix<T> ConvertMatrix(const Eigen::Matrix<T, R, C>& value)
    {
        return casacore::Matrix<T>(casacore::IPosition(2, R, C), value.data());
    }

    template<typename T>
    casacore::Matrix<T> ConvertMatrix(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& value)
    {
        return casacore::Matrix<T>(casacore::IPosition(2, (int)value.rows(), (int)value.cols()), value.data());
    }

    template<typename T>
    Eigen::Matrix<T, Eigen::Dynamic, 1> ConvertVector(casacore::Array<T> value)
    {
        auto output = Eigen::Matrix<T, Eigen::Dynamic, 1>(value.size());
        for(int i = 0; i < value.size(); ++i)
        {
            output(i) = value(casacore::IPosition(1, i)); // TODO consider using std::copy or memcpy
        }
        return output;
    }

    template<typename T>
    casacore::Array<T> ConvertVector(Eigen::Matrix<T, Eigen::Dynamic, 1> value)
    {
        return casacore::Array<T>(casacore::IPosition(value.rows()), value.data());
    }

    icrar::MVuvw ToUVW(const casacore::MVPosition& value);
    std::vector<icrar::MVuvw> ToUVW(const std::vector<casacore::MVuvw>& value);

    casacore::MVuvw ToCasaUVW(icrar::MVuvw value);
    std::vector<casacore::MVuvw> ToCasaUVW(const std::vector<icrar::MVuvw>& value);
}