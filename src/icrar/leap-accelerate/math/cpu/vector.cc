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

#include "vector.h"

#include <Eigen/Core>

namespace icrar
{
namespace cpu
{
    void add(size_t n, const double* a, const double* b, double* c) { add<double>(n, a, b, c); }
    void add(size_t n, const float* a, const float* b, float* c) { add<float>(n, a, b, c); }
    void add(size_t n, const int* a, const int* b, int* c) { add<int>(n, a, b, c); }

    void add(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& c) { add<double>(a, b, c); }
    void add(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& c) { add<float>(a, b, c); }
    void add(const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& c) { add<int>(a, b, c); }

    void add(const Eigen::VectorXcd& a, const Eigen::VectorXcd& b, Eigen::VectorXcd& c) { c = a + b; }
    void add(const Eigen::VectorXd& a, const Eigen::VectorXd& b, Eigen::VectorXd& c) { c = a + b; }
    void add(const Eigen::VectorXf& a, const Eigen::VectorXf& b, Eigen::VectorXf& c) { c = a + b; }
    void add(const Eigen::VectorXi& a, const Eigen::VectorXi& b, Eigen::VectorXi& c) { c = a + b; }
} // namespace cpu
} // namespace icrar
