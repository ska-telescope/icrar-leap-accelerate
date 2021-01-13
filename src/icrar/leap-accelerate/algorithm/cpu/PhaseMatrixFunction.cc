
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

#include "PhaseMatrixFunction.h"

#include <icrar/leap-accelerate/exception/exception.h>

#include <sstream>
#include <set>

namespace icrar
{
namespace cpu
{
    std::pair<Eigen::MatrixXd, Eigen::VectorXi> PhaseMatrixFunction(
        const Eigen::VectorXi& a1,
        const Eigen::VectorXi& a2,
        const Eigen::Matrix<bool, Eigen::Dynamic, 1>& fg,
        int refAnt)
    {
        if(a1.size() != a2.size() && a1.size() != fg.size())
        {
            throw invalid_argument_exception("a1 and a2 must be equal size", "a", __FILE__, __LINE__);
        }

        auto unique = std::set<std::int32_t>(a1.begin(), a1.end());
        unique.insert(a2.begin(), a2.end());
        int nAnt = unique.size();

        if(refAnt >= nAnt)
        {
            std::stringstream ss;
            ss << "refAnt " << refAnt << " is out of bounds";
            throw invalid_argument_exception(ss.str(), "refAnt", __FILE__, __LINE__);
        }
        if(fg[refAnt])
        {
            std::stringstream ss;
            ss << "refAnt " << refAnt << " is flagged";
            throw invalid_argument_exception(ss.str(), "refAnt", __FILE__, __LINE__);
        }

        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(a1.size() + 1, std::max(a1.maxCoeff(), a2.maxCoeff()) + 1);
        Eigen::VectorXi I = Eigen::VectorXi(a1.size());
        I.setConstant(-1);

        int k = 0;

        for(int n = 0; n < a1.size(); n++)
        {
            if(a1(n) != a2(n))
            {
                // skip entry if data not flagged
                if(!fg(n) && ((refAnt < 0) || ((refAnt >= 0) && ((a1(n) == refAnt) || (a2(n) == refAnt)))))
                {
                    A(k, a1(n)) = 1;
                    A(k, a2(n)) = -1;
                    I(k) = n;
                    k++;
                }
            }
        }
        if(refAnt < 0)
        {
            refAnt = 0;
        }

        A(k, refAnt) = 1;
        k++;
        
        A.conservativeResize(k, Eigen::NoChange);
        I.conservativeResize(k-1);

        return std::make_pair(std::move(A), std::move(I));
    }
} // namespace cpu
} // namespace icrar
