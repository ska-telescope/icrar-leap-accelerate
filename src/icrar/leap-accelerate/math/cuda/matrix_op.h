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
#ifdef CUDA_ENABLED

#include <cublas_v2.h>

namespace icrar
{
/// cuda
namespace cuda
{
    enum class MatrixOp
    {
        normal = CUBLAS_OP_N, //< No matrix operation
        transpose = CUBLAS_OP_T, //< diagonal reflection
        hermitian = CUBLAS_OP_C, //< conjugate and diagonal reflection
        conjugate = CUBLAS_OP_T | CUBLAS_OP_C //< element-wise conjugates
    };

    /**
     * @brief Converts a matrix operation to a cublas operation
     * 
     * @param op 
     * @return cublasOperation_t 
     */
    cublasOperation_t ToCublasOp(MatrixOp op);

} // namespace cuda
} // namespace icrar

#endif // CUDA_ENABLED
