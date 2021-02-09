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
#include <string>

namespace icrar
{
    /**
     * @brief Specifies how results are streamed out
     */
    enum class StreamOutType
    {
        COLLECTION, // Results are written to a collection in a single file
        SINGLE_FILE, // Results are continously rewritten to a single file as computed
        MUTLIPLE_FILES // Results are continously written to multiple files as computed
    };

    /**
     * @return true if value was converted succesfully, false otherwise
     */
    bool TryParseStreamOutType(const std::string& value, StreamOutType& out);

    /**
     * @brief True if solutions should be written to IO as soon as they are computed.
     */
    bool IsImmediateMode(StreamOutType streamOutType);
}
