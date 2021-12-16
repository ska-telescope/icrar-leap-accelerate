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
 * MA  02110-1301  USA
 */

#pragma once

#include <cstddef>

namespace icrar
{
    /**
     * @brief Gets the total amount system virtual memory. This includes
     * the system's dynamic RAM plus swap space.
     */
    size_t GetTotalSystemVirtualMemory();

    /**
     * @brief Gets the total amount of used system virtual memory.
     */
    size_t GetTotalUsedSystemVirtualMemory();

    /**
     * @brief Gets the currently available/free virtual system memory.
     */
    size_t GetTotalAvailableSystemVirtualMemory();

    /**
     * @brief Gets the total physical cuda memory on the current cuda device.
     */
    size_t GetTotalCudaPhysicalMemory();

    /**
     * @brief Gets the currently available/free physical cuda memory of the current cuda device.
     * This excludes the memory used by the current process.
     */
    size_t GetAvailableCudaPhysicalMemory();
} // namespace icrar
