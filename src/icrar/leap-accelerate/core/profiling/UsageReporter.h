/**
 * ICRAR - International Centre for Radio Astronomy Research
 * (c) UWA - The University of Western Australia
 * Copyright by UWA(in the framework of the ICRAR)
 * All rights reserved
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#pragma once

namespace icrar
{
namespace profiling
{

/// Reports high-level, process-wide resource usage values on destruction
class UsageReporter
{
public:
    UsageReporter() = default;
    UsageReporter(UsageReporter&&) = default;
    UsageReporter(const UsageReporter&) = default;
    UsageReporter& operator=(const UsageReporter&) = default;
    UsageReporter& operator=(UsageReporter&&) = default;
    ~UsageReporter() noexcept;
};

} // namespace profiling
} // namespace icrar