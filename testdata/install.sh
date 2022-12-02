#!/bin/bash
#
# ICRAR - International Centre for Radio Astronomy Research
# (c) UWA - The University of Western Australia, 2019
# Copyright by UWA (in the framework of the ICRAR)
# All rights reserved
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#


fail() {
	echo -e "$@" 1>&2
	exit 1
}

download_and_extract() {
    directory=`dirname "$2"`
    [ -d "$directory" ] || mkdir "$directory"

    # if extracted folder exists then assume cache is correct
    output="${2%.*.*}.ms"
    if [ ! -d $output ]; then
        echo "Downloading and extracting $2"
        wget -nv "$1" -O "$2" || fail "failed to download $2 from $1"
        tar -C "$directory" -xf "$2" || (fail "failed to extract $output" && rm -rf $output)
    else
        echo "$output already exists"
    fi
}

download_and_extract "https://cloudstor.aarnet.edu.au/plus/s/Eb65Nqy66hUE2tO/download" mwa/1197638568-split.tar.gz
download_and_extract "https://cloudstor.aarnet.edu.au/plus/s/KO5jutEkg3UffmU/download" askap/askap-SS-1100.tar.gz
download_and_extract "https://cloudstor.aarnet.edu.au/plus/s/MCJ1RzuRzGiqW9t/download" aa3/aa3-SS-300.tar.gz
download_and_extract "https://cloudstor.aarnet.edu.au/plus/s/qtIV1HqXfKsQVAu/download" ska/SKA_LOW_SIM_short_EoR0_ionosphere_off_GLEAM.0001.tar.gz
