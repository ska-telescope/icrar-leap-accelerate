
# ICRAR - International Centre for Radio Astronomy Research
# (c) UWA - The University of Western Australia
# Copyright by UWA(in the framework of the ICRAR)
# All rights reserved

# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.

# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the GNU
# Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston,
# MA 02111 - 1307  USA

import collections
from typing import Deque, Optional

import ska_sdp_dal as rpc
from sdp_dal_schemas import PROCEDURES
from cbf_sdp import icd

import leap_accelerate

class PlasmaPayload(icd.Payload):
    def __init__(self):
        self.tm = None
        self.baseline_count = None
        self.channel_count = None
        self.timestamp_count = None
        self.timestamp_fraction = None
        self.visibilities = None


class LeapProcessor(rpc.Processor):
    """ An SKA-SDP-DAL Processor that runs leap-accelerate as a service"""
    
    def __init__(self, *args, **kwargs):
        super(LeapProcessor, self).__init__(
            [PROCEDURES['read_payload']], *args, **kwargs
        )
        self.bytes_received = 0
        self.payloads: Deque[PlasmaPayload] = collections.deque()
    
    def read_payload(self, time_index, times, intervals, exposures, baselines,
                     spectral_window, channels, antennas, field, polarizations,
                     uvw, vis, output):
        pass
