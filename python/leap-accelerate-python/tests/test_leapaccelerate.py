#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import os

import numpy as np
import LeapAccelerate as leap
import json

def callback():
    print("got callback")

def test_calibrate():
    cal = leap.LeapCalibrator("cpu")
    cal.calibrate(
        ms_path="../../testdata/mwa/1197638568-split.ms",
        autocorrelations=True,
        directions=np.array([[0.1,0.2],[0.3, 0.4],[0.5, 0.6]]),
        output_path="calibration.json")

    with open('calibration.json') as f:
        output = json.load(f)
        assert len(output) == 1
