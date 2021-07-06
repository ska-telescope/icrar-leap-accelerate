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
    cal.Calibrate(
        ms_path="../../testdata/mwa/1197638568-split.ms",
        autocorrelations=True,
        directions=np.array([[0.1,0.2],[0.3, 0.4],[0.5, 0.6]]),
        output_path="out.json",
        callback=callback)

    # with open('out.json') as f:
    #     output = json.loads(f)
        #print(str(output))

    assert True

if __name__ == "__main__":
    test_calibrate()