#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import os
import tempfile

# import ska_sdp_dal as rpc
import numpy as np
import LeapAccelerate as leap
import json

def test_calibrate_minimal():
    cal = leap.LeapCalibrator("cpu")
    output_file = tempfile.mktemp(suffix='.json', prefix='output_')
    cal.calibrate(
        ms_path="../../testdata/mwa/1197638568-split.ms",
        directions=np.array([[0.1,0.2],[0.3, 0.4],[0.5,0.6]], np.float64),
        output_path=output_file)

    with open(output_file) as f:
        output = json.load(f)
        assert len(output) == 1

def test_calibrate():
    cal = leap.LeapCalibrator(leap.ComputeImplementation.cpu)
    output_file = tempfile.mktemp(suffix='.json', prefix='output_')
    cal.calibrate(
        ms_path="../../testdata/mwa/1197638568-split.ms",
        directions=np.array([[0.1,0.2],[0.3, 0.4],[0.5, 0.6]]),
        solution_interval=slice(0,None,1),
        output_path=output_file)
    with open(output_file) as f:
        output = json.load(f)
        assert len(output) == 14

    cal.calibrate(
        ms_path="../../testdata/mwa/1197638568-split.ms",
        directions=np.array([[0.1,0.2],[0.3, 0.4],[0.5, 0.6]]),
        solution_interval=slice(0,None,None),
        output_path=output_file)
    with open(output_file) as f:
        output = json.load(f)
        assert len(output) == 1

def test_calibrate_callback():
    cal = leap.LeapCalibrator("cpu")
    
    output = list()
    def append_output(calibration):
        output.append(calibration)

    cal.calibrate(
        ms_path="../../testdata/mwa/1197638568-split.ms",
        directions=np.array([[0.1,0.2],[0.3, 0.4],[0.5, 0.6]]),
        solution_interval=slice(0,None,1),
        callback=append_output)

    assert len(output) == 14

@pytest.mark.skip(reason="segfault on accessing future object")
def test_calibrate_async():
    cal = leap.LeapCalibrator("cpu")
    
    output = list()
    def append_output(calibration):
        output.append(calibration)

    future = cal.calibrate_async(
        ms_path="../../testdata/mwa/1197638568-split.ms",
        directions=np.array([[0.1,0.2],[0.3, 0.4],[0.5, 0.6]]),
        solution_interval=slice(0,None,1),
        callback=append_output)
    future.wait()

    assert len(output) == 14

def test_plasma_calibration():
    cal = leap.LeapCalibrator("cpu")
    # cal.plasma_calibrate()

# def test_calibrate_config():
#     cal = leap.LeapCalibrator("cpu")
#     cal.calibrate(config="../../testdata/mwa.json")