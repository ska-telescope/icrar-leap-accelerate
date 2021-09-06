#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import os
import tempfile

import numpy as np
import LeapAccelerate as leap
import json

def test_tensor():
    array = np.array(leap.Tensor3d(1,2,3), order = 'F', copy = False)
    array[:] = 0
    assert array.shape == (1,2,3)

    mat = leap.create_matrix()
    assert mat.shape == (5,5)
    assert mat.flags.f_contiguous == True
    assert mat.flags.owndata == False

def test_readcoords():
    ms = leap.MeasurementSet("../../testdata/mwa/1197638568-split.ms")
    coords = np.array(ms.read_coords(0,1), order = 'F', copy = False)
    assert coords.shape == (3,5253,1)
    assert coords.flags.f_contiguous == True
    assert coords.flags.owndata == False
    coords = np.array(ms.read_coords(0,2), order = 'F', copy = False)
    assert coords.shape == (3,5253,2)
    assert coords.flags.f_contiguous == True
    assert coords.flags.owndata == False
    # TODO: check values

def test_readcoords_numpy():
    ms = leap.MeasurementSet("../../testdata/mwa/1197638568-split.ms")
    coords = ms.read_coords(0,1).numpy_view
    assert coords.shape == (3,5253,1)
    assert coords.flags.f_contiguous == True
    assert coords.flags.owndata == True
    coords = ms.read_coords(0,2).numpy_view
    assert coords.shape == (3,5253,2)
    assert coords.flags.f_contiguous == True
    assert coords.flags.owndata == True
    # TODO: check values

def test_readvis():
    ms = leap.MeasurementSet("../../testdata/mwa/1197638568-split.ms")
    vis = np.array(ms.read_vis(0,1), order = 'F', copy = False)
    assert vis.shape == (4,48,5253,1)
    assert vis.flags.f_contiguous == True
    assert vis.flags.owndata == False

    # same as MeasurementSetTests.cc
    assert vis[0][0][0][0] == 0
    assert vis[1][0][0][0] == 0
    assert vis[2][0][0][0] == 0
    assert vis[3][0][0][0] == 0
    assert vis[0][0][1][0] == pytest.approx(-0.703454494476318 + -24.7045249938965j)
    assert vis[1][0][1][0] == pytest.approx(5.16687202453613 + -1.57053351402283j)
    assert vis[2][0][1][0] == pytest.approx(-10.9083280563354 + 11.3552942276001j)
    assert vis[3][0][1][0] == pytest.approx(-28.7867774963379 + 20.7210712432861j)

def test_readvis_numpy():
    ms = leap.MeasurementSet("../../testdata/mwa/1197638568-split.ms")
    vis = ms.read_vis(0,1).numpy_view
    assert vis.shape == (4,48,5253,1)
    assert vis.flags.f_contiguous == True
    assert vis.flags.owndata == True

    # same as MeasurementSetTests.cc
    assert vis[0][0][0][0] == 0
    assert vis[1][0][0][0] == 0
    assert vis[2][0][0][0] == 0
    assert vis[3][0][0][0] == 0
    assert vis[0][0][1][0] == pytest.approx(-0.703454494476318 + -24.7045249938965j)
    assert vis[1][0][1][0] == pytest.approx(5.16687202453613 + -1.57053351402283j)
    assert vis[2][0][1][0] == pytest.approx(-10.9083280563354 + 11.3552942276001j)
    assert vis[3][0][1][0] == pytest.approx(-28.7867774963379 + 20.7210712432861j)
