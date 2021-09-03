#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import os
import tempfile

import numpy as np
import LeapAccelerate as leap
import json

def test_readcoords():
    ms = leap.MeasurementSet("../../testdata/mwa/1197638568-split.ms")
    coords = ms.read_coords(0,1)
    assert coords.shape == (3,5253,1)
    coords = ms.read_coords(0,2)
    assert coords.shape == (3,5253,2)
    # TODO: check values

def test_readvis():
    ms = leap.MeasurementSet("../../testdata/mwa/1197638568-split.ms")
    vis = ms.read_vis(0,1)
    assert vis.shape == (1,5253,48,4)
    assert vis.flags.f_contiguous == False
    vis = vis.reshape(vis.shape[::-1], order='F')
    assert vis.shape == (4,48,5253,1)
    assert vis.flags.f_contiguous == True

    # same as MeasurementSetTests.cc
    assert vis[0][0][0][0] == 0
    assert vis[1][0][0][0] == 0
    assert vis[2][0][0][0] == 0
    assert vis[3][0][0][0] == 0
    assert vis[0][0][1][0] == -0.703454494476318 + -24.7045249938965j
    assert vis[1][0][1][0] == 5.16687202453613 + -1.57053351402283j
    assert vis[2][0][1][0] == -10.9083280563354 + 11.3552942276001j
    assert vis[3][0][1][0] == -28.7867774963379 + 20.7210712432861j
