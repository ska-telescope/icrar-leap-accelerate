
import numpy as np
import LeapAccelerate as leap

def callback():
    print("python callback")

cal = leap.LeapCalibrator("cpu")
cal.Calibrate(
    callback,
    "../../testdata/mwa/1197638568-split.ms",
    True,
    np.array([[0.1,0.2],[0.3, 0.4],[0.5, 0.6]]))