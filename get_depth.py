
import numpy as np

np_path = "/mnt/hdd0/NMRF/kitti/1729059831.8904958.npy"
disparity = np.load(np_path)

B = 120 # mm
f = 2.1 # 2.1 or 4 mm

# depth = Baseline * focal-lens / disparity
depth = B * f / disparity

print(disparity)

print()

print(np.max(depth))
