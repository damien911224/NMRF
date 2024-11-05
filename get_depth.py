
import numpy as np
import cv2
import os
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm

# Depth 계산 함수
def disparity_to_depth(disparity_map, focal_length_pixels, baseline_meters):
    depth_map = np.zeros_like(disparity_map, dtype=np.float32)
    non_zero_mask = disparity_map > 0
    depth_map[non_zero_mask] = (focal_length_pixels * baseline_meters) / disparity_map[non_zero_mask]
    return depth_map

import numpy as np
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.mplot3d import Axes3D

def pixel_to_3d(x, y, depth, focal_length, cx, cy):
    Z = depth
    X = (x - cx) * Z / focal_length
    Y = (y - cy) * Z / focal_length
    return np.array([X, Y, Z])

def get_3d_bbox_from_2d(bbox_2d, depth_map, focal_length, cx, cy):
    x_min, y_min, x_max, y_max = bbox_2d

    # 바운딩 박스의 네 꼭짓점에 해당하는 깊이 값 추출
    depth_topleft = depth_map[y_min, x_min]
    depth_topright = depth_map[y_min, x_max]
    depth_bottomleft = depth_map[y_max, x_min]
    depth_bottomright = depth_map[y_max, x_max]

    # 2D 좌표를 3D 좌표로 변환
    corner_3d_topleft = pixel_to_3d(x_min, y_min, depth_topleft, focal_length, cx, cy)
    corner_3d_topright = pixel_to_3d(x_max, y_min, depth_topright, focal_length, cx, cy)
    corner_3d_bottomleft = pixel_to_3d(x_min, y_max, depth_bottomleft, focal_length, cx, cy)
    corner_3d_bottomright = pixel_to_3d(x_max, y_max, depth_bottomright, focal_length, cx, cy)

    avg_depth = np.mean([depth_topleft, depth_topright, depth_bottomleft, depth_bottomright])
    center_x, center_y = (x_min + x_max) // 2, (y_min + y_max) // 2
    center_3d = pixel_to_3d(center_x, center_y, avg_depth, focal_length, cx, cy)

    bbox_3d = {
        "center": center_3d,
        "corners": [corner_3d_topleft, corner_3d_topright, corner_3d_bottomleft, corner_3d_bottomright]
    }
    return bbox_3d



if __name__ == '__main__':
    # 카메라 파라미터 설정
    focal_length_pixels = 1066.7  # Left sensor FX 값
    baseline_meters = 0.12        # baseline in meters (120mm)

    root_folder = os.path.join("/mnt/hdd0/NMRF/depth/npy")
    output_folder = os.path.join("/mnt/hdd0/NMRF/depth/outputs")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    paths = glob.glob(os.path.join(root_folder, "*.png"))

    for path in tqdm(paths):
        # disparity
        disparity_map = np.load(path)
        print(disparity_map.shape)
        print(np.max(disparity_map))

        # Depth map 변환
        depth_map = disparity_to_depth(disparity_map, focal_length_pixels, baseline_meters)

        # Depth map normalization (0-255 범위로 스케일링)
        depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_map_normalized = depth_map_normalized.astype(np.uint8)

        # Depth map 저장
        output_path = "/mnt/hdd0/NMRF/depth/1729059831.8904958.png"
        plt.imsave(output_path, depth_map_normalized, cmap='magma')