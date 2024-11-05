
import numpy as np
import cv2
import os
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm

# Depth 계산 함수
def disparity_to_depth(disparity_map, focal_lengths_pixels, baseline_meters,
                       image_size=None, crop_bbox=None):
    # disparity_map: shape = (H, W)
    # focal_lengths_pixels: 튜플 형태 (fx, fy), pixel 단위의 x와 y축에 대한 focal length
    # baseline_meters: meter 단위의 baseline length
    # image_size: 튜플 형태 (w, h), 크랍 이미지를 이용했을 때, focal length를 조정 하기 위해 필요
    # crop_bbox: 튜플 형태 (x, y, w, h) 크랍 bbox의 중심 좌표와 박스의 사이즈.

    focal_length_x, focal_length_y = focal_lengths_pixels

    if crop_bbox is not None:
        assert image_size is not None

        focal_length_x = focal_length_x * (crop_bbox[2] / image_size[0])
        focal_length_y = focal_length_y * (crop_bbox[3] / image_size[1])

        # 원본 이미지의 중심 좌표
        cx_original = image_size[0] / 2
        cy_original = image_size[1] / 2

        # 크롭된 이미지의 새로운 중심 좌표 계산
        adjusted_cx = cx_original - crop_bbox[0]
        adjusted_cy = cy_original - crop_bbox[1]

        # TODO: 아래 조정식 검증 필요
        # Depth 맵 계산 (각 축에 맞춘 보정된 focal length와 중심 좌표 반영)
        depth_map = np.zeros_like(disparity_map, dtype=np.float32)
        non_zero_mask = disparity_map > 0
        depth_map[non_zero_mask] = (
                (focal_length_x * focal_length_y * baseline) /
                ((disparity_map[non_zero_mask]) *
                 np.sqrt((focal_length_x - adjusted_cx) ** 2 + (focal_length_y - adjusted_cy) ** 2))
        )
    else:
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
    focal_lengths_pixels = (1066.7, 1066.7) # Left sensor fx, fy 값
    baseline_meters = 0.12 # baseline in meters (120mm)

    root_folder = os.path.join("/mnt/hdd0/NMRF/depth/npy")
    output_folder = os.path.join("/mnt/hdd0/NMRF/depth/outputs")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    paths = glob.glob(os.path.join(root_folder, "*.npy"))

    for path in tqdm(paths):
        # disparity
        disparity_map = np.load(path).transpose(0, 1)

        # Depth map 변환
        depth_map = disparity_to_depth(disparity_map, focal_lengths_pixels, baseline_meters)

        # Depth map normalization (0-255 범위로 스케일링)
        depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_map_normalized = depth_map_normalized.astype(np.uint8)

        # Depth map 저장
        output_path = "/mnt/hdd0/NMRF/depth/1729059831.8904958.png"
        plt.imsave(output_path, depth_map_normalized, cmap='magma')