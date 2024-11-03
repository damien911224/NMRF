
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Depth 계산 함수
def disparity_to_depth(disparity_map, focal_length_pixels, baseline_meters):
    depth_map = np.zeros_like(disparity_map, dtype=np.float32)
    non_zero_mask = disparity_map > 0
    depth_map[non_zero_mask] = (focal_length_pixels * baseline_meters) / disparity_map[non_zero_mask]
    return depth_map

if __name__ == '__main__':
    # 카메라 파라미터 설정
    focal_length_pixels = 1066.7  # Left sensor FX 값
    baseline_meters = 0.12        # baseline in meters (120mm)

    # disparity
    np_path = "/mnt/hdd0/NMRF/kitti/1729059831.8904958.npy"
    disparity_map = np.load(np_path)

    # Depth map 변환
    depth_map = disparity_to_depth(disparity_map, focal_length_pixels, baseline_meters)

    print(depth_map)
    print(np.min(depth_map))
    print(np.max(depth_map))

    # Depth map normalization (0-255 범위로 스케일링)
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_map_normalized = depth_map_normalized.astype(np.uint8)

    # Depth map 저장
    output_path = '/mnt/hdd0/NMRF/depth/1729059831.8904958.png'
    plt.imsave(output_path, depth_map_normalized, cmap='magma')