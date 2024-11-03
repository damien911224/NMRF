
import numpy as np
import cv2
import matplotlib.pyplot as plt

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
    cx, cy = 1137.39, 669.69

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
    output_path = "/mnt/hdd0/NMRF/depth/1729059831.8904958.png"
    plt.imsave(output_path, depth_map_normalized, cmap='magma')

    ###
    # 2D to 3D
    ###

    left_image_path = "/mnt/hdd0/stereo/++20241016_150903/1729059831.8904958.png"

    # 임의의 2D 바운딩 박스 좌표 및 depth map 예제
    bbox_2d = [100, 150, 200, 250]
    depth_map = np.random.uniform(0.5, 5.0, (480, 640))
    image = cv2.imread(left_image_path)

    # 3D 바운딩 박스 생성
    bbox_3d = get_3d_bbox_from_2d(bbox_2d, depth_map, focal_length, cx, cy)

    # 2D 바운딩 박스 시각화
    cv2.rectangle(image, (bbox_2d[0], bbox_2d[1]), (bbox_2d[2], bbox_2d[3]), (0, 255, 0), 2)
    cv2.imwrite("/mnt/hdd0/NMRF/depth/2d_bounding_box.png", image)

    # 3D 바운딩 박스 시각화
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 3D 바운딩 박스 코너 시각화
    for corner in bbox_3d["corners"]:
        ax.scatter(corner[0], corner[1], corner[2], color="red", s=50)

    # 중심점 시각화
    ax.scatter(bbox_3d["center"][0], bbox_3d["center"][1], bbox_3d["center"][2], color="blue", s=100, label="Center")

    # 축 레이블 설정
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    # 3D 바운딩 박스를 둘러싼 선 그리기 (테두리 연결)
    corners = bbox_3d["corners"]
    edges = [
        (0, 1), (1, 3), (3, 2), (2, 0)  # 위쪽 면 (top-left to top-right to bottom-right to bottom-left)
    ]

    for edge in edges:
        ax.plot([corners[edge[0]][0], corners[edge[1]][0]],
                [corners[edge[0]][1], corners[edge[1]][1]],
                [corners[edge[0]][2], corners[edge[1]][2]], 'r')

    ax.legend()
    plt.savefig("/mnt/hdd0/NMRF/depth/3d_bounding_box.png")
    plt.close()