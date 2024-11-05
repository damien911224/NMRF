
import numpy as np
import cv2
import os
import glob
import matplotlib.pyplot as plt
import open3d as o3d

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
    else:
        # 이전 식
        # depth_map = np.zeros_like(disparity_map, dtype=np.float32)
        # non_zero_mask = disparity_map > 0
        # depth_map[non_zero_mask] = (focal_length_pixels * baseline_meters) / disparity_map[non_zero_mask]

        adjusted_cx = 0
        adjusted_cy = 0

    # TODO: 아래 조정식 검증 필요
    # Depth 맵 계산 (각 축에 맞춘 보정된 focal length와 중심 좌표 반영)
    depth_map = np.zeros_like(disparity_map, dtype=np.float32)
    non_zero_mask = disparity_map > 0
    depth_map[non_zero_mask] = (
            (focal_length_x * focal_length_y * baseline_meters) /
            ((disparity_map[non_zero_mask]) *
             np.sqrt((focal_length_x - adjusted_cx) ** 2 + (focal_length_y - adjusted_cy) ** 2))
    )

    return depth_map

def depth_to_3d(depth_map, focal_lengths_pixels, camera_positions_pixels):
    # 깊이 맵의 높이와 너비를 가져옵니다.
    height, width = depth_map.shape
    fx, fy = focal_lengths_pixels
    cx, cy = camera_positions_pixels

    # # 3D 포인트를 저장할 배열을 초기화합니다.
    # points_3d = np.zeros((height, width, 3), dtype=np.float32)
    #
    # for v in range(height):
    #     for u in range(width):
    #         Z = depth_map[v, u]  # 깊이 값 (미터 단위)
    #
    #         # 깊이 값이 0이 아닌 경우에만 3D 포인트 계산
    #         if Z > 0:
    #             X = (u - cx) * Z / fx
    #             Y = (v - cy) * Z / fy
    #             points_3d[v, u] = np.array([X, Y, Z])  # X, Y, Z는 미터 단위

    # 깊이 값이 0이 아닌 위치를 마스크로 선택
    mask = depth_map > 0

    # u, v 인덱스 생성
    v_indices, u_indices = np.indices(depth_map.shape, dtype=np.float32)

    # X, Y, Z 좌표 계산 (미터 단위)
    Z = depth_map[mask]
    X = (u_indices[mask] - cx) * Z / fx
    Y = (v_indices[mask] - cy) * Z / fy

    # 결과 배열 초기화 및 3D 포인트 할당
    points_3d = np.zeros((*depth_map.shape, 3))  # 3D 포인트 배열 초기화
    points_3d[mask] = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)

    return points_3d


if __name__ == '__main__':
    # 카메라 파라미터 설정
    focal_lengths_pixels = (1066.7, 1066.67) # Left sensor fx, fy 값
    camera_positions_pixels = (1137.39, 669.699)
    baseline_meters = 0.12 # baseline in meters (120mm)

    root_folder = os.path.join("/mnt/hdd0/NMRF/depth/npy")
    output_folder = os.path.join("/mnt/hdd0/NMRF/depth/outputs")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    paths = glob.glob(os.path.join(root_folder, "*.npy"))

    for path in tqdm(paths):
        if "crop" in path:
            continue

        # disparity
        disparity_map = np.load(path).transpose(0, 1)

        # Depth map 변환
        depth_map = disparity_to_depth(disparity_map, focal_lengths_pixels, baseline_meters)
        points_3d = depth_to_3d(depth_map, focal_lengths_pixels, camera_positions_pixels)

        print(points_3d.shape)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d)

        max_bound = np.max(points_3d, axis=0)
        min_bound = np.min(points_3d, axis=0)
        print("Point Cloud")
        print("shape", points_3d.shape)
        print("max", max_bound, flush=True)
        print("min", min_bound, flush=True)

        origin_path = glob.glob("/mnt/hdd0/stereo/*", os.path.basename(path).replace(".npy", ".png"))[0]
        print(origin_path)

        print("\nVisualize result")
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1., origin=[0, 0, 0])
        # o3d.visualization.draw_geometries([pcd, coordinate_frame])
        # o3d.io.write_point_cloud(filename, pcd)

        exit()

        # # Depth map normalization (0-255 범위로 스케일링)
        # depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        # depth_map_normalized = depth_map_normalized.astype(np.uint8)
        #
        # # Depth map 저장
        # output_path = os.path.join(output_folder, os.path.basename(path).replace(".npy", ".png"))
        # plt.imsave(output_path, depth_map_normalized, cmap='magma')