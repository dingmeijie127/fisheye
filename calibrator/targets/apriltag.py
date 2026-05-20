import cv2
import numpy as np
from ..core.base_target import BaseTargetDetector
from aprilgrid import Detector as AprilgridDetector

class AprilTagDetector(BaseTargetDetector):
    def __init__(self, tag_family, grid_shape, tag_size, tag_spacing):
        """
        tag_size: 单个二维码方格的边长 (米/毫米，决定最终相对平移向量的单位)
        """
        self.tag_family = tag_family
        self.grid_shape = grid_shape
        self.tag_size = tag_size
        self.tag_spacing = tag_spacing

        # 初始化检测器
        self.detector = AprilgridDetector(self.tag_family)

        # [关键] 禁用缩放，保证原图精度
        self.detector.large_image_threshold = 50000.0

    def _id_to_row_col(self, tag_id):
        _, cols = self.grid_shape
        r = tag_id // cols
        c = tag_id %  cols
        return r, c

    def _make_world_points_for_tags(self, tag_ids):
        tag_size = self.tag_size
        spacing = self.tag_spacing
        long_border = tag_size + spacing
        world_points = []

        for tid in tag_ids:
            r, c = self._id_to_row_col(tid)
            x0 = c * long_border
            y0 = r * long_border
            z0 = 0.0
            tl = (x0,             y0,             z0)
            tr = (x0 + tag_size,  y0,             z0)
            br = (x0 + tag_size,  y0 + tag_size,  z0)
            bl = (x0,             y0 + tag_size,  z0)
            world_points.extend([tl, tr, br, bl])

        world_points = np.asarray(world_points, dtype=np.float32)
        return world_points  # (N,3)
    def _pack_results(self, detections):
        all_points = {}
        for det in detections:
            tag_id = det.tag_id
            corners = det.corners
            world_points = self._make_world_points_for_tags([tag_id])
            all_points[tag_id] = (corners, world_points)
        return all_points

    def detect_corners(self, gray_image):
        """标准模式"""
        if len(gray_image.shape) != 2:
            gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)
        detections = self.detector.detect(gray_image)
        return self._pack_results(detections)

    def detect(self, image_path: str):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        return self.detect_corners(img)