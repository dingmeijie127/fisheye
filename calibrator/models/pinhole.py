import numpy as np
import cv2
from ..core.base_camera import BaseCamera

class PinholeCamera(BaseCamera):
    """
    标准针孔相机模型 (Brown-Conrady 畸变)
    内参: [fx, fy, cx, cy, s] -> 增加 s 以支持非正交像素
    畸变: [k1, k2, p1, p2, k3]
    """
    INTRINSIC_DEFS = [
        ('fx', 500.0, 1e-3, np.inf),
        ('fy', 500.0, 1e-3, np.inf),
        ('cx', 320.0),
        ('cy', 240.0),
        ('s',  0.0) # Skew 倾斜因子
    ]

    DISTORTION_DEFS = [
        ('k1', 0.0), ('k2', 0.0),
        ('p1', 0.0), ('p2', 0.0), ('k3', 0.0)
    ]

    def project(self, points_3d, extrinsics):
        R, t = extrinsics[:, :3], extrinsics[:, 3:]
        Xc = (R @ points_3d.T + t).T

        z = Xc[:, 2]
        valid = z > 1e-6
        x = np.full_like(z, np.nan, dtype=np.float64)
        y = np.full_like(z, np.nan, dtype=np.float64)
        x[valid] = Xc[valid, 0] / z[valid]
        y[valid] = Xc[valid, 1] / z[valid]

        # 归一化平面
        x = Xc[:, 0] / Xc[:, 2]
        y = Xc[:, 1] / Xc[:, 2]

        r2 = x**2 + y**2
        r4 = r2**2
        r6 = r2 * r4

        # 提取参数
        fx, fy, cx, cy, s = self.intrinsics
        k1, k2, p1, p2, k3 = self.distortion

        # 径向畸变 + 切向畸变
        radial = (1 + k1*r2 + k2*r4 + k3*r6)
        x_dist = x * radial + (2*p1*x*y + p2*(r2 + 2*x**2))
        y_dist = y * radial + (p1*(r2 + 2*y**2) + 2*p2*x*y)

        # 像素转换 (含 Skew)
        u = fx * x_dist + s * y_dist + cx
        v = fy * y_dist + cy

        return np.stack([u, v], axis=-1)

    def unproject(self, points_2d):
        """
        针孔反投影 (近似处理，不考虑畸变时的逆运算)
        精确逆运算通常需要迭代法，这里返回理想针孔的射线。
        """
        fx, fy, cx, cy, s = self.intrinsics
        mx = (points_2d[:, 0] - cx - (s/fy)*(points_2d[:, 1]-cy)) / fx
        my = (points_2d[:, 1] - cy) / fy

        rays = np.stack([mx, my, np.ones_like(mx)], axis=-1)
        return rays / np.linalg.norm(rays, axis=1, keepdims=True)