import numpy as np
from ..core.base_camera import BaseCamera

class DoubleSphereCamera(BaseCamera):
    """
    Double Sphere Camera Model.
    Reference: Usenko et al., "The Double Sphere Camera Model", 2018.
    """
    INTRINSIC_DEFS = [
        ('fx', 600.0, 1e-3, np.inf),  # 焦距必须 > 0
        ('fy', 600.0, 1e-3, np.inf),
        ('cx', 640.0),                # 无界
        ('cy', 360.0)
    ]

    DISTORTION_DEFS = [
        ('xi', 0.0, -1.0, 1.0),       # 第一个球体的偏移量
        ('alpha', 0.5, 0.0, 1.0)      # 两个球体之间的插值因子
    ]

    def project(self, points_3d: np.ndarray, extrinsics: np.ndarray) -> np.ndarray:
        """世界坐标系点 -> 像素坐标系点"""
        pts = np.asarray(points_3d, dtype=np.float64)
        if pts.ndim == 1: pts = pts.reshape(1, 3)

        R, t = extrinsics[:, :3], extrinsics[:, 3:]
        Xc = (R @ pts.T + t).T  # 世界坐标 -> 相机坐标 (N, 3)

        x, y, z = Xc[:, 0], Xc[:, 1], Xc[:, 2]

        xi, alpha = self.distortion[0], self.distortion[1]

        # Double Sphere 核心投影公式
        r2 = x**2 + y**2
        d1 = np.sqrt(r2 + z**2)
        d2 = np.sqrt(r2 + (z + xi * d1)**2)

        denom = alpha * d2 + (1.0 - alpha) * (z + xi * d1)

        # 处理可能的投影异常
        valid = denom > 1e-8
        u = np.full_like(x, np.nan)
        v = np.full_like(y, np.nan)

        fx, fy, cx, cy = self.intrinsics
        u[valid] = fx * (x[valid] / denom[valid]) + cx
        v[valid] = fy * (y[valid] / denom[valid]) + cy

        return np.stack([u, v], axis=-1)

    def unproject(self, points_2d: np.ndarray) -> np.ndarray:
        """像素坐标 (u,v) -> 相机坐标系下的 3D 单位射线"""
        pts = np.asarray(points_2d, dtype=np.float64)
        fx, fy, cx, cy = self.intrinsics
        xi, alpha = self.distortion[0], self.distortion[1]

        mx = (pts[:, 0] - cx) / fx
        my = (pts[:, 1] - cy) / fy
        r2 = mx**2 + my**2

        # 解 Double Sphere 逆向方程
        mz = (1.0 - alpha**2 * r2) / \
             (alpha * np.sqrt(1.0 - (2.0*alpha - 1.0)*r2) + (1.0 - alpha))

        mz2 = mz**2
        factor = (mz * xi + np.sqrt(mz2 + (1.0 - xi**2) * r2)) / (mz2 + r2)

        rx = factor * mx
        ry = factor * my
        rz = factor * mz - xi

        rays = np.stack([rx, ry, rz], axis=-1)
        # 归一化为单位向量
        norms = np.linalg.norm(rays, axis=1, keepdims=True)
        return rays / norms