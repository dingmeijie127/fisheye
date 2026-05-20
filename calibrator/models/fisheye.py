import numpy as np
from ..core.base_camera import BaseCamera

class FisheyeCamera(BaseCamera):
    """
    Kannala-Brandt Fisheye Camera Model (OpenCV fisheye).
    """
    INTRINSIC_DEFS = [
        ('fx', 600.0, 1e-3, np.inf),
        ('fy', 600.0, 1e-3, np.inf),
        ('cx', 640.0),
        ('cy', 360.0)
    ]

    DISTORTION_DEFS = [
        ('k1', 0.0),  # 无界，通常极值在 -1 到 1 左右
        ('k2', 0.0),
        ('k3', 0.0),
        ('k4', 0.0)
    ]

    def project(self, points_3d: np.ndarray, extrinsics: np.ndarray) -> np.ndarray:
        """世界坐标系点 -> 像素坐标系点"""
        pts = np.asarray(points_3d, dtype=np.float64)
        if pts.ndim == 1: pts = pts.reshape(1, 3)

        R, t = extrinsics[:, :3], extrinsics[:, 3:]
        Xc = (R @ pts.T + t).T

        x, y, z = Xc[:, 0], Xc[:, 1], Xc[:, 2]
        k1, k2, k3, k4 = self.distortion

        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(r, z)

        theta2 = theta**2
        theta4 = theta2**2
        theta6 = theta4 * theta2
        theta8 = theta4**2

        # 畸变角计算
        theta_d = theta * (1.0 + k1*theta2 + k2*theta4 + k3*theta6 + k4*theta8)

        # 避免 r 接近 0 导致的除以零 (极小值时 scale 趋近于 1/z)
        scale = np.zeros_like(r)
        nonzero = r > 1e-8
        scale[nonzero] = theta_d[nonzero] / r[nonzero]
        scale[~nonzero] = 1.0 / (z[~nonzero] + 1e-8)

        xd = x * scale
        yd = y * scale

        # 这里不采用 denominator 过滤，因为鱼眼可以投影到 z<=0 的区域
        # 但需过滤无效的极小 z（防完全处于相机光心）
        valid = (z > -1e-8) | (r > 1e-8)
        u = np.full_like(x, np.nan)
        v = np.full_like(y, np.nan)

        fx, fy, cx, cy = self.intrinsics
        u[valid] = fx * xd[valid] + cx
        v[valid] = fy * yd[valid] + cy

        return np.stack([u, v], axis=-1)

    def unproject(self, points_2d: np.ndarray) -> np.ndarray:
        """像素坐标 (u,v) -> 相机坐标系下的 3D 单位射线 (需使用牛顿迭代求解 theta)"""
        pts = np.asarray(points_2d, dtype=np.float64)
        fx, fy, cx, cy = self.intrinsics
        k1, k2, k3, k4 = self.distortion

        mx = (pts[:, 0] - cx) / fx
        my = (pts[:, 1] - cy) / fy
        theta_d = np.sqrt(mx**2 + my**2)

        # 牛顿法求逆解 theta
        theta = theta_d.copy()  # 初始猜测
        for _ in range(10):     # 10次迭代通常能达到极高精度
            theta2 = theta**2
            theta4 = theta2**2
            theta6 = theta4 * theta2
            theta8 = theta4**2

            # 函数及其一阶导数
            f = theta * (1.0 + k1*theta2 + k2*theta4 + k3*theta6 + k4*theta8) - theta_d
            df = 1.0 + 3.0*k1*theta2 + 5.0*k2*theta4 + 7.0*k3*theta6 + 9.0*k4*theta8

            theta -= f / df

        r_ray = np.sin(theta)
        z_ray = np.cos(theta)

        valid = theta_d > 1e-8
        rx = np.zeros_like(theta_d)
        ry = np.zeros_like(theta_d)

        rx[valid] = r_ray[valid] * mx[valid] / theta_d[valid]
        ry[valid] = r_ray[valid] * my[valid] / theta_d[valid]

        # theta_d 为 0 的中心点，射线方向直指 Z 轴正向
        rx[~valid], ry[~valid], z_ray[~valid] = 0.0, 0.0, 1.0

        rays = np.stack([rx, ry, z_ray], axis=-1)
        norms = np.linalg.norm(rays, axis=1, keepdims=True)
        return rays / norms