import numpy as np
from ..core.base_camera import BaseCamera

class EUCMCamera(BaseCamera):
    """
    Extended Unified Camera Model (EUCM).
    声明参数及其物理边界：
    INTRINSIC_DEFS: (参数名, 默认初值, 下界, 上界)
    DISTORTION_DEFS: (参数名, 默认初值, 下界, 上界)
    """
    INTRINSIC_DEFS = [
        ('fx', 600.0, 1e-3, np.inf),  # 焦距必须 > 0
        ('fy', 600.0, 1e-3, np.inf),
        ('cx', 640.0),                # 无界
        ('cy', 360.0)
    ]

    DISTORTION_DEFS = [
        ('alpha', 0.0, 0.0, 1.0),    # 论文规定 0 到 1 之间
        ('beta',  1.0, 1e-6, np.inf) # 论文规定必须大于 0，防止 sqrt NaN
    ]

    def project(self, points_3d: np.ndarray, extrinsics: np.ndarray) -> np.ndarray:
        """世界坐标系点 -> 像素坐标系点"""
        pts = np.asarray(points_3d, dtype=np.float64)
        if pts.ndim == 1: pts = pts.reshape(1, 3)

        R, t = extrinsics[:, :3], extrinsics[:, 3:]
        Xc = (R @ pts.T + t).T  # 世界坐标 -> 相机坐标 (N, 3)

        x, y, z = Xc[:, 0], Xc[:, 1], Xc[:, 2]

        # 声明式参数解析
        alpha, beta = self.distortion[0], self.distortion[1]

        rho2 = x**2 + y**2
        # EUCM 核心公式: $d = \sqrt{\beta(x^2+y^2) + z^2}$
        d = np.sqrt(beta * rho2 + z**2)
        denom = alpha * d + (1.0 - alpha) * z

        # 处理可能的投影异常 (denom 为 0)
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
        alpha, beta = self.distortion[0], self.distortion[1]

        mx = (pts[:, 0] - cx) / fx
        my = (pts[:, 1] - cy) / fy
        r2 = mx**2 + my**2

        # 解 EUCM 逆向方程
        mz = (1 - alpha**2 * beta * r2) / \
             (alpha * np.sqrt(1 - (2*alpha - 1)*beta*r2) + (1 - alpha))

        rays = np.stack([mx, my, mz], axis=-1)
        # 归一化为单位向量
        norms = np.linalg.norm(rays, axis=1, keepdims=True)
        return rays / norms