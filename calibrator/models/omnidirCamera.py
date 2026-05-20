import numpy as np
from ..core.base_camera import BaseCamera

class OmnidirCamera(BaseCamera):
    """
    全向相机模型 (Mei & Rives)
    内参: [fx, fy, cx, cy, s]
    畸变: [xi, k1, k2, p1, p2]
    """
    INTRINSIC_DEFS = [
        ('fx', 400.0, 1e-3, np.inf),
        ('fy', 400.0, 1e-3, np.inf),
        ('cx', 640.0),
        ('cy', 360.0),
        ('s',  0.0)
    ]

    DISTORTION_DEFS = [
        ('xi', 1.0, 1e-6, 5.0), # 核心：镜面参数，通常接近1
        ('k1', 0.0), ('k2', 0.0),('k3',0.0),
        ('p1', 0.0), ('p2', 0.0)
    ]

    def project(self, points_3d, extrinsics):
        R, t = extrinsics[:, :3], extrinsics[:, 3:]
        Xc = (R @ points_3d.T + t).T

        # 1. 投影到单位球面
        norm = np.linalg.norm(Xc, axis=1, keepdims=True)
        Xs = Xc / norm

        # 2. 变换到归一化平面 (基于 xi 参数)
        xi = self.distortion[0]
        # x_u = x / (z + xi), y_u = y / (z + xi)
        denom = Xs[:, 2] + xi
        xu = Xs[:, 0] / denom
        yu = Xs[:, 1] / denom

        # 3. 应用常规畸变 (k1, k2, p1, p2)
        r2 = xu**2 + yu**2
        r4 = r2**2
        k1, k2, k3,p1, p2 = self.distortion[1:6]

        radial = (1 + k1*r2 + k2*r2**2+k3*r4**2)
        xd = xu * radial + (2*p1*xu*yu + p2*(r2 + 2*xu**2))
        yd = yu * radial + (p1*(r2 + 2*yu**2) + 2*p2*xu*yu)

        # 4. 像素转换
        fx, fy, cx, cy, s = self.intrinsics
        u = fx * xd + s * yd + cx
        v = fy * yd + cy

        return np.stack([u, v], axis=-1)

    def unproject(self, points_2d):
        """全向模型的反投影：计算球面射线"""
        # 简化的反投影逻辑（用于初始化或校正）
        fx, fy, cx, cy, s = self.intrinsics
        xi = self.distortion[0]

        xd = (points_2d[:, 0] - cx - (s/fy)*(points_2d[:, 1]-cy)) / fx
        yd = (points_2d[:, 1] - cy) / fy

        r2 = xd**2 + yd**2
        # 计算 z_s 坐标
        tmp = (1 - xi**2 * r2) / (1 + r2)
        zs = (xi + np.sqrt(np.maximum(0, 1 + (1-xi**2)*r2))) / (1 + r2) - xi

        # 进而推导出单位球面上的 x, y
        # ... 这里略过复杂的几何反推，通常用于立体校正中的射线查找 ...
        return np.stack([xd, yd, np.ones_like(xd)], axis=-1) # 简化示意