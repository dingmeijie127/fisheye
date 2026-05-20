import numpy as np
import cv2
import json

class StereoRectifier:
    def __init__(self, cam_l, cam_r, R_rel, t_rel, orig_size):
        """
        初始化极线校正器
        :param cam_l, cam_r: 继承自 BaseCamera 的左右相机实例 (如 DS, omnidir 等)
        :param R_rel, t_rel: 双目标定求出的相对位姿 (Right to Left 或 Left to Right 依据你的约定)
        :param orig_size: (width, height) 原始图像尺寸
        """
        self.cam_l = cam_l
        self.cam_r = cam_r
        self.R_rel = R_rel
        self.t_rel = t_rel
        self.w, self.h = orig_size

        # 1. 利用 OpenCV 获取纯几何旋转的校正矩阵 R1 (左) 和 R2 (右)
        # 传入单位矩阵以骗过其内置的针孔模型限制
        self.R1, self.R2, _, _, _, _, _ = cv2.stereoRectify(
            cameraMatrix1=np.eye(3), distCoeffs1=np.zeros(4),
            cameraMatrix2=np.eye(3), distCoeffs2=np.zeros(4),
            imageSize=(self.w, self.h),
            R=self.R_rel, T=self.t_rel,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
        )


    def generate_remap_arrays(self, target_size=None, proj_type='cylindrical', **kwargs):
        """
        生成极线校正所需的 mapx, mapy
        :param target_size: (width, height) 输出图像的分辨率，默认与原图相同
        :param proj_type: 'pinhole' (传统针孔) 或 'cylindrical' (柱面投影，无黑边大视野)
        :param kwargs:
               - 若为 pinhole: 可传 scale_factor (默认0.8，越小视野越大但越容易撕裂)
               - 若为 cylindrical: 可传 fov_h_deg (默认180.0，水平展开视场角)
        """
        if target_size is None:
            target_size = (self.w, self.h)
        tw, th = target_size

        # 生成目标图像的像素网格
        u, v = np.meshgrid(np.arange(tw), np.arange(th))
        u_flat = u.flatten()
        v_flat = v.flatten()

        # 2. 根据不同的投影模型，生成虚拟相机坐标系下的 3D 射线 (X, Y, Z)
        if proj_type == 'pinhole':
            scale = kwargs.get('scale_factor', 0.8)
            # 启发式焦距: 基准焦距为宽度的 1/3，乘上缩放因子
            f_new = (tw / 3.0) * scale
            cx_new, cy_new = tw / 2.0, th / 2.0

            X = (u_flat - cx_new) / f_new
            Y = (v_flat - cy_new) / f_new
            Z = np.ones_like(X)

        elif proj_type == 'cylindrical':
            # 柱面投影：将 u 映射为水平角度，将 v 映射为圆柱高度
            fov_h_rad = np.radians(kwargs.get('fov_h_deg', 180.0))

            # 计算虚拟柱面的焦距 (保证像素等比)
            f_cyl = tw / fov_h_rad

            # 水平角度 theta: 从 -FOV/2 到 +FOV/2
            theta = (u_flat / tw - 0.5) * fov_h_rad
            # 垂直高度 Y
            Y = (v_flat - th / 2.0) / f_cyl

            # 柱面坐标转 3D 直角坐标 (X: 右, Y: 下, Z: 前)
            X = np.sin(theta)
            Z = np.cos(theta)
            # Y 保持不变

        else:
            raise ValueError("不支持的 proj_type！请选择 'pinhole' 或 'cylindrical'")

        rays_rect = np.stack([X, Y, Z], axis=-1)  # shape: (N, 3)

        # 3. 过滤掉指向背后的无效射线 (特别是 FOV > 180° 时)
        # 对于有效射线保留，无效射线强行置为极小的负无穷大，避免数学报错
        valid_mask = rays_rect[:, 2] > 0
        rays_rect[~valid_mask] = [0.0, 0.0, -1.0]

        # 4. 将射线从虚拟相机坐标系，旋转回原始真实的左右相机坐标系
        rays_orig_l = (np.linalg.inv(self.R1) @ rays_rect.T).T
        rays_orig_r = (np.linalg.inv(self.R2) @ rays_rect.T).T

        # 补充一个 Dummy 的外参矩阵 (R=I, t=0)，因为射线已经是在相机局部坐标系下了
        ext_identity = np.hstack([np.eye(3), np.zeros((3, 1))]).astype(np.float64)

        # 5. 调用你的多态 BaseCamera 实例，计算射线在原图上的重投影坐标
        uv_src_l = self.cam_l.project(rays_orig_l, ext_identity)
        uv_src_r = self.cam_r.project(rays_orig_r, ext_identity)

        # 针对刚才无效射线的像素，将其坐标设为 NaN，cv2.remap 会自动将其置黑
        uv_src_l[~valid_mask] = np.nan
        uv_src_r[~valid_mask] = np.nan

        # 6. 打包输出
        map1_l = uv_src_l[:, 0].reshape((th, tw)).astype(np.float32)
        map2_l = uv_src_l[:, 1].reshape((th, tw)).astype(np.float32)

        map1_r = uv_src_r[:, 0].reshape((th, tw)).astype(np.float32)
        map2_r = uv_src_r[:, 1].reshape((th, tw)).astype(np.float32)

        return (map1_l, map2_l), (map1_r, map2_r)



