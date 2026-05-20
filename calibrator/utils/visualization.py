import cv2
import numpy as np

def draw_reprojection_errors(img: np.ndarray, obs_pts: np.ndarray,
                             proj_pts: np.ndarray, error_scale: float = 1.0) -> np.ndarray:
    """
    在画布上绘制观测点 (绿圆) 和重投影点 (红十字) 的差异。
    error_scale: 误差向量放大倍数。
    """
    canvas = img.copy()

    # 过滤掉投影失败的点
    valid_mask = ~np.isnan(proj_pts[:, 0])
    obs_pts = obs_pts[valid_mask]
    proj_pts = proj_pts[valid_mask]

    for obs, proj in zip(obs_pts, proj_pts):
        p_obs = (int(round(obs[0])), int(round(obs[1])))
        p_proj = (int(round(proj[0])), int(round(proj[1])))

        # 1. 观测点 -> 绿色空心圆
        cv2.circle(canvas, p_obs, radius=4, color=(0, 255, 0), thickness=1)

        # 2. 投影点 -> 红色十字
        cv2.drawMarker(canvas, p_proj, color=(0, 0, 255), markerType=cv2.MARKER_CROSS,
                       markerSize=6, thickness=1)

        # 3. 误差连线 -> 黄色线
        if error_scale != 1.0:
            # 放大误差向量，方便肉眼看
            dx = (proj[0] - obs[0]) * error_scale
            dy = (proj[1] - obs[1]) * error_scale
            p_end = (int(round(obs[0] + dx)), int(round(obs[1] + dy)))
            cv2.line(canvas, p_obs, p_end, color=(0, 255, 255), thickness=1)
        else:
            cv2.line(canvas, p_obs, p_proj, color=(0, 255, 255), thickness=1)

    return canvas