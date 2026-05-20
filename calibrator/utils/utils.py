from pathlib import Path
import cv2
import numpy as np
from natsort import natsorted

__im_suffix = [".jpg", ".png", ".jpeg", ".bmp", ".tiff", ".tif"]


def glob_images(image_dir, pattern):
    """加载图像目录下的所有图像"""
    image_paths = Path(image_dir).rglob(pattern)
    image_paths = natsorted(list(map(str, image_paths)))
    image_paths = [p for p in image_paths if Path(p).suffix.lower() in __im_suffix]
    return image_paths


def glob_images_pairs(image_dir, pattern):
    """加载双目图像对，确保左右目图像数量相同且按文件名匹配"""
    left_paths = Path(image_dir).rglob(pattern)
    left_paths = natsorted(list(map(str, left_paths)))
    right_paths = [str(p).replace("left", "right") for p in left_paths]
    img_pairs = zip(left_paths, right_paths)
    im_pairs = (
        (lp, rp) for lp, rp in img_pairs if Path(lp).exists() and Path(rp).exists()
    )
    im_pairs = tuple(im_pairs)
    return im_pairs


def det_chessboard_corners(img_path, chessboard_size=(11, 8)):
    gray = cv2.imread(img_path, 0)

    if gray is None:
        return None, None, None

    # 检测棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if not ret:
        return None, gray.shape[:2], gray

    # 亚像素角点优化
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_refined = cv2.cornerSubPix(
        gray,
        corners,
        (11, 11),
        (-1, -1),
        criteria,
    )

    return corners_refined, gray.shape[:2], gray


def det_chessboard_corners_pair(img_pair, chessboard_size=(11, 8)):
    left_corners, left_size, gray_left = det_chessboard_corners(
        img_pair[0], chessboard_size
    )
    right_corners, right_size, gray_right = det_chessboard_corners(
        img_pair[1], chessboard_size
    )
    if left_corners is None or right_corners is None:
        return (None, None), (left_size, right_size), (gray_left, gray_right)

    return (left_corners, right_corners), (left_size, right_size), (gray_left, gray_right)


def gen_object_points(chessboard_size, square_size):
    WW, HH = chessboard_size
    objp = np.zeros((WW * HH, 3), np.float32)
    objp[:, :2] = np.mgrid[0:WW, 0:HH].T.reshape(-1, 2)
    objp *= square_size
    return objp



def compute_omni_q_matrix(K_left, D_left, xi_left, K_right, D_right, xi_right, R, T):
    """
    改进版Q矩阵计算（考虑全向相机xi参数）
    参数:
        xi_left, xi_right: 左右相机的镜面参数
        其他参数同标准Q矩阵计算
    """
    # 1. 计算等效焦距（考虑xi的影响）
    effective_focal = K_left[0,0] * (1 + xi_left)
    
    # 2. 计算修正后的主点
    cx = K_left[0,2] / (1 + xi_left)
    cy = K_left[1,2] / (1 + xi_left)
    
    # 3. 构造Q矩阵（含xi补偿）
    Q = np.zeros((4,4))
    Q[0,3] = -cx
    Q[1,3] = -cy
    Q[2,3] = effective_focal
    Q[3,2] = -1.0/np.linalg.norm(T)
    Q[2,2] = (cx - K_right[0,2]/(1+xi_right)) / np.linalg.norm(T)
    
    # 4. 畸变参数补偿（含xi影响）
    distortion_factor = 1.0 / (1.0 + 0.2*(np.sum(D_left) + np.sum(D_right)))
    Q[3,2] *= distortion_factor * (1 + (xi_left + xi_right)/2)
    
    return Q
def compute_relative_pose(rvec_left, tvec_left, rvec_right, tvec_right):
    """从左右相机外参计算相对位姿初始值"""
    R_l0, _ = cv2.Rodrigues(rvec_left)  # 第一组左相机旋转
    R_r0, _ = cv2.Rodrigues(rvec_right)  # 第一组右相机旋转
    # 右相对左的初始旋转（R_r = R_rel @ R_l → R_rel = R_r @ R_l.T）
    R_rel_init = (
            R_r0 @ R_l0.T
    )
    t_l0 = tvec_left.reshape(3, 1)
    t_r0 = tvec_right.reshape(3, 1)
    # 右相对左的初始平移（t_r = R_rel @ t_l + T_rel → T_rel = t_r - R_rel @ t_l）
    T_rel_init = (
            t_r0 - R_rel_init @ t_l0
    )
    R_rel_vec_init = cv2.Rodrigues(R_rel_init)[0].flatten()

    return R_rel_vec_init.flatten(), T_rel_init.flatten()