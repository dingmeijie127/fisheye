import sys
import os
import json
import cv2
import numpy as np
from pathlib import Path

# 确保能 import calibrator
root_path = str(Path(__file__).resolve().parent.parent)
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from calibrator.utils.utils import glob_images_pairs
from calibrator.models.DS import DoubleSphereCamera
from calibrator.core.stereo_rectifier import StereoRectifier


def load_calib_result(json_path):
    """加载 JSON 标定结果"""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"找不到标定结果文件: {json_path}\n请先运行标定脚本！")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    cam_l = DoubleSphereCamera()
    cam_r = DoubleSphereCamera()

    # 假设你已经在 base_camera.py 中实现了 from_dict 方法
    cam_l.from_dict(data["camera_left"])
    cam_r.from_dict(data["camera_right"])

    R_rel = np.array(data["extrinsics"]["R_rel"], dtype=np.float64)
    t_rel = np.array(data["extrinsics"]["t_rel"], dtype=np.float64)

    return cam_l, cam_r, R_rel, t_rel


def run_stereo_remap_test():
    # 1. 路径配置
    json_path = os.path.join(root_path, "calibrator", "data", "results", "stereo_ds_calib_result.json")
    img_dir = "/mnt/d/data/fisheye/images/back"  # 你的测试数据集路径
    save_dir = os.path.join(root_path, "calibrator", "data", "results", "remap_output")
    os.makedirs(save_dir, exist_ok=True)

    # 2. 获取图像并加载相机参数
    img_pairs = glob_images_pairs(img_dir, "left*.bmp")
    if not img_pairs:
        print(f"在 {img_dir} 中没有找到图像对！")
        return

    print(f"成功读取标定文件，开始处理 {len(img_pairs)} 对图像...")
    cam_l, cam_r, R_rel, t_rel = load_calib_result(json_path)

    # 读取第一张图获取分辨率
    sample_img = cv2.imread(img_pairs[0][0])
    orig_h, orig_w = sample_img.shape[:2]

    # 3. 初始化极线校正器
    rectifier = StereoRectifier(cam_l, cam_r, R_rel, t_rel, orig_size=(orig_w, orig_h))

    # ==========================================================
    # 🌟 核心配置区：你可以随时在这里切换 pinhole 还是 cylindrical
    # ==========================================================
    PROJ_TYPE = 'cylindrical'  # 'cylindrical' (无黑边大视野) 或 'pinhole' (传统针孔)
    TARGET_W = orig_w
    TARGET_H = int(orig_h * 0.8) # 柱面投影时，适当裁剪上下高度可消除黑边

    print(f"正在生成 {PROJ_TYPE} 映射表 (目标分辨率: {TARGET_W}x{TARGET_H})... 这可能需要几秒钟。")
    maps_l, maps_r = rectifier.generate_remap_arrays(
        target_size=(TARGET_W, TARGET_H),
        proj_type=PROJ_TYPE,
        fov_h_deg=120.0,    # 如果是柱面，填入镜头的水平视场角
        scale_factor=0.6    # 如果是针孔，填入缩放因子
    )
    print("映射表生成完毕！开始 Remap...\n")

    # 4. 遍历处理并保存
    for idx, (left_path, right_path) in enumerate(img_pairs):
        img_l = cv2.imread(left_path)
        img_r = cv2.imread(right_path)

        if img_l is None or img_r is None:
            continue

        # OpenCV 重映射加速 (INTER_LINEAR 双线性插值，边缘用黑色填充)
        rect_l = cv2.remap(img_l, maps_l[0], maps_l[1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        rect_r = cv2.remap(img_r, maps_r[0], maps_r[1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        # 把左右图横向拼在一起，方便画连贯的极线
        concat_img = cv2.hconcat([rect_l, rect_r])

        # 画等间距的绿色水平对齐线
        for y in range(50, TARGET_H, 50):
            cv2.line(concat_img, (0, y), (TARGET_W * 2, y), (0, 255, 0), 1, cv2.LINE_AA)

        # 保存结果
        base_name = os.path.basename(left_path).replace("left", "remap_pair")
        out_path = os.path.join(save_dir, base_name)
        cv2.imwrite(out_path, concat_img)

        # 终端进度提示
        print(f"[{idx+1}/{len(img_pairs)}] 已保存: {out_path}")

        # 如果你想在屏幕上实时看，解除下面的注释 (会对宽图进行缩放显示)
        show_img = cv2.resize(concat_img, (0,0), fx=0.5, fy=0.5)
        cv2.imshow("Stereo Rectification Check", show_img)
        key = cv2.waitKey(10) & 0xFF
        if key == 27 or key == ord('q'):  # 按 ESC 或 q 退出预览
            break

    cv2.destroyAllWindows()
    print("\n✅ 所有重映射图像处理完毕！")

if __name__ == "__main__":
    run_stereo_remap_test()