import sys
from pathlib import Path
import os

from calibrator.models.omnidirCamera import OmnidirCamera
from calibrator.models.pinhole import PinholeCamera
from calibrator.models.DS import DoubleSphereCamera
from calibrator.models.eucm import EUCMCamera
from calibrator.models.fisheye import FisheyeCamera

root_path = str(Path(__file__).resolve().parent.parent)
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from calibrator.utils.utils import glob_images_pairs
from calibrator.models.eucm import EUCMCamera
from calibrator.targets.apriltag import AprilTagDetector
from calibrator.eigines.mono_calibrator import MonoCalibratorEngine
from calibrator.eigines.stereo_calibrator import StereoCalibratorEngine


def run_calibration_flow():
    # 0) Detector + data
    target = AprilTagDetector(
        tag_family="t36h11",
        grid_shape=(6, 6),
        tag_size=0.055,
        tag_spacing=0.0165
    )
    save_dir = "../calibrator/data/results"
    os.makedirs(save_dir, exist_ok=True)

    img_pairs = glob_images_pairs("/mnt/d/data/fisheye/images/back", "left*.bmp")
    # img_pairs = glob_images_pairs("/mnt/d/data/P300下/fisheye/116/new/GDU9PZ22P300A25M0116/144250_下双目/calibdata", "left*.bmp")
    left_imgs = [p[0] for p in img_pairs]
    right_imgs = [p[1] for p in img_pairs]

    # 1) Mono Left
    print("\n--- Step 1: Mono Left ---")
    cam_l = OmnidirCamera()
    mono_l = MonoCalibratorEngine(cam_l, target)

    # debug_vis 建议先关掉，不然会非常慢；需要看点再开
    mono_l.load_data_and_initialize(left_imgs, debug_vis=False)
    mono_l.optimize(use_sparse_jac=True, loss="soft_l1", f_scale=2.0, max_nfev=200)
    mono_l.save_results(os.path.join(save_dir, "mono_left_calib_result.yaml"))

    # 2) Mono Right
    print("\n--- Step 2: Mono Right ---")
    cam_r = OmnidirCamera()
    mono_r = MonoCalibratorEngine(cam_r, target)
    mono_r.load_data_and_initialize(right_imgs, debug_vis=False)
    mono_r.optimize(use_sparse_jac=True, loss="soft_l1", f_scale=2.0, max_nfev=200)
    mono_r.save_results(os.path.join(save_dir, "mono_left_calib_result.yaml"))

    # 3) Stereo
    print("\n--- Step 3: Stereo Pose Initialization ---")
    stereo_engine = StereoCalibratorEngine(cam_l, cam_r, target)


    stereo_engine.load_and_match_data(left_imgs, right_imgs, min_common_tags=3)

    # ✅ 交集过滤（关键：避免 pair_idx 在 mono 里不存在）
    stereo_engine.filter_pairs_by_mono(mono_l, mono_r)

    # ✅ 初始化 left 外参（按 frame_id/pair_idx 映射，不再 list 下标）
    stereo_engine.initialize_left_extrinsics_from_mono(mono_l)

    # ✅ 初始化相对位姿（更稳：从共同帧算）
    stereo_engine.initialize_pose_from_monos(mono_l, mono_r)

    stereo_engine.fix_intrinsics = False

    # 4) Stereo Joint BA
    print("\n--- Step 4: Stereo Joint BA ---")
    stereo_engine.optimize(use_sparse_jac=True, loss="soft_l1", f_scale=1.0, max_nfev=200)


    # 5) Visualize
    # stereo_engine.visualize_reprojection(error_scale=5.0)



    result_path = os.path.join(save_dir, "stereo_ds_calib_result.json")
    stereo_engine.save_results(result_path)


if __name__ == "__main__":
    run_calibration_flow()