import cv2
import numpy as np
import os
import scipy.sparse as sp
import json
import yaml
from ..core.base_engine import BaseCalibratorEngine
from ..core.base_camera import BaseCamera
from ..core.base_target import BaseTargetDetector
from ..utils.visualization import draw_reprojection_errors


def _draw_points(img, pts, color=(0, 255, 0), r=4, thickness=2):
    pts = np.asarray(pts)
    if pts.ndim == 1:
        pts = pts.reshape(1, 2)
    elif pts.ndim == 3 and pts.shape[1] == 1 and pts.shape[2] == 2:
        pts = pts.reshape(-1, 2)
    else:
        pts = pts.reshape(-1, 2)

    for u, v in pts:
        u = int(np.round(float(u)))
        v = int(np.round(float(v)))
        cv2.circle(img, (u, v), r, color, thickness, lineType=cv2.LINE_AA)


def _draw_text(img, text, org=(20, 40), color=(255, 255, 0), scale=1.0, thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


class MonoCalibratorEngine(BaseCalibratorEngine):
    def __init__(self, camera: BaseCamera, target_detector: BaseTargetDetector):
        super().__init__(target_detector)
        self.camera = camera
        self.extrinsics = []  # 与 detections 一一对应的 3x4 外参

    def load_data_and_initialize(self, image_paths: list, debug_vis=False, save_dir="./debug_pts"):
        """
        加载单目图片，提取特征点，并使用 PnP 给外参初值。
        关键修复：
          - detection 中保存 frame_id（原始 idx），用于 stereo 对齐
          - PnP 用 ITERATIVE + RANSAC + refineLM，更稳
          - 如果 intrinsics 初值不靠谱，会用图像尺寸初始化 fx/fy/cx/cy
        """
        print(f"开始提取 {len(image_paths)} 张图像的特征点...")

        self.detections = []
        self.extrinsics = []

        if debug_vis:
            os.makedirs(save_dir, exist_ok=True)

        # 用第一张可读图初始化一个合理内参（避免 EUCM/Omni 初值太烂）
        img0 = None
        for p in image_paths:
            img0 = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img0 is not None:
                break
        if img0 is None:
            raise RuntimeError("No readable images in image_paths.")

        h, w = img0.shape[:2]
        # 如果你 camera.intrinsics 是默认 600/640/360 那类，也没关系；这里给更稳的初值
        if hasattr(self.camera, "intrinsics") and len(self.camera.intrinsics) >= 4:
            fx, fy, cx, cy = self.camera.intrinsics[:4]
            # 若 fx/fy 太小，说明初值不可信，重置一下
            if fx < 50 or fy < 50:
                self.camera.intrinsics[0] = 0.8 * w
                self.camera.intrinsics[1] = 0.8 * h
                self.camera.intrinsics[2] = 0.5 * w
                self.camera.intrinsics[3] = 0.5 * h

        fx, fy, cx, cy = self.camera.intrinsics[:4]
        K_init = np.array([[fx, 0, cx],
                           [0, fy, cy],
                           [0, 0, 1]], dtype=np.float64)

        # PnP 用的 dist 先给 0（你真正的畸变由 BA 来估计）
        dist_init = np.zeros((4, 1), dtype=np.float64)

        for idx, path in enumerate(image_paths):
            res = self.detector.detect(path)
            if not res:
                print(f"[{idx}] no detection: {path}")
                continue

            img_pts_all = np.vstack([data[0] for data in res.values()]).astype(np.float64).reshape(-1, 2)
            obj_pts_all = np.vstack([data[1] for data in res.values()]).astype(np.float64).reshape(-1, 3)

            if len(img_pts_all) < 20:
                print(f"[{idx}] too few points: {len(img_pts_all)} {path}")
                continue

            # ✅ 稳：RANSAC + ITERATIVE
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                obj_pts_all, img_pts_all, K_init, dist_init,
                flags=cv2.SOLVEPNP_ITERATIVE,
                reprojectionError=3.0,
                iterationsCount=200
            )
            if not ok:
                print(f"警告: 图像 {path} PnP 初始化失败，跳过。")
                continue

            # ✅ refine（可选，但常常让外参初值更稳）
            try:
                rvec, tvec = cv2.solvePnPRefineLM(obj_pts_all, img_pts_all, K_init, dist_init, rvec, tvec)
            except Exception:
                pass

            # debug 可视化
            if debug_vis:
                img_vis = cv2.imread(path, cv2.IMREAD_COLOR)
                if img_vis is not None:
                    _draw_points(img_vis, img_pts_all, color=(0, 255, 0), r=4, thickness=2)
                    proj, _ = cv2.projectPoints(obj_pts_all, rvec, tvec, K_init, dist_init)
                    proj = proj.reshape(-1, 2)
                    _draw_points(img_vis, proj, color=(0, 0, 255), r=3, thickness=2)

                    err = np.linalg.norm(img_pts_all - proj, axis=1)
                    _draw_text(img_vis, f"PnP OK | pts={len(img_pts_all)} | mean={err.mean():.2f}px max={err.max():.2f}px",
                               org=(20, 40))

                    # resize show
                    hh, ww = img_vis.shape[:2]
                    scale = 1200 / ww if ww > 1200 else 1.0
                    if scale != 1.0:
                        img_vis = cv2.resize(img_vis, None, fx=scale, fy=scale)

                    cv2.imshow("Detections (green) + PnP reproj (red)", img_vis)
                    key = cv2.waitKey(0) & 0xFF
                    if key == ord('q'):
                        cv2.destroyAllWindows()
                        return
                    elif key == ord('s'):
                        out_name = os.path.join(save_dir, f"{idx:03d}_" + os.path.basename(path).rsplit('.', 1)[0] + ".png")
                        cv2.imwrite(out_name, img_vis)
                        print("saved:", out_name)

            # 记录 detection（✅ 用 frame_id 对齐 stereo pair_idx）
            self.detections.append({
                "frame_id": idx,          # ✅ 关键：原始序号
                "img_pts": img_pts_all,
                "obj_pts": obj_pts_all,
                "path": path
            })

            R, _ = cv2.Rodrigues(rvec)
            self.extrinsics.append(np.hstack((R, tvec.reshape(3, 1))))

        if debug_vis:
            cv2.destroyAllWindows()

        if not self.detections:
            raise RuntimeError("mono: no valid detections.")

    def _pack_params(self) -> np.ndarray:
        params = [self.camera.pack_params()]
        for ext in self.extrinsics:
            rvec, _ = cv2.Rodrigues(ext[:, :3])
            params.append(rvec.flatten())
            params.append(ext[:, 3].flatten())
        return np.concatenate(params)

    def _unpack_params(self, params: np.ndarray):
        n_cam = self.camera.num_intrinsic_params + self.camera.num_dist_params
        self.camera.unpack_params(params[:n_cam])

        idx = n_cam
        for i in range(len(self.extrinsics)):
            rvec = params[idx: idx + 3]
            tvec = params[idx + 3: idx + 6].reshape(3, 1)
            idx += 6
            R, _ = cv2.Rodrigues(rvec)
            self.extrinsics[i] = np.hstack((R, tvec))

    def _pack_bounds(self) -> tuple:
        cam_lower, cam_upper = self.camera.get_bounds()
        n_ext = 6 * len(self.extrinsics)
        ext_lower = np.full(n_ext, -np.inf)
        ext_upper = np.full(n_ext, np.inf)
        return np.concatenate([cam_lower, ext_lower]), np.concatenate([cam_upper, ext_upper])

    # ✅ 稀疏 Jacobian：每帧残差只依赖 cam 参数 + 本帧外参
    def _jac_sparsity(self):
        n_cam = self.camera.num_intrinsic_params + self.camera.num_dist_params
        ext_dim = 6
        n_frames = len(self.detections)

        rows_per_frame = [int(np.asarray(d["img_pts"]).reshape(-1, 2).shape[0]) * 2 for d in self.detections]
        m = int(sum(rows_per_frame))
        n = int(n_cam + n_frames * ext_dim)

        J = sp.lil_matrix((m, n), dtype=np.int8)

        row0 = 0
        for fi, rcount in enumerate(rows_per_frame):
            r1 = row0 + rcount

            # cam block
            J[row0:r1, 0:n_cam] = 1
            # this frame extrinsic block
            c0 = n_cam + fi * ext_dim
            J[row0:r1, c0:c0 + ext_dim] = 1

            row0 = r1

        return J.tocsr()

    def _residual_func(self, params: np.ndarray) -> np.ndarray:
        self._unpack_params(params)

        residuals = []
        for i, data in enumerate(self.detections):
            obs = np.asarray(data["img_pts"], dtype=np.float64).reshape(-1, 2)
            proj = np.asarray(self.camera.project(data["obj_pts"], self.extrinsics[i]), dtype=np.float64).reshape(-1, 2)

            valid = np.isfinite(proj).all(axis=1)
            # err = np.zeros_like(obs)
            err = np.full_like(obs, 500.0)
            err[valid] = obs[valid] - proj[valid]
            residuals.append(err.reshape(-1))

        return np.concatenate(residuals, axis=0)

    def get_extrinsics_map(self):
        """
        给 stereo 用：frame_id -> ext
        """
        m = {}
        for det, ext in zip(self.detections, self.extrinsics):
            m[int(det["frame_id"])] = ext
        return m

    def visualize_reprojection(self, wait_time: int = 0, error_scale: float = 10.0):
        if not self.extrinsics:
            print("请先执行优化 optimize()!")
            return

        print(f"\n=== 开始单目重投影验证 (误差放大 {error_scale} 倍，按 'q' 退出) ===")

        for i, data in enumerate(self.detections):
            img = cv2.imread(data["path"])
            if img is None:
                continue

            obs_pts = data["img_pts"]
            proj_pts = self.camera.project(data["obj_pts"], self.extrinsics[i])

            canvas = draw_reprojection_errors(img, obs_pts, proj_pts, error_scale=error_scale)

            h, w = canvas.shape[:2]
            scale = 1000 / w if w > 1000 else 1.0
            if scale != 1.0:
                canvas = cv2.resize(canvas, None, fx=scale, fy=scale)

            cv2.imshow("Mono Reprojection", canvas)
            key = cv2.waitKey(wait_time) & 0xFF
            if key == ord('q'):
                break

        cv2.destroyAllWindows()


    # ================= 补充在 MonoCalibratorEngine 类的最后 =================

    def save_results(self, filepath: str, save_extrinsics: bool = False):
        """
        将单目标定结果保存到 JSON 或 YAML 文件中。
        """
        calib_data = {
            "camera": self.camera.to_dict(),
            "stats": {
                "valid_frames": len(self.detections),
            }
        }

        # 外参通常很大，默认不保存，调试需要时可以开启
        if save_extrinsics:
            ext_list = []
            for det, ext in zip(self.detections, self.extrinsics):
                ext_list.append({
                    "frame_id": int(det.get("frame_id", -1)),
                    "path": str(det.get("path", "")),
                    "extrinsic": ext.tolist()
                })
            calib_data["extrinsics"] = ext_list

        ext = os.path.splitext(filepath)[-1].lower()
        if ext in ['.yaml', '.yml']:
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(calib_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        else:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(calib_data, f, indent=4, ensure_ascii=False)

        print(f"✅ 单目标定结果已保存至: {filepath}")

    def load_results(self, filepath: str):
        """
        从文件动态加载单目标定结果
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"找不到标定结果文件: {filepath}")

        ext = os.path.splitext(filepath)[-1].lower()
        if ext in ['.yaml', '.yml']:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        else:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

        from ..core.base_camera import create_camera_from_dict

        # 魔法在这：自动识别模型并创建实例
        self.camera = create_camera_from_dict(data["camera"])

        print(f"✅ 成功从 {filepath} 加载单目标定参数！")
        print(f"   -> 自动识别相机模型: {self.camera.__class__.__name__}")

        if "extrinsics" in data:
            self.extrinsics = [np.array(item["extrinsic"], dtype=np.float64) for item in data["extrinsics"]]
            self.detections = [{"frame_id": item["frame_id"], "path": item["path"]} for item in data["extrinsics"]]