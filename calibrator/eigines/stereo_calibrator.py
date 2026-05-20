import numpy as np
import cv2
import scipy.sparse as sp

import os
import cv2
import numpy as np
import json

from ..core.base_engine import BaseCalibratorEngine


class StereoCalibratorEngine(BaseCalibratorEngine):
    def __init__(self, cam_l, cam_r, target_detector):
        super().__init__(target_detector)
        self.cam_l = cam_l
        self.cam_r = cam_r
        self.R_rel = np.eye(3, dtype=np.float64)
        self.t_rel = np.zeros((3, 1), dtype=np.float64)
        self.left_extrinsics = []
        self.fix_intrinsics = False

    def load_and_match_data(self, left_img_paths, right_img_paths, min_common_tags: int = 1):
        assert len(left_img_paths) == len(right_img_paths), "左右图像数量必须一致！"
        self.detections = []
        self.left_extrinsics = []

        valid_pairs = 0
        total_pairs = len(left_img_paths)
        print(f"开始处理 {total_pairs} 对双目图像进行特征匹配...")

        for pair_idx, (left_path, right_path) in enumerate(zip(left_img_paths, right_img_paths)):
            res_l = self.detector.detect(left_path)
            res_r = self.detector.detect(right_path)
            if not res_l or not res_r:
                continue

            common_ids = sorted(list(set(res_l.keys()) & set(res_r.keys())))
            if len(common_ids) < int(min_common_tags):
                continue

            img_l, img_r, obj = [], [], []
            for tid in common_ids:
                pts_l, obj_3d = res_l[tid]
                pts_r, _ = res_r[tid]
                img_l.append(np.asarray(pts_l, np.float64).reshape(-1, 2))
                img_r.append(np.asarray(pts_r, np.float64).reshape(-1, 2))
                obj.append(np.asarray(obj_3d, np.float64).reshape(-1, 3))

            self.detections.append({
                "img_pts_l": np.vstack(img_l),
                "img_pts_r": np.vstack(img_r),
                "obj_pts": np.vstack(obj),
                "left_path": left_path,
                "right_path": right_path,
                "pair_idx": pair_idx
            })
            self.left_extrinsics.append(np.hstack([np.eye(3), np.zeros((3, 1))]).astype(np.float64))
            valid_pairs += 1

        print(f"匹配完成！共提取了 {valid_pairs}/{total_pairs} 对有效双目共视数据。")
        assert len(self.detections) == len(self.left_extrinsics)

    # ---- mono 对齐 ----
    def _mono_extrinsic_map(self, mono_engine):
        ext_map = {}
        if hasattr(mono_engine, "detections") and mono_engine.detections:
            for det, ext in zip(mono_engine.detections, mono_engine.extrinsics):
                fid = det.get("frame_id", det.get("pair_idx", None))
                if fid is None:
                    continue
                ext_map[int(fid)] = ext
        if ext_map:
            return ext_map
        for i, ext in enumerate(mono_engine.extrinsics):
            ext_map[i] = ext
        return ext_map

    def filter_pairs_by_mono(self, mono_l_engine, mono_r_engine):
        mapL = self._mono_extrinsic_map(mono_l_engine)
        mapR = self._mono_extrinsic_map(mono_r_engine)

        new_dets, new_left = [], []
        dropped = 0
        for det in self.detections:
            k = int(det["pair_idx"])
            if (k in mapL) and (k in mapR):
                new_dets.append(det)
                new_left.append(np.hstack([np.eye(3), np.zeros((3, 1))]).astype(np.float64))
            else:
                dropped += 1

        self.detections = new_dets
        self.left_extrinsics = new_left
        print(f"Stereo pairs filtered by mono: kept={len(self.detections)}, dropped={dropped}")

    def initialize_left_extrinsics_from_mono(self, mono_l_engine):
        ext_map = self._mono_extrinsic_map(mono_l_engine)
        missing = 0
        for i, det in enumerate(self.detections):
            k = int(det["pair_idx"])
            if k in ext_map:
                self.left_extrinsics[i] = ext_map[k].copy()
            else:
                missing += 1
                self.left_extrinsics[i] = np.hstack([np.eye(3), np.zeros((3, 1))]).astype(np.float64)
        if missing:
            print(f"initialize_left_extrinsics_from_mono: missing={missing}/{len(self.detections)}")

    # ---- 相对位姿初始化：矩阵公式 + 自动选方向 ----
    def initialize_pose_from_monos(self, mono_l_engine, mono_r_engine):
        mapL = {int(d["frame_id"]): ext for d, ext in zip(mono_l_engine.detections, mono_l_engine.extrinsics)}
        mapR = {int(d["frame_id"]): ext for d, ext in zip(mono_r_engine.detections, mono_r_engine.extrinsics)}
        common = sorted(set(mapL.keys()) & set(mapR.keys()))
        if not common:
            raise RuntimeError("initialize_pose_from_monos: no common frame_id")

        k = common[0]
        ext_l = mapL[k]
        ext_r = mapR[k]

        def ext_to_T(ext):
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = ext[:, :3]
            T[:3, 3] = ext[:, 3]
            return T

        T_lo = ext_to_T(ext_l)  # obj->left
        T_ro = ext_to_T(ext_r)  # obj->right
        T_rl = T_ro @ np.linalg.inv(T_lo)
        T_lr = np.linalg.inv(T_rl)

        def score_T(T_test):
            if not self.detections:
                return 1e18
            d0 = self.detections[0]
            ext_l0 = self.left_extrinsics[0]
            R = T_test[:3, :3]
            t = T_test[:3, 3:4]
            R_r = R @ ext_l0[:, :3]
            t_r = R @ ext_l0[:, 3:4] + t
            ext_r0 = np.hstack((R_r, t_r))
            proj_r = np.asarray(self.cam_r.project(d0["obj_pts"], ext_r0), np.float64).reshape(-1, 2)
            obs_r = np.asarray(d0["img_pts_r"], np.float64).reshape(-1, 2)
            v = np.isfinite(proj_r).all(1)
            if v.sum() < 10:
                return 1e18
            return float(np.linalg.norm(obs_r[v] - proj_r[v], axis=1).mean())

        T_lo = ext_to_T(ext_l)  # obj->left
        T_ro = ext_to_T(ext_r)  # obj->right

        # 严格计算 Left -> Right，不需要再瞎猜方向
        T_rl = T_ro @ np.linalg.inv(T_lo)

        self.R_rel = T_rl[:3, :3].copy()
        self.t_rel = T_rl[:3, 3:4].copy()

        baseline = np.linalg.norm(self.t_rel)
        print(f">>> 相对位姿初始化完成(frame_id={k}). baseline={baseline:.4f}m")


    # ---- 必须实现的抽象方法（BaseCalibratorEngine 要求）----
    def _pack_params(self) -> np.ndarray:
        params = []
        if not self.fix_intrinsics:
            params.append(self.cam_l.pack_params())
            params.append(self.cam_r.pack_params())

        rvec_rel, _ = cv2.Rodrigues(self.R_rel)
        params.append(rvec_rel.flatten())
        params.append(self.t_rel.flatten())

        for ext in self.left_extrinsics:
            rvec, _ = cv2.Rodrigues(ext[:, :3])
            params.append(rvec.flatten())
            params.append(ext[:, 3].flatten())

        return np.concatenate(params) if params else np.array([], dtype=np.float64)

    def _unpack_params(self, params: np.ndarray):
        idx = 0
        if not self.fix_intrinsics:
            nl = self.cam_l.num_intrinsic_params + self.cam_l.num_dist_params
            nr = self.cam_r.num_intrinsic_params + self.cam_r.num_dist_params
            self.cam_l.unpack_params(params[idx:idx+nl]); idx += nl
            self.cam_r.unpack_params(params[idx:idx+nr]); idx += nr

        rvec_rel = params[idx:idx+3]
        self.t_rel = params[idx+3:idx+6].reshape(3, 1)
        self.R_rel, _ = cv2.Rodrigues(rvec_rel)
        idx += 6

        for i in range(len(self.left_extrinsics)):
            rvec = params[idx:idx+3]
            tvec = params[idx+3:idx+6].reshape(3, 1)
            idx += 6
            R, _ = cv2.Rodrigues(rvec)
            self.left_extrinsics[i] = np.hstack((R, tvec))

    def _pack_bounds(self) -> tuple:
        lower_all, upper_all = [], []
        if not self.fix_intrinsics:
            l_low, l_up = self.cam_l.get_bounds()
            r_low, r_up = self.cam_r.get_bounds()
            lower_all.extend([l_low, r_low])
            upper_all.extend([l_up, r_up])

        n_ext = 6 + 6 * len(self.left_extrinsics)
        lower_all.append(np.full(n_ext, -np.inf))
        upper_all.append(np.full(n_ext, np.inf))
        return np.concatenate(lower_all), np.concatenate(upper_all)

    def _residual_func(self, params: np.ndarray) -> np.ndarray:
        self._unpack_params(params)
        res = []
        for i, d in enumerate(self.detections):
            ext_l = self.left_extrinsics[i]
            R_r = self.R_rel @ ext_l[:, :3]
            t_r = self.R_rel @ ext_l[:, 3:4] + self.t_rel
            ext_r = np.hstack((R_r, t_r))

            obs_l = np.asarray(d["img_pts_l"], np.float64).reshape(-1, 2)
            obs_r = np.asarray(d["img_pts_r"], np.float64).reshape(-1, 2)

            proj_l = np.asarray(self.cam_l.project(d["obj_pts"], ext_l), np.float64).reshape(-1, 2)
            proj_r = np.asarray(self.cam_r.project(d["obj_pts"], ext_r), np.float64).reshape(-1, 2)

            vl = np.isfinite(proj_l).all(1)
            vr = np.isfinite(proj_r).all(1)

            # err_l = np.zeros_like(obs_l)
            # err_r = np.zeros_like(obs_r)
            err_l = np.full_like(obs_l, 1000.0)
            err_r = np.full_like(obs_r, 1000.0)
            err_l[vl] = obs_l[vl] - proj_l[vl]
            err_r[vr] = obs_r[vr] - proj_r[vr]

            res.append(err_l.reshape(-1))
            res.append(err_r.reshape(-1))

        return np.concatenate(res, axis=0) if res else np.array([], dtype=np.float64)

    def _jac_sparsity(self):
        pts_per_frame = [int(np.asarray(d["img_pts_l"]).reshape(-1, 2).shape[0]) for d in self.detections]
        rows_per_frame = [4 * n for n in pts_per_frame]
        m = int(sum(rows_per_frame))

        camL = len(self.cam_l.pack_params())
        camR = len(self.cam_r.pack_params())
        rel_dim = 6
        ext_dim = 6
        n_frames = len(self.detections)

        if not self.fix_intrinsics:
            n = camL + camR + rel_dim + n_frames * ext_dim
            camL0 = 0
            camR0 = camL
            rel0 = camL + camR
            ext0_base = rel0 + rel_dim
        else:
            n = rel_dim + n_frames * ext_dim
            rel0 = 0
            ext0_base = rel_dim

        J = sp.lil_matrix((m, n), dtype=np.int8)
        row0 = 0
        for fi, rcount in enumerate(rows_per_frame):
            r1 = row0 + rcount
            if not self.fix_intrinsics:
                J[row0:r1, camL0:camL0+camL] = 1
                J[row0:r1, camR0:camR0+camR] = 1
            J[row0:r1, rel0:rel0+rel_dim] = 1
            c0 = ext0_base + fi * ext_dim
            J[row0:r1, c0:c0+ext_dim] = 1
            row0 = r1
        return J.tocsr()


    def _draw_reproj_overlay(self,img, obs_pts, proj_pts, error_scale=10.0):
        """
        红绿十字法（与 mono 类似）：
        - 绿点：观测
        - 红点：重投影
        - 蓝线：从观测指向重投影（可选放大）
        """
        canvas = img.copy()

        obs = np.asarray(obs_pts, dtype=np.float64).reshape(-1, 2)
        proj = np.asarray(proj_pts, dtype=np.float64).reshape(-1, 2)

        # 误差向量（可放大）
        vec = (proj - obs) * float(error_scale)

        for (u, v), (du, dv), (pu, pv) in zip(obs, vec, proj):
            u_i, v_i = int(np.round(u)), int(np.round(v))
            pr_i, pv_i = int(np.round(pu)), int(np.round(pv))

            # 观测点：绿
            cv2.drawMarker(canvas, (u_i, v_i), (0, 255, 0),
                           markerType=cv2.MARKER_CROSS, markerSize=2, thickness=2,
                           line_type=cv2.LINE_AA)

            # 重投影点：红
            cv2.drawMarker(canvas, (pr_i, pv_i), (0, 0, 255),
                           markerType=cv2.MARKER_TILTED_CROSS, markerSize=2, thickness=2,
                           line_type=cv2.LINE_AA)

            # 误差线（放大后的方向）
            end_pt = (int(np.round(u + du)), int(np.round(v + dv)))
            cv2.line(canvas, (u_i, v_i), end_pt, (255, 0, 0), 1, cv2.LINE_AA)

        return canvas

    def _put_text(self,img, lines, org=(20, 40), line_h=30, scale=0.8, color=(255, 255, 0), thickness=2):
        x, y = org
        for i, t in enumerate(lines):
            cv2.putText(img, str(t), (x, y + i * line_h),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

    def visualize_reprojection(self, wait_time: int = 0, error_scale: float = 10.0,
                               max_pairs: int | None = None, save_dir: str | None = None):
        """
        双目重投影可视化：
        左窗口：Left Reprojection
        右窗口：Right Reprojection

        按键：
          - q：退出
          - n / 空格：下一对
          - b：上一对
          - s：保存当前叠加图到 save_dir（默认 ./stereo_reproj_debug）
        """
        if not self.detections:
            print("请先执行 load_and_match_data()！")
            return
        if not self.left_extrinsics or len(self.left_extrinsics) != len(self.detections):
            print("left_extrinsics 数量不匹配！请确保在 load_and_match_data() 中为每对有效数据初始化 left_extrinsics。")
            return

        if save_dir is None:
            save_dir = "./stereo_reproj_debug"
        os.makedirs(save_dir, exist_ok=True)

        N = len(self.detections)
        if max_pairs is not None:
            N = min(N, int(max_pairs))

        print(f"\n=== 开始双目重投影验证 (共 {N} 对，误差放大 {error_scale} 倍) ===")
        print("按键: [n/空格]下一对  [b]上一对  [s]保存  [q]退出")

        i = 0
        while 0 <= i < N:
            data = self.detections[i]
            left_path = data.get("left_path", None)
            right_path = data.get("right_path", None)

            img_l = cv2.imread(left_path, cv2.IMREAD_COLOR) if left_path else None
            img_r = cv2.imread(right_path, cv2.IMREAD_COLOR) if right_path else None

            if img_l is None or img_r is None:
                print(f"[{i}] 读图失败，跳过。left={left_path}, right={right_path}")
                i += 1
                continue

            obs_l = np.asarray(data["img_pts_l"], dtype=np.float64).reshape(-1, 2)
            obs_r = np.asarray(data["img_pts_r"], dtype=np.float64).reshape(-1, 2)
            obj   = np.asarray(data["obj_pts"],   dtype=np.float64).reshape(-1, 3)

            # 当前这对的左外参
            ext_l = self.left_extrinsics[i]

            # 由相对位姿计算右外参
            R_r = self.R_rel @ ext_l[:, :3]
            t_r = self.R_rel @ ext_l[:, 3:4] + self.t_rel
            ext_r = np.hstack((R_r, t_r))

            # 重投影
            proj_l = self.cam_l.project(obj, ext_l)
            proj_r = self.cam_r.project(obj, ext_r)

            proj_l = np.nan_to_num(proj_l, nan=1e5).reshape(-1, 2)
            proj_r = np.nan_to_num(proj_r, nan=1e5).reshape(-1, 2)

            # 误差统计
            err_l = np.linalg.norm(obs_l - proj_l, axis=1)
            err_r = np.linalg.norm(obs_r - proj_r, axis=1)

            # 叠加绘制
            canvas_l = self._draw_reproj_overlay(img_l, obs_l, proj_l, error_scale=error_scale)
            canvas_r = self._draw_reproj_overlay(img_r, obs_r, proj_r, error_scale=error_scale)

            self._put_text(canvas_l, [
                f"[{i+1}/{N}] LEFT",
                f"pts={len(obs_l)}",
                f"mean={err_l.mean():.2f}px  max={err_l.max():.2f}px",
            ], org=(20, 40))

            self._put_text(canvas_r, [
                f"[{i+1}/{N}] RIGHT",
                f"pts={len(obs_r)}",
                f"mean={err_r.mean():.2f}px  max={err_r.max():.2f}px",
            ], org=(20, 40))

            # 缩放显示（防止窗口太大）
            def _resize_to_width(img, max_w=1200):
                h, w = img.shape[:2]
                if w <= max_w:
                    return img
                s = max_w / w
                return cv2.resize(img, None, fx=s, fy=s)

            canvas_l_show = _resize_to_width(canvas_l, 1200)
            canvas_r_show = _resize_to_width(canvas_r, 1200)

            cv2.imshow("Stereo Reprojection - LEFT", canvas_l_show)
            cv2.imshow("Stereo Reprojection - RIGHT", canvas_r_show)

            key = cv2.waitKey(wait_time) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('b'):
                i = max(0, i - 1)
            elif key == ord('s'):
                base = f"{i:03d}"
                out_l = os.path.join(save_dir, f"{base}_left.png")
                out_r = os.path.join(save_dir, f"{base}_right.png")
                cv2.imwrite(out_l, canvas_l)
                cv2.imwrite(out_r, canvas_r)
                print("saved:", out_l, out_r)
            else:
                # n / 空格 / 其它键：下一张
                i += 1

        cv2.destroyAllWindows()
    def save_results(self, filepath: str):
        """将双目标定结果保存到 JSON 文件中"""
        # 构建要保存的数据结构
        calib_data = {
            "camera_left": self.cam_l.to_dict(),
            "camera_right": self.cam_r.to_dict(),
            "extrinsics": {
                "R_rel": self.R_rel.tolist(),
                "t_rel": self.t_rel.tolist(),
                "baseline_meters": float(np.linalg.norm(self.t_rel))
            },
            "stats": {
                "valid_pairs": len(self.detections),
                # 如果你算了重投影误差，可以存进来
                # "reproj_error_left": float(err_l),
                # "reproj_error_right": float(err_r)
            }
        }

        # 写入文件
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(calib_data, f, indent=4, ensure_ascii=False)
        print(f"✅ 双目标定结果已成功保存至: {filepath}")

    def load_results(self, filepath: str):
        """从文件加载双目标定结果 (为后续 Remap 脚本准备)"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.cam_l.from_dict(data["camera_left"])
        self.cam_r.from_dict(data["camera_right"])
        self.R_rel = np.array(data["extrinsics"]["R_rel"], dtype=np.float64)
        self.t_rel = np.array(data["extrinsics"]["t_rel"], dtype=np.float64)
        print(f"✅ 成功从 {filepath} 加载标定参数！")