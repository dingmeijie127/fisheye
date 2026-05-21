from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Optional

def create_camera_from_dict(data: dict) -> 'BaseCamera':
    """相机工厂：根据字典中的 model_type 动态实例化相机"""
    model_type = data.get("model_type")

    # 延迟导入，避免循环引用
    from calibrator.models.DS import DoubleSphereCamera
    from calibrator.models.eucm import EUCMCamera
    from calibrator.models.fisheye import FisheyeCamera
    from calibrator.models.omnidirCamera import OmnidirCamera
    from calibrator.models.pinhole import PinholeCamera

    # 注册你的所有相机模型
    CAMERA_REGISTRY = {
        "DoubleSphereCamera": DoubleSphereCamera,
        "EUCMCamera": EUCMCamera,
        "FisheyeCamera": FisheyeCamera,
        "OmnidirCamera": OmnidirCamera,
        "PinholeCamera": PinholeCamera
    }

    if model_type not in CAMERA_REGISTRY:
        raise ValueError(f"未知的相机模型: {model_type}，请在工厂类中注册！")

    # 1. 动态实例化对应类的对象 (不需要传参，使用默认初值初始化)
    cam_instance = CAMERA_REGISTRY[model_type]()

    # 2. 调用基类的 from_dict 填入真实的标定参数
    cam_instance.from_dict(data)

    return cam_instance

class BaseCamera(ABC):
    """
    终极版相机基类：采用声明式参数注册。
    子类只需定义两个元组列表，格式: ('参数名', 默认初值, 下界(可选), 上界(可选))
    如果没有界限，填 None 或 `-np.inf, np.inf`
    """
    # 格式示例: [('fx', 500.0, 1e-3, np.inf), ...]
    INTRINSIC_DEFS = []
    DISTORTION_DEFS = []

    def __init__(self, intrinsics: Optional[List[float]] = None,
                 distortion: Optional[List[float]] = None):
        """
        初始化相机参数。如果传入，使用传入值；否则使用 DEFS 中的默认初值。
        """
        self.intrinsics = self._initialize_params(self.INTRINSIC_DEFS, intrinsics)
        self.distortion = self._initialize_params(self.DISTORTION_DEFS, distortion)

    def _initialize_params(self, defs, input_vals):
        """根据定义表和输入，初始化参数数组"""
        if input_vals is not None:
            if len(input_vals) != len(defs):
                raise ValueError(f"参数长度不匹配，期望 {len(defs)}，实际输入 {len(input_vals)}")
            return np.array(input_vals, dtype=np.float64)
        else:
            # 使用定义表里的默认初值 (defs[i][1])
            return np.array([d[1] for d in defs], dtype=np.float64)

    @property
    def num_intrinsic_params(self) -> int:
        return len(self.INTRINSIC_DEFS)

    @property
    def num_dist_params(self) -> int:
        return len(self.DISTORTION_DEFS)

    @property
    def dist_param_names(self) -> List[str]:
        return [d[0] for d in self.DISTORTION_DEFS]

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        [神级操作] 基类自动遍历定义表，提取边界。
        利用 `len(d)` 判断自动兼容没有写 None 的偷懒写法。
        """
        all_defs = self.INTRINSIC_DEFS + self.DISTORTION_DEFS
        lower_bounds = []
        upper_bounds = []

        for d in all_defs:
            # d[2] 是下界，d[3] 是上界。如果没写，默认为负无穷到正无穷
            lower = d[2] if len(d) > 2 and d[2] is not None else -np.inf
            upper = d[3] if len(d) > 3 and d[3] is not None else np.inf
            lower_bounds.append(lower)
            upper_bounds.append(upper)

        return np.array(lower_bounds), np.array(upper_bounds)

    def pack_params(self) -> np.ndarray:
        """打包内参和畸变为 1D 数组"""
        return np.concatenate([self.intrinsics, self.distortion])

    def unpack_params(self, params: np.ndarray):
        """完全动态的解包逻辑"""
        n_in = self.num_intrinsic_params
        self.intrinsics = params[:n_in]
        self.distortion = params[n_in:]

    @abstractmethod
    def project(self, points_3d: np.ndarray, extrinsics: np.ndarray) -> np.ndarray:
        """3D 到 2D 投影接口"""
        pass

    @abstractmethod
    def unproject(self, points_2d: np.ndarray) -> np.ndarray:
        """2D 到 3D 射线反投影接口 (用于立体校正或射线空间匹配)"""
        pass

    def to_dict(self) -> dict:
        """将当前相机的所有参数打包为字典"""
        return {
            "model_type": self.__class__.__name__,
            "intrinsics": self.intrinsics.tolist(),
            "distortion": self.distortion.tolist(),
            # "image_size": [self.width, self.height] # 如果你存了的话
        }

    def from_dict(self, data: dict):
        """从字典加载参数"""
        self.intrinsics = np.array(data["intrinsics"], dtype=np.float64)
        self.distortion = np.array(data["distortion"], dtype=np.float64)