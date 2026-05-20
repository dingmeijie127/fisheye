from abc import ABC, abstractmethod
from typing import Tuple, Dict
import numpy as np

class BaseTargetDetector(ABC):
    @abstractmethod
    def detect(self, image_path: str) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        检测图像中的标定板特征点。
        :return: 字典 {特征点ID: (2D像素坐标数组, 3D世界坐标数组)}
        """
        pass