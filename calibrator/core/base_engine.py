from abc import ABC, abstractmethod
import numpy as np
from scipy.optimize import least_squares
import scipy.sparse as sp
class BaseCalibratorEngine(ABC):
    def __init__(self, target_detector):
        self.detector = target_detector
        # 存储观测数据管线
        self.detections = []

    @abstractmethod


    @abstractmethod
    def _pack_params(self) -> np.ndarray:
        """打包所有需要优化的参数 (内参+畸变+外参等)"""
        pass

    @abstractmethod
    def _unpack_params(self, params: np.ndarray):
        """将一维数组解析回对象实例中"""
        pass

    @abstractmethod
    def _pack_bounds(self) -> tuple:
        """组装所有参数的边界：(lower_array, upper_array)"""
        pass

    @abstractmethod
    def _residual_func(self, params: np.ndarray) -> np.ndarray:
        """计算重投影残差"""
        pass

    def optimize(self, verbose=2, use_sparse_jac=True, **kwargs):
        """
        TRF + bounds + (可选) jac_sparsity 稀疏加速 + 鲁棒损失
        """
        if not getattr(self, "detections", None):
            raise ValueError("没有检测到有效数据，请先加载图像！")

        x0 = self._pack_params()
        bounds = self._pack_bounds()

        print(f"\n开始光束平差优化，待优化变量维度: {len(x0)}")

        loss = kwargs.pop("loss", "soft_l1")
        f_scale = kwargs.pop("f_scale", 2.0)
        ftol = kwargs.pop("ftol", 1e-8)
        xtol = kwargs.pop("xtol", 1e-8)
        gtol = kwargs.pop("gtol", 1e-8)
        max_nfev = kwargs.pop("max_nfev", 500)

        jac_sparsity = None
        x_scale = kwargs.pop("x_scale", "jac" if use_sparse_jac else 1.0)
        if use_sparse_jac and hasattr(self, "_jac_sparsity"):
            jac_sparsity = self._jac_sparsity()

        res = least_squares(
            self._residual_func,
            x0,
            bounds=bounds,
            method="trf",
            loss=loss,
            f_scale=f_scale,
            ftol=ftol,
            xtol=xtol,
            gtol=gtol,
            max_nfev=max_nfev,
            jac_sparsity=jac_sparsity,
            x_scale=x_scale,  # ✅ 关键修复点：自动平衡内参和外参的梯度
            verbose=verbose,
            **kwargs
        )

        self._unpack_params(res.x)

        rmse = float(np.sqrt(np.mean(res.fun ** 2))) if res.fun.size else float("nan")
        print(f"✅ 优化完成! 平均重投影误差 (RMSE): {rmse:.4f} px")
        print(f"nfev={res.nfev}, njev={res.njev}, cost={res.cost:.4e}, status={res.status}")

        return res