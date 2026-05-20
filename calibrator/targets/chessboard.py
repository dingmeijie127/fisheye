from ..core.base_target import BaseTargetDetector
from ..utils.utils import det_chessboard_corners, gen_object_points

class ChessboardDetector(BaseTargetDetector):
    def __init__(self, chessboard_size=(11, 8), square_size=25.0):
        self.chessboard_size = chessboard_size
        self.square_size = square_size
        self.obj_pts = gen_object_points(chessboard_size, square_size)

    def detect(self, image_path: str):
        # 调用你的 utils 函数
        corners, _, _ = det_chessboard_corners(image_path, self.chessboard_size)
        if corners is not None:
            # 统一返回 ID=0 的字典格式
            return {0: (corners.reshape(-1, 2), self.obj_pts)}
        return {}