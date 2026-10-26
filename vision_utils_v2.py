#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视觉定位系统工具类 - 改进版
包含数据结构、算法实现、相机标定等核心功能
"""

import cv2
import numpy as np
import math
import json
import pickle
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any


@dataclass
class CalibrationData:
    """
    相机标定数据
    
    标定原理说明：
    1. 针孔相机模型：相机成像遵循针孔模型，需要求解内参矩阵
    2. 内参矩阵包含：焦距(fx, fy)、主点(cx, cy)、畸变系数
    3. 外参矩阵包含：旋转矩阵(R)和平移向量(t)
    4. 标定流程：通过多个视角的棋盘格图像，求解相机内外参数
    
    标定步骤：
    1. 准备标定板（棋盘格），已知角点物理坐标
    2. 从不同角度拍摄多张标定板图像（至少10张）
    3. 检测每张图像中的角点像素坐标
    4. 建立物理坐标与像素坐标的对应关系
    5. 求解相机内参矩阵和畸变系数
    6. 计算重投影误差评估标定精度
    """
    camera_matrix: np.ndarray = field(default_factory=lambda: np.eye(3))
    dist_coeffs: np.ndarray = field(default_factory=lambda: np.zeros((5, 1)))
    pixel_per_mm_x: float = 1.0
    pixel_per_mm_y: float = 1.0
    offset_x: float = 0.0
    offset_y: float = 0.0
    points_pixel: List[Tuple[float, float]] = field(default_factory=list)
    points_physical: List[Tuple[float, float]] = field(default_factory=list)
    is_calibrated: bool = False
    reprojection_error: float = 0.0
    image_width: int = 1280
    image_height: int = 720


@dataclass
class DualLineTemplate:
    """
    双直线模板（每侧两条垂直线）
    用于矩形面纸的精确定位
    
    结构说明：
    - 每侧有两条互相垂直的直线（长边和宽边）
    - 通过两条直线的交点确定角点位置
    - 通过直线角度确定面纸方向
    """
    left_line_long: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
    left_line_short: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
    right_line_long: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
    right_line_short: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
    left_roi: Optional[Tuple[float, float, float, float]] = None
    right_roi: Optional[Tuple[float, float, float, float]] = None
    valid: bool = False
    
    def is_complete(self) -> bool:
        """检查模板是否完整"""
        return (self.left_line_long is not None and 
                self.left_line_short is not None and
                self.right_line_long is not None and 
                self.right_line_short is not None and
                self.left_roi is not None and 
                self.right_roi is not None)
    
    def has_lines(self) -> bool:
        """检查是否已设置所有直线"""
        return (self.left_line_long is not None and 
                self.left_line_short is not None and
                self.right_line_long is not None and 
                self.right_line_short is not None)


@dataclass
class ShapeTemplate:
    """
    异形轮廓模板
    用于不规则形状的识别
    """
    left_contour: Optional[np.ndarray] = None
    right_contour: Optional[np.ndarray] = None
    left_roi: Optional[Tuple[float, float, float, float]] = None
    right_roi: Optional[Tuple[float, float, float, float]] = None
    center: Tuple[float, float] = (0, 0)
    angle: float = 0.0
    area: float = 0.0
    name: str = ""
    valid: bool = False
    
    def is_complete(self) -> bool:
        """检查模板是否完整"""
        return (self.left_contour is not None and 
                self.right_contour is not None and
                self.left_roi is not None and 
                self.right_roi is not None)
    
    def has_contours(self) -> bool:
        """检查是否已设置所有轮廓"""
        return self.left_contour is not None and self.right_contour is not None


@dataclass
class DetectionResult:
    x_pixel: float = 0.0
    y_pixel: float = 0.0
    x_mm: float = 0.0
    y_mm: float = 0.0
    z_mm: float = 0.0
    angle: float = 0.0
    score: float = 0.0
    valid: bool = False
    timestamp: float = 0.0


class CameraCalibration:
    """
    相机标定类
    
    标定原理：
    ==========
    
    1. 针孔相机模型
       相机成像可以建模为针孔模型，3D空间点通过投影变换映射到2D图像平面。
       
    2. 内参矩阵 (Intrinsic Matrix)
       K = | fx  0  cx |
           |  0 fy  cy |
           |  0  0   1 |
       
       - fx, fy: 焦距（像素单位）
       - cx, cy: 主点坐标（图像中心）
       
    3. 畸变模型 (Distortion Model)
       径向畸变：k1, k2, k3, k4, k5, k6
       切向畸变：p1, p2
       
       畸变校正公式：
       x_distorted = x(1 + k1*r² + k2*r⁴ + k3*r⁶) / (1 + k4*r² + k5*r⁴ + k6*r⁶) + 2*p1*x*y + p2*(r² + 2*x²)
       y_distorted = y(1 + k1*r² + k2*r⁴ + k3*r⁶) / (1 + k4*r² + k5*r⁴ + k6*r⁶) + p1*(r² + 2*y²) + 2*p2*x*y
       
    4. 外参矩阵 (Extrinsic Matrix)
       [R|t] = | r11 r12 r13 tx |
              | r21 r22 r23 ty |
              | r31 r32 r33 tz |
       
       - R: 旋转矩阵（3x3）
       - t: 平移向量（3x1）
       
    5. 投影方程
       s * [u, v, 1]ᵀ = K * [R|t] * [X, Y, Z, 1]ᵀ
       
    标定流程：
    ==========
    
    1. 准备标定板
       - 使用棋盘格标定板（推荐9x6或10x7内角点）
       - 打印精度要求高，确保平整
       
    2. 采集图像
       - 从不同角度、距离拍摄至少15-20张图像
       - 覆盖整个视场，包括边缘区域
       - 确保图像清晰，无模糊
       
    3. 检测角点
       - 使用findChessboardCorners检测角点
       - 使用cornerSubPix亚像素精度优化
       
    4. 求解参数
       - 使用calibrateCamera求解内参和畸变系数
       - 使用solvePnP求解外参
       
    5. 评估精度
       - 计算重投影误差（RMS）
       - 误差应小于0.5像素为佳
       
    6. 应用校正
       - 使用initUndistortRectifyMap生成映射
       - 使用remap进行图像校正
    """
    
    @staticmethod
    def detect_checkerboard(image: np.ndarray, pattern_size: Tuple[int, int]) -> Tuple[bool, np.ndarray]:
        """
        检测棋盘格角点
        
        参数：
            image: 输入图像
            pattern_size: 棋盘格内角点尺寸 (cols, rows)
            
        返回：
            success: 是否检测成功
            corners: 角点坐标 (N, 1, 2)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        
        if ret:
            # 亚像素精度优化
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
        return ret, corners
    
    @staticmethod
    def calibrate_camera(images: List[np.ndarray], pattern_size: Tuple[int, int], 
                         square_size: float = 1.0) -> Tuple[CalibrationData, float]:
        """
        相机标定
        
        参数：
            images: 标定图像列表
            pattern_size: 棋盘格内角点尺寸 (cols, rows)
            square_size: 棋盘格方格物理尺寸 (mm)
            
        返回：
            calib_data: 标定数据
            rms_error: 重投影误差
        """
        # 准备物体点坐标
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        objp *= square_size
        
        objpoints = []  # 3D点
        imgpoints = []  # 2D点
        image_size = None
        
        for img in images:
            ret, corners = CameraCalibration.detect_checkerboard(img, pattern_size)
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)
                if image_size is None:
                    image_size = (img.shape[1], img.shape[0])
        
        if len(objpoints) < 3:
            raise ValueError("需要至少3张有效标定图像")
        
        # 执行标定
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, image_size, None, None
        )
        
        # 计算重投影误差
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], 
                                              camera_matrix, dist_coeffs)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error
        mean_error /= len(objpoints)
        
        # 创建标定数据
        calib_data = CalibrationData()
        calib_data.camera_matrix = camera_matrix
        calib_data.dist_coeffs = dist_coeffs
        calib_data.is_calibrated = True
        calib_data.reprojection_error = mean_error
        calib_data.image_width = image_size[0]
        calib_data.image_height = image_size[1]
        
        # 计算像素/毫米比例（基于焦距）
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        # 假设工作距离为500mm
        work_distance = 500.0
        calib_data.pixel_per_mm_x = fx / work_distance
        calib_data.pixel_per_mm_y = fy / work_distance
        
        return calib_data, mean_error
    
    @staticmethod
    def calibrate_from_points(pixel_points: List[Tuple[float, float]], 
                             physical_points: List[Tuple[float, float]]) -> CalibrationData:
        """
        通过点对应进行简化标定
        
        参数：
            pixel_points: 像素坐标列表
            physical_points: 物理坐标列表 (mm)
            
        返回：
            calib_data: 标定数据
        """
        if len(pixel_points) < 2:
            raise ValueError("需要至少2个标定点")
        
        calib_data = CalibrationData()
        
        # 计算像素/毫米比例
        px_list = np.array(pixel_points, dtype=float)
        phys_list = np.array(physical_points, dtype=float)
        
        # 逐轴比例：遍历所有点对，取"该轴物理跨度最大"的点对做估计，
        # 避免仅用前两个点时（点对同轴排列）另一轴跨度为 0 导致比例退化为默认值
        def _axis_ratio(axis: int) -> Optional[float]:
            best_span = 0.0
            best_ratio = None
            n = len(px_list)
            for i in range(n):
                for j in range(i + 1, n):
                    d_phys = abs(phys_list[j][axis] - phys_list[i][axis])
                    if d_phys <= 1e-9:
                        continue
                    d_pix = abs(px_list[j][axis] - px_list[i][axis])
                    ratio = d_pix / d_phys
                    if d_phys > best_span:
                        best_span = d_phys
                        best_ratio = ratio
            return best_ratio
        
        rx = _axis_ratio(0)
        ry = _axis_ratio(1)
        if rx is not None:
            calib_data.pixel_per_mm_x = rx
        if ry is not None:
            calib_data.pixel_per_mm_y = ry
        # 若某轴无法估计（所有点在该轴无跨度），退化为另一轴比例，
        # 避免出现 1:1 默认值造成的量纲错误
        if rx is None and ry is not None:
            calib_data.pixel_per_mm_x = ry
        if ry is None and rx is not None:
            calib_data.pixel_per_mm_y = rx
        
        calib_data.points_pixel = pixel_points
        calib_data.points_physical = physical_points
        calib_data.is_calibrated = True
        
        return calib_data
    
    @staticmethod
    def undistort_image(image: np.ndarray, calib_data: CalibrationData) -> np.ndarray:
        """
        图像去畸变
        
        参数：
            image: 输入图像
            calib_data: 标定数据
            
        返回：
            undistorted: 去畸变后的图像
        """
        if not calib_data.is_calibrated:
            return image
            
        h, w = image.shape[:2]
        map1, map2 = cv2.initUndistortRectifyMap(
            calib_data.camera_matrix, calib_data.dist_coeffs, None,
            calib_data.camera_matrix, (w, h), cv2.CV_16SC2
        )
        undistorted = cv2.remap(image, map1, map2, cv2.INTER_LINEAR)
        
        return undistorted


class VisionAlgorithm:
    """视觉算法类"""
    
    @staticmethod
    def detect_lines_in_roi(frame: np.ndarray, roi: Tuple[float, float, float, float],
                           canny_low=50, canny_high=150, hough_threshold=50) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        在指定ROI区域内检测直线
        
        参数：
            frame: 输入图像
            roi: 区域 (x1, y1, x2, y2)
            canny_low: Canny低阈值
            canny_high: Canny高阈值
            hough_threshold: Hough变换阈值
            
        返回：
            lines: 检测到的直线列表 [((x1,y1), (x2,y2)), ...]
        """
        x1, y1, x2, y2 = [int(v) for v in roi]
        roi_img = frame[y1:y2, x1:x2]
        
        if roi_img.size == 0:
            return []
        
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY) if len(roi_img.shape) == 3 else roi_img
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
        edges = cv2.Canny(blurred, canny_low, canny_high)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, hough_threshold, minLineLength=30, maxLineGap=10)
        
        detected_lines = []
        if lines is not None:
            for line in lines:
                px1, py1, px2, py2 = line[0]
                length = math.sqrt((px2-px1)**2 + (py2-py1)**2)
                if length > 30:
                    # 转换到原图坐标
                    detected_lines.append(((px1+x1, py1+y1), (px2+x1, py2+y1), length))
            detected_lines.sort(key=lambda x: x[2], reverse=True)
        
        return [(line[0], line[1]) for line in detected_lines]

    @staticmethod
    def detect_lines_canny_standard(gray: np.ndarray, offset_x: int, offset_y: int, 
                                   canny_low=50, canny_high=150) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """标准Canny边缘检测 + Hough直线检测"""
        lines = []
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
        edges = cv2.Canny(blurred, canny_low, canny_high)
        hough_lines = cv2.HoughLinesP(edges, 1, np.pi/180, 40, minLineLength=25, maxLineGap=10)
        if hough_lines is not None:
            for line in hough_lines:
                px1, py1, px2, py2 = line[0]
                length = math.sqrt((px2-px1)**2 + (py2-py1)**2)
                if length > 25:
                    lines.append(((px1+offset_x, py1+offset_y), (px2+offset_x, py2+offset_y)))
        return lines

    @staticmethod
    def detect_lines_canny_low(gray: np.ndarray, offset_x: int, offset_y: int,
                              canny_low=30, canny_high=100) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """低阈值Canny边缘检测（更敏感）"""
        lines = []
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
        edges = cv2.Canny(blurred, canny_low, canny_high)
        # 降低阈值，增加检测灵敏度
        hough_lines = cv2.HoughLinesP(edges, 1, np.pi/180, 20, minLineLength=15, maxLineGap=10)
        if hough_lines is not None:
            print(f"[Canny Low] 检测到 {len(hough_lines)} 条原始直线")
            for line in hough_lines:
                px1, py1, px2, py2 = line[0]
                length = math.sqrt((px2-px1)**2 + (py2-py1)**2)
                if length > 15:
                    lines.append(((px1+offset_x, py1+offset_y), (px2+offset_x, py2+offset_y)))
        print(f"[Canny Low] 过滤后剩余 {len(lines)} 条直线")
        return lines

    @staticmethod
    def detect_lines_sobel(gray: np.ndarray, offset_x: int, offset_y: int,
                          threshold=50) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Sobel边缘检测"""
        lines = []
        sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        sobel_edges = cv2.magnitude(sobelx, sobely)
        sobel_edges = cv2.normalize(sobel_edges, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        _, sobel_binary = cv2.threshold(sobel_edges, threshold, 255, cv2.THRESH_BINARY)
        hough_lines = cv2.HoughLinesP(sobel_binary, 1, np.pi/180, 35, minLineLength=25, maxLineGap=10)
        if hough_lines is not None:
            for line in hough_lines:
                px1, py1, px2, py2 = line[0]
                length = math.sqrt((px2-px1)**2 + (py2-py1)**2)
                if length > 25:
                    lines.append(((px1+offset_x, py1+offset_y), (px2+offset_x, py2+offset_y)))
        return lines

    @staticmethod
    def detect_lines_scharr(gray: np.ndarray, offset_x: int, offset_y: int,
                           threshold=40) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Scharr边缘检测（更精确）"""
        lines = []
        scharrx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
        scharry = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
        scharr_edges = cv2.magnitude(scharrx, scharry)
        scharr_edges = cv2.normalize(scharr_edges, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        _, scharr_binary = cv2.threshold(scharr_edges, threshold, 255, cv2.THRESH_BINARY)
        hough_lines = cv2.HoughLinesP(scharr_binary, 1, np.pi/180, 35, minLineLength=25, maxLineGap=10)
        if hough_lines is not None:
            for line in hough_lines:
                px1, py1, px2, py2 = line[0]
                length = math.sqrt((px2-px1)**2 + (py2-py1)**2)
                if length > 25:
                    lines.append(((px1+offset_x, py1+offset_y), (px2+offset_x, py2+offset_y)))
        return lines

    @staticmethod
    def detect_lines_single_algorithm(frame: np.ndarray, roi: Tuple[float, float, float, float],
                                     algorithm: str, template_lines=None, **kwargs) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """使用指定的单个算法检测直线"""
        x1, y1, x2, y2 = [int(v) for v in roi]
        roi_img = frame[y1:y2, x1:x2]
        
        if roi_img.size == 0:
            return []
        
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY) if len(roi_img.shape) == 3 else roi_img
        
        if algorithm == 'canny_standard':
            lines = VisionAlgorithm.detect_lines_canny_standard(gray, x1, y1,
                kwargs.get('canny_low', 50), kwargs.get('canny_high', 150))
        elif algorithm == 'canny_low':
            lines = VisionAlgorithm.detect_lines_canny_low(gray, x1, y1,
                kwargs.get('canny_low', 30), kwargs.get('canny_high', 100))
        elif algorithm == 'sobel':
            lines = VisionAlgorithm.detect_lines_sobel(gray, x1, y1, kwargs.get('threshold', 50))
        elif algorithm == 'scharr':
            lines = VisionAlgorithm.detect_lines_scharr(gray, x1, y1, kwargs.get('threshold', 40))
        elif algorithm == 'fusion':
            return VisionAlgorithm.detect_lines_multialgorithm(frame, roi, template_lines)
        else:
            return VisionAlgorithm.detect_lines_multialgorithm(frame, roi, template_lines)
        
        if template_lines and len(template_lines) > 0:
            template_angles = [VisionAlgorithm.calculate_line_angle(line) for line in template_lines]
            filtered_lines = []
            for line in lines:
                line_angle = VisionAlgorithm.calculate_line_angle(line)
                for template_angle in template_angles:
                    # 直线无方向，角度差需按模180归一化（垂直线 +90/-90 视为同向）
                    angle_diff = VisionAlgorithm.undirected_angle_diff(line_angle, template_angle)
                    if angle_diff < 20:
                        filtered_lines.append(line)
                        break
            return filtered_lines

        return lines

    @staticmethod
    def undirected_angle_diff(angle1: float, angle2: float) -> float:
        """
        计算两条无向直线的夹角（0~90度）。
        atan2 返回 -180~180，端点顺序会导致 +90/-90 跳变，
        因此按模 180 归一化后取较小角。
        """
        d = abs(angle1 - angle2) % 180.0
        return min(d, 180.0 - d)

    @staticmethod
    def merge_collinear_lines(lines: List, angle_tol: float = 10.0,
                              lateral_tol: float = 12.0,
                              min_length: float = 30.0) -> List:
        """
        将角度一致、横向位置接近的断裂线段合并为完整线段。
        用于噪声/遮挡工况下纸张边缘断裂后的重建，恢复正确的长度与中心。

        聚类准则：无向夹角 < angle_tol，且线段中心在法线方向距离 < lateral_tol。
        重建方法：簇内所有端点用 cv2.fitLine 拟合，再沿主方向投影取两端。
        """
        if not lines:
            return []

        items = []
        for line in lines:
            (x1, y1), (x2, y2) = line
            length = math.hypot(x2 - x1, y2 - y1)
            if length < min_length:
                continue
            ang = VisionAlgorithm.calculate_line_angle(line)
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            items.append({"line": line, "len": length, "ang": ang,
                          "cx": cx, "cy": cy, "used": False})

        merged = []
        for i, anchor in enumerate(items):
            if anchor["used"]:
                continue
            anchor["used"] = True
            cluster_pts = [anchor["line"][0], anchor["line"][1]]
            rad = math.radians(anchor["ang"])
            dx_n, dy_n = -math.sin(rad), math.cos(rad)  # 法线方向

            for j in range(i + 1, len(items)):
                other = items[j]
                if other["used"]:
                    continue
                if VisionAlgorithm.undirected_angle_diff(anchor["ang"], other["ang"]) > angle_tol:
                    continue
                # 中心在法线方向的距离
                lateral = abs((other["cx"] - anchor["cx"]) * dx_n +
                              (other["cy"] - anchor["cy"]) * dy_n)
                if lateral < lateral_tol:
                    other["used"] = True
                    cluster_pts.append(other["line"][0])
                    cluster_pts.append(other["line"][1])

            if len(cluster_pts) < 2:
                merged.append(anchor["line"])
                continue

            pts = np.array(cluster_pts, dtype=np.float32).reshape(-1, 1, 2)
            vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
            # 各点沿主方向的投影参数 t
            ts = [(p[0] - x0) * vx + (p[1] - y0) * vy for p in cluster_pts]
            t_min, t_max = min(ts), max(ts)
            p_a = (float(x0 + t_min * vx), float(y0 + t_min * vy))
            p_b = (float(x0 + t_max * vx), float(y0 + t_max * vy))
            merged.append((p_a, p_b))

        # 合并后按长度降序
        merged.sort(key=lambda ln: math.hypot(ln[1][0] - ln[0][0], ln[1][1] - ln[0][1]),
                    reverse=True)
        return merged

    @staticmethod
    def refine_line_with_edges(frame: np.ndarray, line,
                               roi: Optional[Tuple[float, float, float, float]] = None,
                               half_width: float = 6.0,
                               expected_length: Optional[float] = None,
                               canny_low: int = 25, canny_high: int = 75,
                               iterations: int = 2):
        """
        用目标直线周围"紧凑定向条带"内的 Canny 边缘点做稳健最小二乘拟合，精修直线。
        解决噪声/弱对比下 Hough 线段断裂、中心偏移、以及边缘外噪点拉偏中心的问题。

        采用"定向条带 + 最密长度区间"策略：
          - 法线方向仅取 ±half_width 内的点，保证只拟合同一条边；
          - 真实边缘是一段长度约 expected_length 的密集连续点，而噪声为稀疏孤立点；
            在沿线坐标上用双指针找出"覆盖点数最多、长度为 expected_length 的区间"，
            从而自动对准真实边缘、剔除越界噪点，不依赖初始 Hough 中心是否偏移；
          - 区间内点做 cv2.fitLine，端点取投影极值，并可迭代刷新方向。

        参数：
            frame: 原图（BGR 或灰度）
            line: 初始直线 ((x1,y1),(x2,y2))
            roi: 可选边界约束 (x1,y1,x2,y2)，搜索范围会被裁剪到该区域与图像内
            half_width: 法线方向纳入距离（像素）
            expected_length: 预期边缘长度（像素），通常传模板线长度
            iterations: 方向/区间刷新次数
        返回：
            精修后的直线（点数不足时返回原线）
        """
        try:
            (a1, a2), (b1, b2) = line
            vx, vy = b1 - a1, b2 - a2
            line_len = math.hypot(vx, vy)
            if line_len < 1e-6:
                return line
            ux, uy = vx / line_len, vy / line_len      # 沿线单位向量
            nx, ny = -uy, ux                            # 法线单位向量
            cx0, cy0 = (a1 + b1) / 2.0, (a2 + b2) / 2.0

            target_len = expected_length if expected_length else line_len
            # 候选点搜索范围：在预期半长外再留滑动余量，允许初始中心有偏差
            search_half = target_len / 2.0 + 35.0

            ex = abs(ux) * search_half + abs(nx) * half_width
            ey = abs(uy) * search_half + abs(ny) * half_width
            rx1, ry1 = cx0 - ex, cy0 - ey
            rx2, ry2 = cx0 + ex, cy0 + ey

            h_img, w_img = frame.shape[:2]
            rx1 = max(0, int(rx1)); ry1 = max(0, int(ry1))
            rx2 = min(w_img, int(rx2) + 1); ry2 = min(h_img, int(ry2) + 1)
            if roi is not None:
                bx1, by1, bx2, by2 = [int(v) for v in roi]
                rx1 = max(rx1, bx1); ry1 = max(ry1, by1)
                rx2 = min(rx2, bx2); ry2 = min(ry2, by2)
            if rx1 >= rx2 or ry1 >= ry2:
                return line

            roi_img = frame[ry1:ry2, rx1:rx2]
            gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY) if roi_img.ndim == 3 else roi_img
            edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 1.4), canny_low, canny_high)
            ys_all, xs_all = np.where(edges > 0)
            if len(xs_all) < 20:
                return line
            xs_all = xs_all.astype(np.float32) + rx1
            ys_all = ys_all.astype(np.float32) + ry1

            cur_cx, cur_cy = cx0, cy0
            cur_ux, cur_uy, cur_nx, cur_ny = ux, uy, nx, ny
            best = line

            def _fit(cxc, cyc, dxu, dyu, dense_only):
                """以中心/方向取法线条带点；dense_only=True 用最密长度区间粗定位，
                False 时放宽沿向并用分位数端点恢复完整边缘。"""
                dxu2, dyu2 = dxu, dyu
                nxv, nyv = -dyu2, dxu2
                dxv = xs_all - cxc
                dyv = ys_all - cyc
                lat = dxv * nxv + dyv * nyv
                band = np.abs(lat) < half_width
                if dense_only:
                    pvx, pvy = xs_all[band], ys_all[band]
                    if len(pvx) < 20:
                        return None
                    tv = (pvx - cxc) * dxu2 + (pvy - cyc) * dyu2
                    order = np.argsort(tv)
                    tvs = tv[order]
                    jj, bc, bi, bj = 0, -1, 0, 0
                    nn = len(tvs)
                    for ii in range(nn):
                        if jj < ii:
                            jj = ii
                        while jj < nn and tvs[jj] - tvs[ii] <= target_len:
                            jj += 1
                        if jj - ii > bc:
                            bc, bi, bj = jj - ii, ii, jj - 1
                    idx = order[bi:bj + 1]
                    qx, qy = xs_all[idx], ys_all[idx]
                else:
                    al = dxv * dxu2 + dyv * dyu2
                    m2 = band & (np.abs(al) < target_len / 2.0 + 10.0)
                    qx, qy = xs_all[m2], ys_all[m2]
                if len(qx) < 20:
                    return None
                p = np.column_stack([qx, qy]).reshape(-1, 1, 2).astype(np.float32)
                v2x, v2y, ox, oy = cv2.fitLine(p, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
                if v2x * ux + v2y * uy < 0:
                    v2x, v2y = -v2x, -v2y
                tp2 = (qx - ox) * v2x + (qy - oy) * v2y
                if dense_only:
                    lo, hi = float(tp2.min()), float(tp2.max())
                else:
                    lo = float(np.quantile(tp2, 0.015))
                    hi = float(np.quantile(tp2, 0.985))
                return (v2x, v2y,
                        (float(ox + lo * v2x), float(oy + lo * v2y)),
                        (float(ox + hi * v2x), float(oy + hi * v2y)))

            # 第一轮：最密长度区间稳健定位中心/方向（抗边缘外噪点）
            r1 = _fit(cur_cx, cur_cy, cur_ux, cur_uy, True)
            if r1 is None:
                return line
            vx2, vy2, p_lo, p_hi = r1
            cur_ux, cur_uy = float(vx2), float(vy2)
            cur_cx = (p_lo[0] + p_hi[0]) / 2.0
            cur_cy = (p_lo[1] + p_hi[1]) / 2.0
            best = (p_lo, p_hi)
            # 第二轮：以稳健中心对称开窗，放宽沿向 + 分位数端点恢复完整跨度
            r2 = _fit(cur_cx, cur_cy, cur_ux, cur_uy, False)
            if r2 is not None:
                _, _, p_lo2, p_hi2 = r2
                best = (p_lo2, p_hi2)
            return best
        except Exception:
            return line

    @staticmethod
    def _edge_support_density(frame: np.ndarray, line,
                              roi: Optional[Tuple[float, float, float, float]] = None,
                              band: float = 4.0) -> float:
        """
        统计一条直线周围窄条带内的 Canny 边缘点密度（点/像素²量级），
        作为该线是否为"真实连续边缘"的支持度。真实边缘点密集连续 → 高密度；
        孤立噪声/误检线 → 低密度。
        """
        try:
            (a1, a2), (b1, b2) = line
            vx, vy = b1 - a1, b2 - a2
            ln = math.hypot(vx, vy)
            if ln < 1e-6:
                return 0.0
            ux, uy = vx / ln, vy / ln
            nx, ny = -uy, ux
            cx, cy = (a1 + b1) / 2.0, (a2 + b2) / 2.0
            along_half = ln / 2.0 + 6.0
            ex = abs(ux) * along_half + abs(nx) * band
            ey = abs(uy) * along_half + abs(ny) * band
            h_img, w_img = frame.shape[:2]
            rx1 = max(0, int(cx - ex)); ry1 = max(0, int(cy - ey))
            rx2 = min(w_img, int(cx + ex) + 1); ry2 = min(h_img, int(cy + ey) + 1)
            if roi is not None:
                bx1, by1, bx2, by2 = [int(v) for v in roi]
                rx1 = max(rx1, bx1); ry1 = max(ry1, by1)
                rx2 = min(rx2, bx2); ry2 = min(ry2, by2)
            if rx1 >= rx2 or ry1 >= ry2:
                return 0.0
            sub = frame[ry1:ry2, rx1:rx2]
            gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY) if sub.ndim == 3 else sub
            edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 1.4), 25, 75)
            ys, xs = np.where(edges > 0)
            if len(xs) == 0:
                return 0.0
            xs = xs.astype(np.float32) + rx1
            ys = ys.astype(np.float32) + ry1
            dx, dy = xs - cx, ys - cy
            lat = dx * nx + dy * ny
            along = dx * ux + dy * uy
            inside = (np.abs(lat) < band) & (np.abs(along) < along_half)
            cnt = int(np.count_nonzero(inside))
            return cnt / max(1.0, (2.0 * band) * (2.0 * along_half))
        except Exception:
            return 0.0

    @staticmethod
    def refine_best_candidate(frame: np.ndarray, raw_lines, template_line,
                              roi: Optional[Tuple[float, float, float, float]] = None,
                              angle_tol: float = 25.0, half_width: float = 6.0,
                              expected_length: Optional[float] = None,
                              max_candidates: int = 6,
                              pos_normal: float = 75.0, pos_along: float = 95.0):
        """
        从 Hough 候选线中选出真实边缘并精修。
        强噪声叠加旋转时，角度/距离评分的首选线可能是噪声线段；而真实边缘的
        本质特征是"沿线有密集连续的 Canny 边缘点支持"。因此：
          1) 保留与模板无向夹角 ≤ angle_tol、且中心相对模板位置在预期工作范围
             （法线 ≤ pos_normal、沿线 ≤ pos_along，覆盖 ±50px 偏移 + ±5° 旋转）内的候选，
             以排除纸张对侧边、遮挡块/干扰线等远处同方向结构；
          2) 逐条做边缘点精修；
          3) 用精修线周围窄条带的边缘点密度投票，返回支持最强者。
        返回 (精修线, 支持密度)；无合格候选返回 (None, 0.0)。
        """
        if not raw_lines or template_line is None:
            return None, 0.0
        (ta1, ta2), (tb1, tb2) = template_line
        tvx, tvy = tb1 - ta1, tb2 - ta2
        tln = math.hypot(tvx, tvy)
        if tln < 1e-6:
            return None, 0.0
        tux, tuy = tvx / tln, tvy / tln
        tnx, tny = -tuy, tux
        tcx, tcy = (ta1 + tb1) / 2.0, (ta2 + tb2) / 2.0
        tpl_ang = VisionAlgorithm.calculate_line_angle(template_line)
        cands = []
        for ln in raw_lines:
            if VisionAlgorithm.undirected_angle_diff(
                    VisionAlgorithm.calculate_line_angle(ln), tpl_ang) > angle_tol:
                continue
            ccx, ccy = VisionAlgorithm.calculate_line_center(ln)
            dcx, dcy = ccx - tcx, ccy - tcy
            if abs(dcx * tnx + dcy * tny) > pos_normal:
                continue
            if abs(dcx * tux + dcy * tuy) > pos_along:
                continue
            cands.append(ln)
        if not cands:
            return None, 0.0
        cands.sort(key=lambda ln: math.hypot(ln[1][0] - ln[0][0],
                                             ln[1][1] - ln[0][1]), reverse=True)
        best_line, best_score = None, -1.0
        target_len = expected_length if expected_length else None
        for ln in cands[:max_candidates]:
            # 原始候选（可能已是多段拟合的准确合并线）与精修结果分别评分，
            # 综合"边缘点支持密度"与"长度契合度"：仅当精修未明显降低支持度、
            # 且长度不更差时才采用，否则回退原始线，防止精修漂移到噪声结构
            score_raw = VisionAlgorithm._edge_support_density(frame, ln, roi, band=4.0)
            refined = VisionAlgorithm.refine_line_with_edges(
                frame, ln, roi, half_width=half_width, expected_length=expected_length)
            score_ref = VisionAlgorithm._edge_support_density(frame, refined, roi, band=4.0)

            def _len_fit(q):
                if not target_len:
                    return 0.0
                ql = math.hypot(q[1][0] - q[0][0], q[1][1] - q[0][1])
                return abs(ql - target_len)

            # 支持度 = 密度（主导）；长度用于精修/原始的采纳裁决
            if score_ref >= score_raw or (
                    score_ref >= score_raw * 0.85
                    and _len_fit(refined) <= _len_fit(ln) + 1e-6):
                cand_line, cand_score = refined, score_ref
            else:
                cand_line, cand_score = ln, score_raw
            if cand_score > best_score:
                best_score, best_line = cand_score, cand_line
        return best_line, float(best_score)

    @staticmethod
    def get_available_algorithms() -> dict:
        """获取可用的检测算法列表"""
        return {
            'canny_standard': {'name': '标准Canny', 'description': '适用于对比度较好的场景'},
            'canny_low': {'name': '低阈值Canny', 'description': '更敏感，适用于低对比度场景'},
            'sobel': {'name': 'Sobel检测', 'description': '抗噪声能力较强'},
            'scharr': {'name': 'Scharr检测', 'description': '更精确的边缘定位'},
            'fusion': {'name': '算法融合', 'description': '融合所有算法（推荐）'}
        }

    @staticmethod
    def detect_lines_multialgorithm(frame: np.ndarray, roi: Tuple[float, float, float, float],
                                   template_lines=None) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        使用多种边缘检测算法融合来检测直线，提高检测精度
        
        参数：
            frame: 输入图像
            roi: 区域 (x1, y1, x2, y2)
            template_lines: 模板直线列表（用于方向过滤）
            
        返回：
            lines: 检测到的直线列表 [((x1,y1), (x2,y2)), ...]
        """
        try:
            x1, y1, x2, y2 = [int(v) for v in roi]
            
            # 检查ROI是否有效
            if x1 >= x2 or y1 >= y2:
                return []
            
            # 检查边界
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            
            if x1 >= x2 or y1 >= y2:
                return []
            
            roi_img = frame[y1:y2, x1:x2]
            
            if roi_img.size == 0:
                return []
            
            gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY) if len(roi_img.shape) == 3 else roi_img
            
            all_lines = []
            line_signatures = set()
            
            # 算法1: 标准Canny + Hough
            blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
            edges1 = cv2.Canny(blurred, 50, 150)
            lines1 = cv2.HoughLinesP(edges1, 1, np.pi/180, 40, minLineLength=25, maxLineGap=10)
            if lines1 is not None:
                for line in lines1:
                    px1, py1, px2, py2 = line[0]
                    length = math.sqrt((px2-px1)**2 + (py2-py1)**2)
                    if length > 25:
                        sig = (tuple(sorted([(px1+x1, py1+y1), (px2+x1, py2+y1)])))
                        if sig not in line_signatures:
                            line_signatures.add(sig)
                            all_lines.append(((px1+x1, py1+y1), (px2+x1, py2+y1), length))
            
            # 算法2: 低阈值Canny（更敏感）
            edges2 = cv2.Canny(blurred, 30, 100)
            lines2 = cv2.HoughLinesP(edges2, 1, np.pi/180, 35, minLineLength=20, maxLineGap=8)
            if lines2 is not None:
                for line in lines2:
                    px1, py1, px2, py2 = line[0]
                    length = math.sqrt((px2-px1)**2 + (py2-py1)**2)
                    if length > 20:
                        sig = (tuple(sorted([(px1+x1, py1+y1), (px2+x1, py2+y1)])))
                        if sig not in line_signatures:
                            line_signatures.add(sig)
                            all_lines.append(((px1+x1, py1+y1), (px2+x1, py2+y1), length))
            
            # 算法3: Sobel边缘检测
            sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            sobel_edges = cv2.magnitude(sobelx, sobely)
            sobel_edges = cv2.normalize(sobel_edges, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            _, sobel_binary = cv2.threshold(sobel_edges, 50, 255, cv2.THRESH_BINARY)
            lines3 = cv2.HoughLinesP(sobel_binary, 1, np.pi/180, 35, minLineLength=25, maxLineGap=10)
            if lines3 is not None:
                for line in lines3:
                    px1, py1, px2, py2 = line[0]
                    length = math.sqrt((px2-px1)**2 + (py2-py1)**2)
                    if length > 25:
                        sig = (tuple(sorted([(px1+x1, py1+y1), (px2+x1, py2+y1)])))
                        if sig not in line_signatures:
                            line_signatures.add(sig)
                            all_lines.append(((px1+x1, py1+y1), (px2+x1, py2+y1), length))
            
            # 算法4: Scharr边缘检测（更精确）
            scharrx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
            scharry = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
            scharr_edges = cv2.magnitude(scharrx, scharry)
            scharr_edges = cv2.normalize(scharr_edges, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            _, scharr_binary = cv2.threshold(scharr_edges, 40, 255, cv2.THRESH_BINARY)
            lines4 = cv2.HoughLinesP(scharr_binary, 1, np.pi/180, 35, minLineLength=25, maxLineGap=10)
            if lines4 is not None:
                for line in lines4:
                    px1, py1, px2, py2 = line[0]
                    length = math.sqrt((px2-px1)**2 + (py2-py1)**2)
                    if length > 25:
                        sig = (tuple(sorted([(px1+x1, py1+y1), (px2+x1, py2+y1)])))
                        if sig not in line_signatures:
                            line_signatures.add(sig)
                            all_lines.append(((px1+x1, py1+y1), (px2+x1, py2+y1), length))
            
            # 按长度排序
            all_lines.sort(key=lambda x: x[2], reverse=True)
            
            # 如果有模板直线，过滤出与模板角度相近的直线
            if template_lines and len(template_lines) > 0:
                template_angles = [VisionAlgorithm.calculate_line_angle(line) for line in template_lines]
                filtered_lines = []
                for line in all_lines:
                    line_angle = VisionAlgorithm.calculate_line_angle((line[0], line[1]))
                    for template_angle in template_angles:
                        # 直线无方向：必须按模180归一化夹角判断，
                        # 否则竖直线 atan2=-90 与模板 +90 相差 180° 被误杀
                        angle_diff = VisionAlgorithm.undirected_angle_diff(line_angle, template_angle)
                        if angle_diff < 20:
                            filtered_lines.append((line[0], line[1]))
                            break
                return filtered_lines
            
            return [(line[0], line[1]) for line in all_lines]
        
        except Exception as e:
            print(f"detect_lines_multialgorithm error: {e}")
            import traceback
            traceback.print_exc()
            return []

    @staticmethod
    def find_best_matching_lines(detected_lines: List[Tuple[Tuple[int, int], Tuple[int, int]]],
                                template_lines: List[Tuple[Tuple[int, int], Tuple[int, int]]]) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        在检测到的直线中找到与模板最匹配的直线（改进版，更鲁棒）
        
        参数：
            detected_lines: 检测到的直线列表
            template_lines: 模板直线列表
            
        返回：
            matched_lines: 匹配的直线列表
        """
        if not detected_lines or not template_lines:
            return detected_lines[:len(template_lines)]
        
        matched_lines = []
        
        for template_line in template_lines:
            template_angle = VisionAlgorithm.calculate_line_angle(template_line)
            template_length = math.sqrt((template_line[1][0] - template_line[0][0])**2 + 
                                       (template_line[1][1] - template_line[0][1])**2)
            template_center = ((template_line[0][0] + template_line[1][0]) / 2,
                              (template_line[0][1] + template_line[1][1]) / 2)
            
            best_match = None
            best_score = float('inf')
            
            for detected_line in detected_lines:
                if detected_line in matched_lines:
                    continue
                
                detected_angle = VisionAlgorithm.calculate_line_angle(detected_line)
                detected_length = math.sqrt((detected_line[1][0] - detected_line[0][0])**2 + 
                                           (detected_line[1][1] - detected_line[0][1])**2)
                detected_center = ((detected_line[0][0] + detected_line[1][0]) / 2,
                                   (detected_line[0][1] + detected_line[1][1]) / 2)
                
                # 角度差异（无向直线，按模180归一化，容忍 +90/-90 端点顺序跳变）
                angle_diff = VisionAlgorithm.undirected_angle_diff(detected_angle, template_angle)
                
                # 长度差异（归一化）
                length_diff = abs(detected_length - template_length) / max(template_length, 1)
                
                # 中心点距离（归一化）
                dist = math.sqrt((detected_center[0] - template_center[0])**2 +
                                (detected_center[1] - template_center[1])**2)
                
                # 综合评分（角度权重更高，允许更大的位置变化）
                score = angle_diff * 0.5 + length_diff * 0.3 + dist * 0.01
                
                if score < best_score:
                    best_score = score
                    best_match = detected_line
            
            if best_match is not None and best_score < 50:
                matched_lines.append(best_match)
        
        # 如果匹配结果不足，补充检测到的直线
        if len(matched_lines) < len(template_lines):
            for line in detected_lines:
                if line not in matched_lines:
                    matched_lines.append(line)
                    if len(matched_lines) >= len(template_lines):
                        break
        
        return matched_lines

    @staticmethod
    def calculate_line_distance(line1: Tuple[Tuple[float, float], Tuple[float, float]],
                                line2: Tuple[Tuple[float, float], Tuple[float, float]]) -> float:
        """计算两条直线之间的距离"""
        (x1, y1), (x2, y2) = line1
        (x3, y3), (x4, y4) = line2
        
        # 计算line1上的点到line2的平均距离
        dist1 = abs((y4-y3)*x1 - (x4-x3)*y1 + x4*y3 - y4*x3) / math.sqrt((y4-y3)**2 + (x4-x3)**2)
        dist2 = abs((y4-y3)*x2 - (x4-x3)*y2 + x4*y3 - y4*x3) / math.sqrt((y4-y3)**2 + (x4-x3)**2)
        
        return (dist1 + dist2) / 2

    @staticmethod
    def detect_contours_in_roi(frame: np.ndarray, roi: Tuple[float, float, float, float],
                               canny_low=50, canny_high=150) -> List[np.ndarray]:
        """
        在指定ROI区域内检测轮廓

        参数：
            frame: 输入图像
            roi: 区域 (x1, y1, x2, y2)
            canny_low: Canny低阈值
            canny_high: Canny高阈值

        返回：
            contours: 检测到的轮廓列表
        """
        x1, y1, x2, y2 = [int(v) for v in roi]
        roi_img = frame[y1:y2, x1:x2]

        if roi_img.size == 0:
            return []

        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY) if len(roi_img.shape) == 3 else roi_img
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
        edges = cv2.Canny(blurred, canny_low, canny_high)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 转换到原图坐标
        adjusted_contours = []
        for contour in contours:
            adjusted = contour.copy()
            adjusted[:, 0, 0] += x1
            adjusted[:, 0, 1] += y1
            adjusted_contours.append(adjusted)

        return adjusted_contours

    @staticmethod
    def _get_contour_effective_area(contour: np.ndarray) -> float:
        """获取轮廓的有效面积（处理开放式轮廓）"""
        area = cv2.contourArea(contour)
        if area > 0:
            return area
        # 对于开放式轮廓（折线），使用边界框面积
        x, y, w, h = cv2.boundingRect(contour)
        return w * h

    def detect_contours_multialgorithm(frame: np.ndarray, roi: Tuple[float, float, float, float],
                                      template_contour=None) -> List[Tuple[np.ndarray, float]]:
        """
        使用多种算法融合检测轮廓

        参数：
            frame: 输入图像
            roi: 区域 (x1, y1, x2, y2)
            template_contour: 模板轮廓（用于尺寸过滤）

        返回：
            contours: 检测到的轮廓列表，每个元素包含(轮廓, 面积)
        """
        x1, y1, x2, y2 = [int(v) for v in roi]
        roi_img = frame[y1:y2, x1:x2]

        if roi_img.size == 0:
            return []

        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY) if len(roi_img.shape) == 3 else roi_img

        all_contours = []
        contour_areas = {}

        # 算法1: 标准Canny
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
        edges1 = cv2.Canny(blurred, 50, 150)
        contours1, _ = cv2.findContours(edges1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours1:
            area = VisionAlgorithm._get_contour_effective_area(contour)
            if area > 100:
                adjusted = contour.copy()
                adjusted[:, 0, 0] += x1
                adjusted[:, 0, 1] += y1
                key = len(contour)
                if key not in contour_areas or contour_areas[key] < area:
                    contour_areas[key] = area
                    all_contours.append((adjusted, area))

        # 算法2: 低阈值Canny
        edges2 = cv2.Canny(blurred, 30, 100)
        contours2, _ = cv2.findContours(edges2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours2:
            area = VisionAlgorithm._get_contour_effective_area(contour)
            if area > 100:
                adjusted = contour.copy()
                adjusted[:, 0, 0] += x1
                adjusted[:, 0, 1] += y1
                key = len(contour)
                if key not in contour_areas or contour_areas[key] < area:
                    contour_areas[key] = area
                    all_contours.append((adjusted, area))

        # 算法3: Otsu自动阈值
        _, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours3, _ = cv2.findContours(binary_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours3:
            area = VisionAlgorithm._get_contour_effective_area(contour)
            if area > 100:
                adjusted = contour.copy()
                adjusted[:, 0, 0] += x1
                adjusted[:, 0, 1] += y1
                key = len(contour)
                if key not in contour_areas or contour_areas[key] < area:
                    contour_areas[key] = area
                    all_contours.append((adjusted, area))

        # 如果有模板轮廓，按尺寸过滤
        if template_contour is not None:
            template_area = VisionAlgorithm._get_contour_effective_area(template_contour)
            if template_area > 0:
                min_area = template_area * 0.5
                max_area = template_area * 2.0
                filtered = [(c, a) for c, a in all_contours if min_area <= a <= max_area]
                if filtered:
                    all_contours = filtered

        # 按面积排序
        all_contours.sort(key=lambda x: x[1], reverse=True)

        return all_contours

    @staticmethod
    def match_contours_by_position_size(template_contour: np.ndarray,
                                       detected_contours: List[np.ndarray]) -> Tuple[Optional[np.ndarray], float, Tuple[float, float]]:
        """
        根据位置和尺寸匹配轮廓（支持开放式轮廓）

        参数：
            template_contour: 模板轮廓
            detected_contours: 检测到的轮廓列表

        返回：
            (最佳匹配轮廓, 匹配分数, 中心点偏移)
        """
        if template_contour is None or not detected_contours:
            return None, 0.0, (0.0, 0.0)

        # 使用有效面积（处理开放式轮廓）
        template_area = VisionAlgorithm._get_contour_effective_area(template_contour)
        template_center = VisionAlgorithm._get_contour_center(template_contour)
        template_perimeter = cv2.arcLength(template_contour, closed=False)  # 开放式周长

        best_match = None
        best_score = float('inf')
        best_offset = (0.0, 0.0)

        for contour in detected_contours:
            # 使用有效面积（处理开放式轮廓）
            area = VisionAlgorithm._get_contour_effective_area(contour)
            center = VisionAlgorithm._get_contour_center(contour)
            perimeter = cv2.arcLength(contour, closed=False)  # 开放式周长

            # 面积差异（处理面积为0的情况）
            if template_area > 0:
                area_diff = abs(area - template_area) / template_area
            else:
                area_diff = 0.0

            # 周长差异（处理周长为0的情况）
            if template_perimeter > 0:
                perimeter_diff = abs(perimeter - template_perimeter) / template_perimeter
            else:
                perimeter_diff = 0.0

            # 中心点偏移
            offset_x = center[0] - template_center[0]
            offset_y = center[1] - template_center[1]
            offset_dist = math.sqrt(offset_x**2 + offset_y**2)

            # 形状匹配（对于开放式轮廓，使用自定义匹配）
            shape_score = cv2.matchShapes(template_contour, contour, cv2.CONTOURS_MATCH_I1, 0)

            # 综合评分（越小越好）
            # 权重：形状40%，面积20%，周长20%，位置20%
            score = shape_score * 0.4 + area_diff * 0.2 + perimeter_diff * 0.2 + (offset_dist / 100.0) * 0.2

            if score < best_score:
                best_score = score
                best_match = contour
                best_offset = (offset_x, offset_y)

        match_score = 1.0 / (1.0 + best_score)
        return best_match, match_score, best_offset

    @staticmethod
    def _get_contour_center(contour: np.ndarray) -> Tuple[float, float]:
        """获取轮廓中心点"""
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)
        else:
            x, y, w, h = cv2.boundingRect(contour)
            return (x + w // 2, y + h // 2)

    @staticmethod
    def calculate_contour_deviation(template_contour: np.ndarray,
                                   detected_contour: np.ndarray) -> Dict[str, float]:
        """
        计算模板轮廓与检测轮廓之间的偏差

        返回：
            偏差字典，包含：
            - offset_x: X方向偏移(像素)
            - offset_y: Y方向偏移(像素)
            - angle_diff: 角度差异
            - area_diff: 面积差异百分比
            - scale_diff: 尺度差异百分比
        """
        if template_contour is None or detected_contour is None:
            return {
                'offset_x': 0.0, 'offset_y': 0.0,
                'angle_diff': 0.0, 'area_diff': 0.0, 'scale_diff': 0.0
            }

        template_center = VisionAlgorithm._get_contour_center(template_contour)
        detected_center = VisionAlgorithm._get_contour_center(detected_contour)

        offset_x = detected_center[0] - template_center[0]
        offset_y = detected_center[1] - template_center[1]

        template_area = cv2.contourArea(template_contour)
        detected_area = cv2.contourArea(detected_contour)
        area_diff = abs(detected_area - template_area) / template_area * 100 if template_area > 0 else 0

        template_perimeter = cv2.arcLength(template_contour, True)
        detected_perimeter = cv2.arcLength(detected_contour, True)
        scale_diff = abs(detected_perimeter - template_perimeter) / template_perimeter * 100 if template_perimeter > 0 else 0

        template_angle = VisionAlgorithm._get_contour_orientation(template_contour)
        detected_angle = VisionAlgorithm._get_contour_orientation(detected_contour)
        angle_diff = abs(detected_angle - template_angle)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff

        return {
            'offset_x': offset_x,
            'offset_y': offset_y,
            'angle_diff': angle_diff,
            'area_diff': area_diff,
            'scale_diff': scale_diff
        }

    @staticmethod
    def _get_contour_orientation(contour: np.ndarray) -> float:
        """获取轮廓方向角"""
        try:
            rect = cv2.minAreaRect(contour)
            angle = rect[2]
            if abs(rect[1][0]) < abs(rect[1][1]):
                angle = angle + 90
            return angle
        except:
            return 0.0

    @staticmethod
    def match_shape(template_contour: np.ndarray, target_contour: np.ndarray) -> float:
        """形状匹配"""
        if template_contour is None or target_contour is None:
            return 0.0
        if len(template_contour) < 3 or len(target_contour) < 3:
            return 0.0
        score = cv2.matchShapes(template_contour, target_contour, cv2.CONTOURS_MATCH_I1, 0)
        return 1.0 / (1.0 + score)
    
    @staticmethod
    def calculate_line_intersection(line1: Tuple[Tuple[float, float], Tuple[float, float]],
                                    line2: Tuple[Tuple[float, float], Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        """
        计算两条直线的交点
        
        参数：
            line1: 直线1 ((x1, y1), (x2, y2))
            line2: 直线2 ((x3, y3), (x4, y4))
            
        返回：
            intersection: 交点坐标 (x, y) 或 None（平行线）
        """
        (x1, y1), (x2, y2) = line1
        (x3, y3), (x4, y4) = line2
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-6:
            return None
        
        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
        
        return (px, py)
    
    @staticmethod
    def calculate_line_angle(line: Tuple[Tuple[float, float], Tuple[float, float]]) -> float:
        """计算直线角度（度）"""
        (x1, y1), (x2, y2) = line
        return math.degrees(math.atan2(y2 - y1, x2 - x1))

    @staticmethod
    def calculate_line_center(line: Tuple[Tuple[float, float], Tuple[float, float]]) -> Tuple[float, float]:
        """计算直线中心点"""
        (x1, y1), (x2, y2) = line
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @staticmethod
    def pixel_to_mm(x_pixel: float, y_pixel: float, calib_data: CalibrationData) -> Tuple[float, float]:
        """像素坐标转物理坐标"""
        if not calib_data.is_calibrated:
            return (0.0, 0.0)
        x_mm = (x_pixel - calib_data.offset_x) / calib_data.pixel_per_mm_x
        y_mm = (y_pixel - calib_data.offset_y) / calib_data.pixel_per_mm_y
        return (x_mm, y_mm)


class TemplateManager:
    """模板管理类"""
    
    @staticmethod
    def save_dual_line_template(filename: str, template: DualLineTemplate):
        """保存双直线模板"""
        try:
            data = {
                'left_line_long': None,
                'left_line_short': None,
                'right_line_long': None,
                'right_line_short': None,
                'left_roi': None,
                'right_roi': None,
                'valid': False
            }
            
            # 转换函数：确保所有数值都是 Python 原生类型
            def to_python_type(val):
                if isinstance(val, (np.integer, np.int32, np.int64, np.intc)):
                    return int(val)
                elif isinstance(val, (np.floating, np.float32, np.float64)):
                    return float(val)
                return val
            
            # 保存左侧长边
            if template.left_line_long and isinstance(template.left_line_long, tuple) and len(template.left_line_long) == 2:
                start, end = template.left_line_long
                if isinstance(start, (tuple, list)) and len(start) == 2 and isinstance(end, (tuple, list)) and len(end) == 2:
                    data['left_line_long'] = {
                        'start': [to_python_type(start[0]), to_python_type(start[1])],
                        'end': [to_python_type(end[0]), to_python_type(end[1])]
                    }
            
            # 保存左侧短边
            if template.left_line_short and isinstance(template.left_line_short, tuple) and len(template.left_line_short) == 2:
                start, end = template.left_line_short
                if isinstance(start, (tuple, list)) and len(start) == 2 and isinstance(end, (tuple, list)) and len(end) == 2:
                    data['left_line_short'] = {
                        'start': [to_python_type(start[0]), to_python_type(start[1])],
                        'end': [to_python_type(end[0]), to_python_type(end[1])]
                    }
            
            # 保存右侧长边
            if template.right_line_long and isinstance(template.right_line_long, tuple) and len(template.right_line_long) == 2:
                start, end = template.right_line_long
                if isinstance(start, (tuple, list)) and len(start) == 2 and isinstance(end, (tuple, list)) and len(end) == 2:
                    data['right_line_long'] = {
                        'start': [to_python_type(start[0]), to_python_type(start[1])],
                        'end': [to_python_type(end[0]), to_python_type(end[1])]
                    }
            
            # 保存右侧短边
            if template.right_line_short and isinstance(template.right_line_short, tuple) and len(template.right_line_short) == 2:
                start, end = template.right_line_short
                if isinstance(start, (tuple, list)) and len(start) == 2 and isinstance(end, (tuple, list)) and len(end) == 2:
                    data['right_line_short'] = {
                        'start': [to_python_type(start[0]), to_python_type(start[1])],
                        'end': [to_python_type(end[0]), to_python_type(end[1])]
                    }
            
            # 保存ROI
            if template.left_roi and isinstance(template.left_roi, (tuple, list)) and len(template.left_roi) == 4:
                data['left_roi'] = [to_python_type(template.left_roi[0]),
                                    to_python_type(template.left_roi[1]),
                                    to_python_type(template.left_roi[2]),
                                    to_python_type(template.left_roi[3])]
            if template.right_roi and isinstance(template.right_roi, (tuple, list)) and len(template.right_roi) == 4:
                data['right_roi'] = [to_python_type(template.right_roi[0]),
                                     to_python_type(template.right_roi[1]),
                                     to_python_type(template.right_roi[2]),
                                     to_python_type(template.right_roi[3])]
            
            data['valid'] = bool(template.valid)
            
            # 先写入临时文件，确保数据完整后再替换
            temp_filename = filename + '.tmp'
            with open(temp_filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # 替换原文件
            import os
            if os.path.exists(filename):
                os.remove(filename)
            os.rename(temp_filename, filename)
            
        except Exception as e:
            print(f"save_dual_line_template error: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    @staticmethod
    def load_dual_line_template(filename: str) -> DualLineTemplate:
        """加载双直线模板"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            template = DualLineTemplate()
            
            # 加载左侧长边
            if data.get('left_line_long') and isinstance(data['left_line_long'], dict):
                start = data['left_line_long'].get('start')
                end = data['left_line_long'].get('end')
                if isinstance(start, list) and isinstance(end, list) and len(start) == 2 and len(end) == 2:
                    template.left_line_long = (tuple(start), tuple(end))
            
            # 加载左侧短边
            if data.get('left_line_short') and isinstance(data['left_line_short'], dict):
                start = data['left_line_short'].get('start')
                end = data['left_line_short'].get('end')
                if isinstance(start, list) and isinstance(end, list) and len(start) == 2 and len(end) == 2:
                    template.left_line_short = (tuple(start), tuple(end))
            
            # 加载右侧长边
            if data.get('right_line_long') and isinstance(data['right_line_long'], dict):
                start = data['right_line_long'].get('start')
                end = data['right_line_long'].get('end')
                if isinstance(start, list) and isinstance(end, list) and len(start) == 2 and len(end) == 2:
                    template.right_line_long = (tuple(start), tuple(end))
            
            # 加载右侧短边
            if data.get('right_line_short') and isinstance(data['right_line_short'], dict):
                start = data['right_line_short'].get('start')
                end = data['right_line_short'].get('end')
                if isinstance(start, list) and isinstance(end, list) and len(start) == 2 and len(end) == 2:
                    template.right_line_short = (tuple(start), tuple(end))
            
            # 加载左侧ROI
            if data.get('left_roi') and isinstance(data['left_roi'], (list, tuple)) and len(data['left_roi']) == 4:
                template.left_roi = tuple(data['left_roi'])
            
            # 加载右侧ROI
            if data.get('right_roi') and isinstance(data['right_roi'], (list, tuple)) and len(data['right_roi']) == 4:
                template.right_roi = tuple(data['right_roi'])
            
            template.valid = bool(data.get('valid', False))
            
            return template
            
        except FileNotFoundError:
            # 文件缺失属于可预期情况（尚未保存模板），返回 None 由调用方提示
            print(f"load_dual_line_template: 文件不存在 {filename}")
            return None
        except Exception as e:
            print(f"load_dual_line_template error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    @staticmethod
    def save_shape_template(filename: str, template: ShapeTemplate, params: Dict[str, Any] = None):
        """保存异形模板（JSON格式，与双直线模板一致）"""
        def contour_to_list(contour):
            """将numpy轮廓数组转换为列表"""
            if contour is None:
                return None
            return contour.tolist()
        
        data = {
            'left_contour': contour_to_list(template.left_contour),
            'right_contour': contour_to_list(template.right_contour),
            'left_roi': list(template.left_roi) if template.left_roi else None,
            'right_roi': list(template.right_roi) if template.right_roi else None,
            'center': list(template.center) if template.center else [0, 0],
            'angle': template.angle,
            'area': template.area,
            'name': template.name,
            'valid': template.valid,
            'params': params or {}
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load_shape_template(filename: str) -> Tuple[ShapeTemplate, Dict[str, Any]]:
        """加载异形模板（JSON格式）"""
        def list_to_contour(contour_list):
            """将列表转换为numpy轮廓数组"""
            if contour_list is None:
                return None
            return np.array(contour_list, dtype=np.int32)
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            template = ShapeTemplate()
            template.left_contour = list_to_contour(data.get('left_contour'))
            template.right_contour = list_to_contour(data.get('right_contour'))
            template.left_roi = tuple(data['left_roi']) if data.get('left_roi') else None
            template.right_roi = tuple(data['right_roi']) if data.get('right_roi') else None
            template.center = tuple(data.get('center', [0, 0]))
            template.angle = float(data.get('angle', 0.0))
            template.area = float(data.get('area', 0.0))
            template.name = data.get('name', '')
            template.valid = bool(data.get('valid', False))
            
            return template, data.get('params', {})
            
        except FileNotFoundError:
            print(f"load_shape_template: 文件不存在 {filename}")
            return None, {}
        except Exception as e:
            print(f"load_shape_template error: {e}")
            import traceback
            traceback.print_exc()
            return None, {}
    
    @staticmethod
    def save_calibration(filename: str, calib_data: CalibrationData):
        """保存标定数据"""
        data = {
            'camera_matrix': calib_data.camera_matrix.tolist(),
            'dist_coeffs': calib_data.dist_coeffs.tolist(),
            'pixel_per_mm_x': calib_data.pixel_per_mm_x,
            'pixel_per_mm_y': calib_data.pixel_per_mm_y,
            'offset_x': calib_data.offset_x,
            'offset_y': calib_data.offset_y,
            'is_calibrated': calib_data.is_calibrated,
            'reprojection_error': calib_data.reprojection_error,
            'image_width': calib_data.image_width,
            'image_height': calib_data.image_height
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def load_calibration(filename: str) -> CalibrationData:
        """加载标定数据"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"load_calibration: 文件不存在 {filename}")
            return None
        except Exception as e:
            print(f"load_calibration error: {e}")
            return None
        
        calib_data = CalibrationData()
        calib_data.camera_matrix = np.array(data['camera_matrix'])
        calib_data.dist_coeffs = np.array(data['dist_coeffs'])
        calib_data.pixel_per_mm_x = data.get('pixel_per_mm_x', 1.0)
        calib_data.pixel_per_mm_y = data.get('pixel_per_mm_y', 1.0)
        calib_data.offset_x = data.get('offset_x', 0.0)
        calib_data.offset_y = data.get('offset_y', 0.0)
        calib_data.is_calibrated = data.get('is_calibrated', False)
        calib_data.reprojection_error = data.get('reprojection_error', 0.0)
        calib_data.image_width = data.get('image_width', 1280)
        calib_data.image_height = data.get('image_height', 720)
        
        return calib_data


class CameraUtils:
    """摄像头工具类"""
    
    @staticmethod
    def detect_cameras(max_devices=10) -> List[Tuple[int, int, int]]:
        """检测可用摄像头"""
        cameras = []
        for i in range(max_devices):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cameras.append((i, w, h))
                cap.release()
        return cameras


# =====================================================================
# 坐标空间 mm 化管线 + 机械手坐标输出（2026-10-26 浩达验收补全）
# 对齐实体封面机 ZFM-500B 口径：输出单位 mm、估读精度 ±0.001mm
# =====================================================================

@dataclass
class RobotPose:
    """机械手可直接消费的坐标姿态（对齐浩达 ZFM-500B 机械手接口）"""
    x_mm: float = 0.0
    y_mm: float = 0.0
    z_mm: float = 0.0
    a_deg: float = 0.0          # 绕Z轴角度（度）
    timestamp_ms: float = 0.0
    valid: bool = False
    score: float = 0.0

    def to_list(self):
        return [round(self.x_mm, 3), round(self.y_mm, 3),
                round(self.z_mm, 3), round(self.a_deg, 3),
                self.valid, round(self.score, 3)]

    def to_protocol_string(self, proto: str = "csv") -> str:
        """生成机械手协议字符串（逗号分隔，最后一行带换行）"""
        if proto == "csv":
            return f"{self.x_mm:.3f},{self.y_mm:.3f},{self.z_mm:.3f},{self.a_deg:.3f},{int(self.valid)},{self.score:.3f}\n"
        elif proto == "tcp":
            # 简单 Modbus/TCP 风格寄存器打包（占位格式：0x55 帧头 + 6个float + 0xAA 帧尾）
            import struct
            header = bytes([0x55])
            payload = struct.pack("<4f1f", self.x_mm, self.y_mm, self.z_mm, self.a_deg, self.score)
            footer = bytes([0xAA])
            return (header + payload + footer).hex() + "\n"
        return self.to_list()


class CoordinatePipeline:
    """
    像素 → mm 坐标空间转换管线 + 机械手输出接口
    
    实体机对齐口径：
    - 视觉输出坐标 → 机械手纠偏参考（坐标偏差平均 0.085mm / 最大 0.112mm）
    - 坐标系：世界坐标 mm，估读精度 ±0.001mm
    - 支持 offset 补偿（装配偏差、工装基准偏移）
    
    用法：
        cp = CoordinatePipeline(calib_data, offsets=(0,0,0,0))
        result_mm = cp.pixel_to_mm(detection_result)
        robot_pose = cp.to_robot_pose(result_mm)
        tcp_string = cp.to_tcp_stream(robot_pose)
    """

    def __init__(self, calib_data: CalibrationData,
                 offsets: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
                 default_z_mm: float = 0.0):
        self.calib = calib_data
        self.off_x, self.off_y, self.off_z, self.off_a = offsets
        self.default_z = default_z_mm
        # 未标定时提供默认 10px/mm 的仿真换算（仅测试用，实际必须标定）
        self.px_per_mm_x = calib_data.pixel_per_mm_x if calib_data.is_calibrated and calib_data.pixel_per_mm_x > 0 else 10.0
        self.px_per_mm_y = calib_data.pixel_per_mm_y if calib_data.is_calibrated and calib_data.pixel_per_mm_y > 0 else 10.0

    def pixel_to_mm(self, px: float, axis: str = "x") -> float:
        """像素 → mm 单轴"""
        if axis.lower() == "x":
            return px / self.px_per_mm_x - self.off_x
        else:
            return px / self.px_per_mm_y - self.off_y

    def detection_to_mm(self, det: DetectionResult) -> DetectionResult:
        """DetectionResult 从像素空间完整转换到 mm 空间"""
        det.x_mm = self.pixel_to_mm(det.x_pixel, "x")
        det.y_mm = self.pixel_to_mm(det.y_pixel, "y")
        det.z_mm = self.default_z - self.off_z
        # 角度一般在像素空间已解算好，仅保留。a_offset 用于机械臂零位补偿
        return det

    def to_robot_pose(self, det: DetectionResult) -> RobotPose:
        """从检测结果生成机械手 ready 的坐标姿态"""
        det = self.detection_to_mm(det)
        return RobotPose(
            x_mm=det.x_mm, y_mm=det.y_mm, z_mm=det.z_mm,
            a_deg=det.angle - self.off_a,
            timestamp_ms=det.timestamp,
            valid=det.valid, score=det.score
        )

    def robot_pose_to_protocol(self, pose: RobotPose, proto: str = "csv") -> str:
        return pose.to_protocol_string(proto)


class MMSpaceEvaluator:
    """
    mm 空间误差评估器（对齐实体封面机验收口径）
    
    实体机指标：
    - 视觉坐标偏差：平均 0.085mm / 最大 0.112mm（供机械手纠偏参考）
    - 考核线：识别准确率 ≥99.5%、抓取成功率 ≥99.8%、定位误差 ≤0.1mm
    
    用法：
        evaluator = MMSpaceEvaluator(tolerance_mm=0.1)
        evaluator.feed_batch(batch_predictions_mm, batch_ground_truth_mm)
        report = evaluator.summary()  # 返回对齐实体机口径的统计 dict
    """

    def __init__(self, tolerance_mm: float = 0.1,
                 accuracy_threshold: float = 0.995,
                 angular_tolerance_deg: float = 0.5):
        self.tol_mm = tolerance_mm
        self.acc_threshold = accuracy_threshold
        self.ang_tol = angular_tolerance_deg
        self.errors_mm: List[float] = []
        self.angle_errors_deg: List[float] = []
        self.detection_results: List[bool] = []      # True=成功检测
        self.timestamp_samples: List[float] = []

    def feed(self, pred_xy: Tuple[float, float], gt_xy: Tuple[float, float],
             angle_pred: float = 0.0, angle_gt: float = 0.0,
             detected: bool = True, timestamp: float = 0.0):
        dx = pred_xy[0] - gt_xy[0]
        dy = pred_xy[1] - gt_xy[1]
        dist = math.sqrt(dx * dx + dy * dy)
        self.errors_mm.append(dist)
        self.angle_errors_deg.append(abs(angle_pred - angle_gt))
        self.detection_results.append(detected)
        if timestamp:
            self.timestamp_samples.append(timestamp)

    def feed_batch(self, preds, gts, angles=None, angles_gt=None,
                   detected_flags=None, timestamps=None):
        for i in range(len(preds)):
            angle_p = angles[i] if angles else 0.0
            angle_g = angles_gt[i] if angles_gt else 0.0
            det = detected_flags[i] if detected_flags else True
            ts = timestamps[i] if timestamps else 0.0
            self.feed(preds[i], gts[i], angle_p, angle_g, det, ts)

    def summary(self) -> Dict[str, Any]:
        n = len(self.errors_mm)
        if n == 0:
            return {"error": "no samples"}
        import statistics as st
        mean_err = st.mean(self.errors_mm)
        max_err = max(self.errors_mm)
        mean_ang_err = st.mean(self.angle_errors_deg) if self.angle_errors_deg else 0.0
        acc = sum(self.detection_results) / n
        # 坐标定位精度 ≤ tol_mm 的样本占比（对齐实体机定位误差≤0.1mm 口径）
        pos_pass = sum(1 for e in self.errors_mm if e <= self.tol_mm) / n
        ang_pass = sum(1 for e in self.angle_errors_deg if e <= self.ang_tol) / n if self.angle_errors_deg else 1.0
        overall_pass = (acc >= self.acc_threshold) and (pos_pass >= 0.998) and (ang_pass >= 0.99)

        return {
            "sample_count": n,
            "accuracy": round(acc * 100, 2),                     # 识别准确率 %
            "accuracy_threshold": self.acc_threshold * 100,
            "mean_position_error_mm": round(mean_err, 4),        # 平均坐标偏差 mm
            "max_position_error_mm": round(max_err, 4),          # 最大坐标偏差 mm
            "position_tolerance_mm": self.tol_mm,
            "position_pass_rate_pct": round(pos_pass * 100, 2),  # 定位误差≤tol 的占比
            "mean_angle_error_deg": round(mean_ang_err, 4),
            "angle_pass_rate_pct": round(ang_pass * 100, 2),
            "overall_pass": overall_pass,
            "unit": "mm / deg",
            "caliber": "浩达 ZFM-500B 验收口径对齐",
        }
