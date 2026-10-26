#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视觉定位系统工具类
包含数据结构、算法实现等核心功能
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
    pixel_per_mm_x: float = 1.0
    pixel_per_mm_y: float = 1.0
    offset_x: float = 0.0
    offset_y: float = 0.0
    points_pixel: List[Tuple[float, float]] = field(default_factory=list)
    points_physical: List[Tuple[float, float]] = field(default_factory=list)
    is_calibrated: bool = False


@dataclass
class LineTemplate:
    start: Tuple[float, float] = (0, 0)
    end: Tuple[float, float] = (0, 0)
    rho: float = 0.0
    theta: float = 0.0
    valid: bool = False


@dataclass
class ShapeTemplate:
    contour: np.ndarray = field(default_factory=lambda: np.array([]))
    center: Tuple[float, float] = (0, 0)
    angle: float = 0.0
    area: float = 0.0
    name: str = ""
    valid: bool = False


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


class VisionAlgorithm:
    @staticmethod
    def detect_lines(frame: np.ndarray, canny_low=50, canny_high=150, hough_threshold=50) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
        edges = cv2.Canny(blurred, canny_low, canny_high)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, hough_threshold, minLineLength=30, maxLineGap=10)
        
        detected_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
                if length > 30:
                    detected_lines.append(((x1, y1), (x2, y2), length))
            detected_lines.sort(key=lambda x: x[2], reverse=True)
        
        return [(line[0], line[1]) for line in detected_lines]

    @staticmethod
    def detect_contours(frame: np.ndarray, canny_low=50, canny_high=150) -> List[np.ndarray]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
        edges = cv2.Canny(blurred, canny_low, canny_high)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    @staticmethod
    def match_shape(template_contour: np.ndarray, target_contour: np.ndarray) -> float:
        if len(template_contour) < 3 or len(target_contour) < 3:
            return 0.0
        score = cv2.matchShapes(template_contour, target_contour, cv2.CONTOURS_MATCH_I1, 0)
        return 1.0 / (1.0 + score)

    @staticmethod
    def calculate_center(lines: List[Tuple[Tuple[int, int], Tuple[int, int]]]) -> Tuple[float, float]:
        if len(lines) < 2:
            return (0.0, 0.0)
        all_points = []
        for start, end in lines[:2]:
            all_points.extend([start, end])
        return (sum(p[0] for p in all_points) / len(all_points),
                sum(p[1] for p in all_points) / len(all_points))

    @staticmethod
    def calculate_angle(lines: List[Tuple[Tuple[int, int], Tuple[int, int]]]) -> float:
        if len(lines) < 2:
            return 0.0
        angles = []
        for start, end in lines[:2]:
            angles.append(math.degrees(math.atan2(end[1]-start[1], end[0]-start[0])))
        return sum(angles) / len(angles)

    @staticmethod
    def pixel_to_mm(x_pixel: float, y_pixel: float, calib_data: CalibrationData) -> Tuple[float, float]:
        if not calib_data.is_calibrated:
            return (0.0, 0.0)
        x_mm = (x_pixel - calib_data.offset_x) / calib_data.pixel_per_mm_x
        y_mm = (y_pixel - calib_data.offset_y) / calib_data.pixel_per_mm_y
        return (x_mm, y_mm)


class TemplateManager:
    @staticmethod
    def save_line_template(filename: str, left_line, right_line, left_roi, right_roi):
        data = {
            'left_line': {'start': left_line[0], 'end': left_line[1]} if left_line else None,
            'right_line': {'start': right_line[0], 'end': right_line[1]} if right_line else None,
            'left_roi': left_roi,
            'right_roi': right_roi
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load_line_template(filename: str) -> Dict[str, Any]:
        with open(filename, 'r') as f:
            return json.load(f)

    @staticmethod
    def save_shape_template(filename: str, left_contour, right_contour, canny_low, canny_high, match_threshold):
        data = {
            'left_contour': left_contour,
            'right_contour': right_contour,
            'canny_low': canny_low,
            'canny_high': canny_high,
            'match_threshold': match_threshold
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    @staticmethod
    def load_shape_template(filename: str) -> Dict[str, Any]:
        with open(filename, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def save_paste_params(filename: str, params: Dict[str, Any]):
        with open(filename, 'w') as f:
            json.dump(params, f, indent=2)

    @staticmethod
    def load_paste_params(filename: str) -> Dict[str, Any]:
        with open(filename, 'r') as f:
            return json.load(f)


class CameraUtils:
    @staticmethod
    def detect_cameras(max_devices=10) -> List[Tuple[int, int, int]]:
        cameras = []
        for i in range(max_devices):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cameras.append((i, w, h))
                cap.release()
        return cameras
