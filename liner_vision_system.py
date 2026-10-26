#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
贴内衬机视觉定位系统 - Inner Liner Vision Positioning System
工业级、科技风、HUD风格GUI界面
核心功能: 摄像头采集、相机标定、双直线/双外形模板匹配、贴料参数设置
"""

import sys
import os
import json
import math
import time
import pickle
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QFrame, QGridLayout, QProgressBar, QSlider,
    QComboBox, QGroupBox, QCheckBox, QTabWidget, QTableWidget, 
    QTableWidgetItem, QHeaderView, QSpinBox, QDoubleSpinBox,
    QFileDialog, QMessageBox, QSplitter, QStatusBar, QTextEdit,
    QLineEdit
)
from PyQt5.QtCore import (
    Qt, QTimer, pyqtSignal, QThread, QRect, QPoint, QSize, QPointF, QMutex
)
from PyQt5.QtGui import (
    QPainter, QColor, QPen, QBrush, QFont, QLinearGradient,
    QRadialGradient, QPainterPath, QPolygonF, QImage, QPixmap,
    QCursor, QFontDatabase
)

try:
    import cv2
except ImportError:
    print("请安装OpenCV: pip install opencv-python")
    sys.exit(1)


# ==================== 数据结构定义 ====================
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


@dataclass
class PasteParams:
    offset_x: float = 0.0
    offset_y: float = 0.0
    offset_z: float = 0.0
    offset_a: float = 0.0
    timeout_ms: int = 3000
    score_threshold: float = 0.7
    alarm_x: float = 5.0
    alarm_y: float = 5.0
    alarm_a: float = 3.0
    photo_delay_ms: int = 100
    vacuum_delay_ms: int = 200
    retry_count: int = 3
    fail_mode: str = "alarm"


# ==================== 霓虹风格定义 ====================
class NeonStyle:
    NEON_BLUE = QColor(0, 150, 255)
    NEON_CYAN = QColor(0, 255, 255)
    NEON_PURPLE = QColor(150, 0, 255)
    NEON_GREEN = QColor(0, 255, 150)
    NEON_RED = QColor(255, 50, 100)
    NEON_ORANGE = QColor(255, 150, 0)
    NEON_YELLOW = QColor(255, 255, 0)
    DARK_BG = QColor(5, 10, 25)
    PANEL_BG = QColor(10, 20, 40, 200)
    GRID_COLOR = QColor(20, 40, 80, 100)
    TEXT_COLOR = QColor(0, 200, 255)


# ==================== 摄像头线程 ====================
class CameraThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, camera_index: int = 0, resolution: Tuple[int, int] = (1280, 720)):
        super().__init__()
        self.camera_index = camera_index
        self.resolution = resolution
        self.running = False
        self.paused = False
        self.camera = None
        
    def run(self):
        try:
            self.camera = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if not self.camera.isOpened():
                self.error_occurred.emit(f"无法打开摄像头 {self.camera_index}")
                return
                
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            self.running = True
            while self.running:
                if not self.paused:
                    ret, frame = self.camera.read()
                    if ret:
                        self.frame_ready.emit(frame)
                self.msleep(33)
                
        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            if self.camera:
                self.camera.release()
        
    def stop(self):
        self.running = False
        self.wait()
        
    def set_resolution(self, width: int, height: int):
        self.resolution = (width, height)
        if self.camera and self.camera.isOpened():
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


# ==================== 图像处理线程 ====================
class VisionProcessor(QThread):
    result_ready = pyqtSignal(DetectionResult)
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.paused = False
        self.current_frame = None
        self.current_mode = "line"  # "line" or "shape"
        
        self.left_line_template = LineTemplate()
        self.right_line_template = LineTemplate()
        self.left_shape_template = ShapeTemplate()
        self.right_shape_template = ShapeTemplate()
        
        self.left_roi = None
        self.right_roi = None
        
        self.canny_low = 50
        self.canny_high = 150
        self.hough_threshold = 50
        self.match_threshold = 0.7
        
        self.calibration_data = CalibrationData()
        
    def run(self):
        self.running = True
        while self.running:
            if not self.paused and self.current_frame is not None:
                result = self._process_frame()
                self.result_ready.emit(result)
            self.msleep(33)
            
    def _process_frame(self) -> DetectionResult:
        frame = self.current_frame.copy()
        
        if self.current_mode == "line":
            return self._detect_lines(frame)
        else:
            return self._detect_shapes(frame)
            
    def _detect_lines(self, frame: np.ndarray) -> DetectionResult:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, self.hough_threshold, 30, 10)
        
        if lines is None or len(lines) < 2:
            return DetectionResult(valid=False)
            
        detected_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
            theta = math.degrees(math.atan2(y2-y1, x2-x1))
            detected_lines.append((x1, y1, x2, y2, length, theta))
        
        detected_lines.sort(key=lambda x: x[4], reverse=True)
        
        if len(detected_lines) >= 2:
            l1, l2 = detected_lines[0], detected_lines[1]
            
            cx_pixel = (l1[0] + l1[2] + l2[0] + l2[2]) / 4
            cy_pixel = (l1[1] + l1[3] + l2[1] + l2[3]) / 4
            angle = (l1[5] + l2[5]) / 2
            
            cx_mm = cx_pixel / self.calibration_data.pixel_per_mm_x
            cy_mm = cy_pixel / self.calibration_data.pixel_per_mm_y
            
            return DetectionResult(
                x_pixel=cx_pixel, y_pixel=cy_pixel,
                x_mm=cx_mm, y_mm=cy_mm,
                z_mm=0, angle=angle,
                score=min(1.0, 0.8 + (l1[4] + l2[4]) / 2000),
                valid=True,
                timestamp=time.time()
            )
        
        return DetectionResult(valid=False)
            
    def _detect_shapes(self, frame: np.ndarray) -> DetectionResult:
        if not self.left_shape_template.valid and not self.right_shape_template.valid:
            return DetectionResult(valid=False)
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return DetectionResult(valid=False)
            
        centers = []
        for contour in contours:
            if len(contour) < 3:
                continue
                
            if self.left_shape_template.valid:
                score = cv2.matchShapes(self.left_shape_template.contour, contour, cv2.CONTOURS_MATCH_I1, 0)
                match_score = 1.0 / (1.0 + score)
                if match_score > self.match_threshold:
                    M = cv2.moments(contour)
                    if M['m00'] > 0:
                        cx = M['m10'] / M['m00']
                        cy = M['m01'] / M['m00']
                        centers.append((cx, cy))
                        
            if self.right_shape_template.valid:
                score = cv2.matchShapes(self.right_shape_template.contour, contour, cv2.CONTOURS_MATCH_I1, 0)
                match_score = 1.0 / (1.0 + score)
                if match_score > self.match_threshold:
                    M = cv2.moments(contour)
                    if M['m00'] > 0:
                        cx = M['m10'] / M['m00']
                        cy = M['m01'] / M['m00']
                        centers.append((cx, cy))
        
        if len(centers) >= 2:
            cx_pixel = (centers[0][0] + centers[1][0]) / 2
            cy_pixel = (centers[0][1] + centers[1][1]) / 2
            
            dx = centers[1][0] - centers[0][0]
            dy = centers[1][1] - centers[0][1]
            angle = math.degrees(math.atan2(dy, dx))
            
            cx_mm = cx_pixel / self.calibration_data.pixel_per_mm_x
            cy_mm = cy_pixel / self.calibration_data.pixel_per_mm_y
            
            return DetectionResult(
                x_pixel=cx_pixel, y_pixel=cy_pixel,
                x_mm=cx_mm, y_mm=cy_mm,
                z_mm=0, angle=angle,
                score=0.9,
                valid=True,
                timestamp=time.time()
            )
        
        return DetectionResult(valid=False)


# ==================== 图像显示画布 ====================
class ImageCanvas(QWidget):
    roi_selected = pyqtSignal(tuple)
    line_drawn = pyqtSignal(tuple)
    template_selected = pyqtSignal(tuple)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.setMouseTracking(True)
        
        self.current_frame = None
        self.display_frame = None
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0
        
        self.drawing_mode = None
        self.drawing_start = None
        self.drawing_end = None
        
        self.left_roi = None
        self.right_roi = None
        self.left_line = None
        self.right_line = None
        self.detected_left_line = None
        self.detected_right_line = None
        self.detected_contours = []
        self.detected_center = None
        self.detected_angle = 0.0
        
        self.show_grid = True
        self.show_crosshair = True
        self.show_roi = True
        self.show_lines = True
        self.show_contours = True
        
        self.left_template_contour = None
        self.right_template_contour = None
        
        self.scan_phase = 0
        self.pulse_phase = 0
        
    def set_frame(self, frame: np.ndarray):
        self.current_frame = frame.copy()
        self._update_display()
        self.update()
        
    def _update_display(self):
        if self.current_frame is None:
            return
            
        h, w = self.current_frame.shape[:2]
        canvas_w, canvas_h = self.width(), self.height()
        
        scale_w = canvas_w / w
        scale_h = canvas_h / h
        self.scale_factor = min(scale_w, scale_h) * 0.95
        
        new_w = int(w * self.scale_factor)
        new_h = int(h * self.scale_factor)
        
        self.display_frame = cv2.resize(self.current_frame, (new_w, new_h))
        self.offset_x = (canvas_w - new_w) // 2
        self.offset_y = (canvas_h - new_h) // 2
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        self._draw_background(painter)
        
        if self.display_frame is not None:
            h, w = self.display_frame.shape[:2]
            q_img = QImage(self.display_frame.data, w, h, 3*w, QImage.Format_BGR888)
            painter.drawImage(self.offset_x, self.offset_y, q_img)
        else:
            self._draw_placeholder(painter)
            
        self._draw_scan_effect(painter)
        
        if self.show_roi:
            self._draw_rois(painter)
            
        if self.show_lines:
            self._draw_lines(painter)
            
        if self.show_contours:
            self._draw_contours(painter)
            
        if self.show_crosshair:
            self._draw_crosshair(painter)
            
        if self.drawing_mode:
            self._draw_drawing_preview(painter)
            
        self._draw_hud_overlay(painter)
        self._draw_corner_decorations(painter)
        
        self.scan_phase += 0.05
        self.pulse_phase += 0.1
        
    def _draw_background(self, painter):
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0, QColor(5, 10, 30))
        gradient.setColorAt(0.5, QColor(10, 20, 50))
        gradient.setColorAt(1, QColor(5, 15, 35))
        painter.fillRect(self.rect(), gradient)
        
        if self.show_grid:
            pen = QPen(QColor(20, 40, 80, 80), 1, Qt.DotLine)
            painter.setPen(pen)
            for x in range(0, self.width(), 50):
                painter.drawLine(x, 0, x, self.height())
            for y in range(0, self.height(), 50):
                painter.drawLine(0, y, self.width(), y)
                
    def _draw_placeholder(self, painter):
        cx, cy = self.width()//2, self.height()//2
        
        gradient = QRadialGradient(cx, cy, 200)
        gradient.setColorAt(0, QColor(0, 100, 200, 50))
        gradient.setColorAt(1, QColor(0, 20, 80, 10))
        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(cx-200, cy-200, 400, 400)
        
        pen = QPen(NeonStyle.NEON_BLUE, 2)
        painter.setPen(pen)
        painter.setFont(QFont("Consolas", 16, QFont.Bold))
        painter.drawText(self.rect(), Qt.AlignCenter, "等待摄像头连接...")
        
    def _draw_scan_effect(self, painter):
        scan_x = int((self.scan_phase * 100) % self.width())
        gradient = QLinearGradient(scan_x-30, 0, scan_x+30, 0)
        gradient.setColorAt(0, QColor(0, 255, 255, 0))
        gradient.setColorAt(0.5, QColor(0, 255, 255, 40))
        gradient.setColorAt(1, QColor(0, 255, 255, 0))
        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.NoPen)
        painter.drawRect(scan_x-30, 0, 60, self.height())
        
    def _draw_rois(self, painter):
        painter.setOpacity(0.6)
        
        if self.left_roi:
            x1, y1, x2, y2 = self._scale_rect(self.left_roi)
            pen = QPen(NeonStyle.NEON_GREEN, 2, Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(QBrush(QColor(0, 255, 150, 30)))
            painter.drawRect(x1, y1, x2-x1, y2-y1)
            painter.setFont(QFont("Consolas", 9))
            painter.drawText(x1+5, y1+15, "左监测区")
            
        if self.right_roi:
            x1, y1, x2, y2 = self._scale_rect(self.right_roi)
            pen = QPen(NeonStyle.NEON_ORANGE, 2, Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(QBrush(QColor(255, 150, 0, 30)))
            painter.drawRect(x1, y1, x2-x1, y2-y1)
            painter.drawText(x1+5, y1+15, "右监测区")
            
        painter.setOpacity(1.0)
        
    def _draw_lines(self, painter):
        if self.left_line:
            x1, y1 = self._scale_point(self.left_line[0])
            x2, y2 = self._scale_point(self.left_line[1])
            pen = QPen(NeonStyle.NEON_GREEN, 3)
            painter.setPen(pen)
            painter.drawLine(x1, y1, x2, y2)
            
        if self.right_line:
            x1, y1 = self._scale_point(self.right_line[0])
            x2, y2 = self._scale_point(self.right_line[1])
            pen = QPen(NeonStyle.NEON_ORANGE, 3)
            painter.setPen(pen)
            painter.drawLine(x1, y1, x2, y2)
            
        if self.detected_left_line:
            x1, y1 = self._scale_point(self.detected_left_line[0])
            x2, y2 = self._scale_point(self.detected_left_line[1])
            pen = QPen(NeonStyle.NEON_CYAN, 4)
            painter.setPen(pen)
            painter.drawLine(x1, y1, x2, y2)
            
        if self.detected_right_line:
            x1, y1 = self._scale_point(self.detected_right_line[0])
            x2, y2 = self._scale_point(self.detected_right_line[1])
            pen = QPen(NeonStyle.NEON_PURPLE, 4)
            painter.setPen(pen)
            painter.drawLine(x1, y1, x2, y2)
            
    def _draw_contours(self, painter):
        for contour, color in self.detected_contours:
            if len(contour) < 3:
                continue
            points = [self._scale_point((int(p[0][0]), int(p[0][1]))) for p in contour]
            polygon = QPolygonF([QPointF(x, y) for x, y in points])
            pen = QPen(color, 2)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawPolygon(polygon)
            
        if self.left_template_contour is not None:
            self._draw_single_contour(painter, self.left_template_contour, NeonStyle.NEON_GREEN)
            
        if self.right_template_contour is not None:
            self._draw_single_contour(painter, self.right_template_contour, NeonStyle.NEON_ORANGE)
            
        if self.detected_center:
            cx, cy = self._scale_point(self.detected_center)
            pulse = int(5 * math.sin(self.pulse_phase))
            
            pen = QPen(NeonStyle.NEON_RED, 2)
            painter.setPen(pen)
            size = 20 + pulse
            painter.drawLine(cx-size, cy, cx-5, cy)
            painter.drawLine(cx+5, cy, cx+size, cy)
            painter.drawLine(cx, cy-size, cx, cy-5)
            painter.drawLine(cx, cy+5, cx, cy+size)
            
            painter.drawEllipse(cx-8, cy-8, 16, 16)
            
            if abs(self.detected_angle) > 0.1:
                angle_rad = math.radians(self.detected_angle)
                length = 60
                dx = length * math.cos(angle_rad)
                dy = length * math.sin(angle_rad)
                pen = QPen(NeonStyle.NEON_YELLOW, 2)
                painter.setPen(pen)
                painter.drawLine(int(cx-dx), int(cy-dy), int(cx+dx), int(cy+dy))
                
    def _draw_single_contour(self, painter, contour, color):
        if len(contour) < 3:
            return
        points = [self._scale_point((int(p[0][0]), int(p[0][1]))) for p in contour]
        polygon = QPolygonF([QPointF(x, y) for x, y in points])
        pen = QPen(color, 2, Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 30)))
        painter.drawPolygon(polygon)
        
    def _draw_crosshair(self, painter):
        cx, cy = self.width()//2, self.height()//2
        size = 25
        gap = 8
        
        pulse = int(3 * math.sin(self.pulse_phase))
        pen = QPen(NeonStyle.NEON_CYAN, 1)
        painter.setPen(pen)
        
        painter.drawLine(cx-size-pulse, cy, cx-gap, cy)
        painter.drawLine(cx+gap, cy, cx+size+pulse, cy)
        painter.drawLine(cx, cy-size-pulse, cx, cy-gap)
        painter.drawLine(cx, cy+gap, cx, cy+size+pulse)
        
    def _draw_drawing_preview(self, painter):
        if self.drawing_start and self.drawing_end:
            sx, sy = self.drawing_start
            ex, ey = self.drawing_end
            
            if self.drawing_mode == "roi":
                pen = QPen(NeonStyle.NEON_YELLOW, 2, Qt.DashLine)
                painter.setPen(pen)
                painter.setBrush(QBrush(QColor(255, 255, 0, 50)))
                painter.drawRect(sx, sy, ex-sx, ey-sy)
            elif self.drawing_mode == "line":
                pen = QPen(NeonStyle.NEON_YELLOW, 3)
                painter.setPen(pen)
                painter.drawLine(sx, sy, ex, ey)
            elif self.drawing_mode == "template":
                pen = QPen(NeonStyle.NEON_PURPLE, 2, Qt.DashLine)
                painter.setPen(pen)
                painter.setBrush(QBrush(QColor(150, 0, 255, 50)))
                painter.drawRect(sx, sy, ex-sx, ey-sy)
                
    def _draw_hud_overlay(self, painter):
        pen = QPen(NeonStyle.NEON_CYAN, 1)
        painter.setPen(pen)
        painter.setFont(QFont("Consolas", 10))
        
        painter.drawText(15, 25, f"模式: {self.drawing_mode or '观察'}")
        
        if self.current_frame is not None:
            h, w = self.current_frame.shape[:2]
            painter.drawText(15, 45, f"分辨率: {w}x{h}")
            
    def _draw_corner_decorations(self, painter):
        pen = QPen(NeonStyle.NEON_BLUE, 3)
        painter.setPen(pen)
        size = 30
        
        painter.drawLine(0, size, 0, 0)
        painter.drawLine(0, 0, size, 0)
        
        painter.drawLine(self.width()-size, 0, self.width(), 0)
        painter.drawLine(self.width(), 0, self.width(), size)
        
        painter.drawLine(0, self.height()-size, 0, self.height())
        painter.drawLine(0, self.height(), size, self.height())
        
        painter.drawLine(self.width()-size, self.height(), self.width(), self.height())
        painter.drawLine(self.width(), self.height()-size, self.width(), self.height())
        
    def _scale_point(self, point):
        x = int(point[0] * self.scale_factor + self.offset_x)
        y = int(point[1] * self.scale_factor + self.offset_y)
        return (x, y)
        
    def _scale_rect(self, rect):
        x1 = int(rect[0] * self.scale_factor + self.offset_x)
        y1 = int(rect[1] * self.scale_factor + self.offset_y)
        x2 = int(rect[2] * self.scale_factor + self.offset_x)
        y2 = int(rect[3] * self.scale_factor + self.offset_y)
        return (x1, y1, x2, y2)
        
    def _unscale_point(self, point):
        x = (point[0] - self.offset_x) / self.scale_factor
        y = (point[1] - self.offset_y) / self.scale_factor
        return (x, y)
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing_mode:
            self.drawing_start = (event.x(), event.y())
            self.drawing_end = (event.x(), event.y())
            
    def mouseMoveEvent(self, event):
        if self.drawing_start:
            self.drawing_end = (event.x(), event.y())
            self.update()
            
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing_start and self.drawing_end:
            sx, sy = self._unscale_point(self.drawing_start)
            ex, ey = self._unscale_point(self.drawing_end)
            
            if self.drawing_mode == "roi":
                rect = (min(sx, ex), min(sy, ey), max(sx, ex), max(sy, ey))
                self.roi_selected.emit(rect)
            elif self.drawing_mode == "line":
                self.line_drawn.emit((sx, sy, ex, ey))
            elif self.drawing_mode == "template":
                rect = (min(sx, ex), min(sy, ey), max(sx, ex), max(sy, ey))
                self.template_selected.emit(rect)
                
            self.drawing_start = None
            self.drawing_end = None
            self.drawing_mode = None
            self.update()
            
    def start_drawing(self, mode):
        self.drawing_mode = mode
        self.setCursor(QCursor(Qt.CrossCursor))
            
    def stop_drawing(self):
        self.drawing_mode = None
        self.setCursor(QCursor(Qt.ArrowCursor))


# ==================== 主窗口 ====================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("贴内衬机视觉定位系统")
        self.setGeometry(50, 50, 1600, 1000)
        self.setStyleSheet("QMainWindow { background-color: #0A0F1E; }")
        
        # 系统状态
        self.camera_thread = None
        self.vision_processor = VisionProcessor()
        self.current_mode = "line"  # "line" or "shape"
        self.calibration_data = CalibrationData()
        
        # UI组件
        self.image_canvas = ImageCanvas()
        self.status_label = None
        self.status_light = None
        
        # 结果数据
        self.last_result = DetectionResult()
        self.result_history = deque(maxlen=100)
        
        # 初始化UI
        self._setup_ui()
        self._setup_connections()
        
    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # 左侧面板 - 摄像头与标定
        left_panel = self._create_left_panel()
        main_layout.addWidget(left_panel, stretch=1)
        
        # 中间面板 - 图像显示
        center_panel = self._create_center_panel()
        main_layout.addWidget(center_panel, stretch=3)
        
        # 右侧面板 - 模板与参数
        right_panel = self._create_right_panel()
        main_layout.addWidget(right_panel, stretch=1)
        
        # 状态栏
        self._setup_statusbar()
        
    def _create_left_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        
        # 摄像头控制
        camera_group = QGroupBox("摄像头控制")
        camera_group.setStyleSheet(self._group_style())
        camera_layout = QVBoxLayout(camera_group)
        
        self.btn_detect_cam = QPushButton("🔍 检测摄像头")
        self.btn_detect_cam.setStyleSheet(self._button_style())
        self.btn_detect_cam.clicked.connect(self._detect_cameras)
        camera_layout.addWidget(self.btn_detect_cam)
        
        camera_layout.addWidget(QLabel("选择摄像头:"))
        self.camera_combo = QComboBox()
        self.camera_combo.addItem("请检测摄像头")
        self.camera_combo.setStyleSheet(self._combo_style())
        camera_layout.addWidget(self.camera_combo)
        
        camera_layout.addWidget(QLabel("分辨率:"))
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["640x480", "800x600", "1280x720", "1920x1080"])
        self.resolution_combo.setStyleSheet(self._combo_style())
        camera_layout.addWidget(self.resolution_combo)
        
        self.btn_open_cam = QPushButton("📷 打开摄像头")
        self.btn_open_cam.setStyleSheet(self._button_style("#00AA66"))
        self.btn_open_cam.clicked.connect(self._toggle_camera)
        camera_layout.addWidget(self.btn_open_cam)
        
        layout.addWidget(camera_group)
        
        # 相机标定
        calib_group = QGroupBox("相机标定")
        calib_group.setStyleSheet(self._group_style())
        calib_layout = QVBoxLayout(calib_group)
        
        calib_layout.addWidget(QLabel("标定方法:"))
        self.calib_method = QComboBox()
        self.calib_method.addItems(["九点标定", "单点标定", "手动输入"])
        self.calib_method.setStyleSheet(self._combo_style())
        calib_layout.addWidget(self.calib_method)
        
        self.calib_table = QTableWidget()
        self.calib_table.setColumnCount(4)
        self.calib_table.setHorizontalHeaderLabels(["#", "像素X", "像素Y", "物理值(mm)"])
        self.calib_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.calib_table.setStyleSheet(self._table_style())
        self.calib_table.setMaximumHeight(150)
        calib_layout.addWidget(self.calib_table)
        
        calib_btn_layout = QHBoxLayout()
        self.btn_add_point = QPushButton("添加点")
        self.btn_add_point.setStyleSheet(self._button_style())
        self.btn_add_point.clicked.connect(self._add_calib_point)
        calib_btn_layout.addWidget(self.btn_add_point)
        
        self.btn_clear_points = QPushButton("清除")
        self.btn_clear_points.setStyleSheet(self._button_style("#FF6666"))
        self.btn_clear_points.clicked.connect(self._clear_calib_points)
        calib_btn_layout.addWidget(self.btn_clear_points)
        
        calib_layout.addLayout(calib_btn_layout)
        
        self.btn_calc_calib = QPushButton("计算标定")
        self.btn_calc_calib.setStyleSheet(self._button_style("#00AA66"))
        self.btn_calc_calib.clicked.connect(self._calculate_calibration)
        calib_layout.addWidget(self.btn_calc_calib)
        
        self.calib_info = QLabel("状态: 未标定")
        self.calib_info.setStyleSheet("color: #FF6666; font: 11px Consolas;")
        calib_layout.addWidget(self.calib_info)
        
        layout.addWidget(calib_group)
        layout.addStretch()
        
        return panel
        
    def _create_center_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        
        # 标题栏
        title_bar = QWidget()
        title_layout = QHBoxLayout(title_bar)
        
        title_label = QLabel("贴内衬机视觉定位系统")
        title_label.setFont(QFont("Consolas", 18, QFont.Bold))
        title_label.setStyleSheet("color: #00FFFF;")
        title_layout.addWidget(title_label)
        
        self.mode_switch = QComboBox()
        self.mode_switch.addItems(["双直线模式(矩形)", "双外形模式(异形)"])
        self.mode_switch.setStyleSheet(self._combo_style())
        self.mode_switch.currentIndexChanged.connect(self._change_mode)
        title_layout.addWidget(self.mode_switch)
        
        layout.addWidget(title_bar)
        
        # 图像画布
        self.image_canvas.setStyleSheet("border: 2px solid #0066CC; border-radius: 10px;")
        layout.addWidget(self.image_canvas, stretch=1)
        
        # 实时数据显示
        data_panel = QGroupBox("实时检测数据")
        data_panel.setStyleSheet(self._group_style())
        data_layout = QGridLayout(data_panel)
        
        data_layout.addWidget(QLabel("X(像素):"), 0, 0)
        self.data_x_pixel = QLabel("--")
        self.data_x_pixel.setStyleSheet("color: #00FF96; font: bold 12px Consolas;")
        data_layout.addWidget(self.data_x_pixel, 0, 1)
        
        data_layout.addWidget(QLabel("Y(像素):"), 0, 2)
        self.data_y_pixel = QLabel("--")
        self.data_y_pixel.setStyleSheet("color: #00FF96; font: bold 12px Consolas;")
        data_layout.addWidget(self.data_y_pixel, 0, 3)
        
        data_layout.addWidget(QLabel("X(mm):"), 1, 0)
        self.data_x_mm = QLabel("--")
        self.data_x_mm.setStyleSheet("color: #00FFFF; font: bold 12px Consolas;")
        data_layout.addWidget(self.data_x_mm, 1, 1)
        
        data_layout.addWidget(QLabel("Y(mm):"), 1, 2)
        self.data_y_mm = QLabel("--")
        self.data_y_mm.setStyleSheet("color: #00FFFF; font: bold 12px Consolas;")
        data_layout.addWidget(self.data_y_mm, 1, 3)
        
        data_layout.addWidget(QLabel("Z(mm):"), 2, 0)
        self.data_z = QLabel("0.00")
        self.data_z.setStyleSheet("color: #FF9600; font: bold 12px Consolas;")
        data_layout.addWidget(self.data_z, 2, 1)
        
        data_layout.addWidget(QLabel("角度(A):"), 2, 2)
        self.data_angle = QLabel("--")
        self.data_angle.setStyleSheet("color: #FF9600; font: bold 12px Consolas;")
        data_layout.addWidget(self.data_angle, 2, 3)
        
        data_layout.addWidget(QLabel("匹配分数:"), 3, 0)
        self.data_score = QLabel("--")
        self.data_score.setStyleSheet("color: #AA00AA; font: bold 12px Consolas;")
        data_layout.addWidget(self.data_score, 3, 1)
        
        data_layout.addWidget(QLabel("状态:"), 3, 2)
        self.data_status = QLabel("等待")
        self.data_status.setStyleSheet("color: #FF6666; font: bold 12px Consolas;")
        data_layout.addWidget(self.data_status, 3, 3)
        
        layout.addWidget(data_panel)
        
        return panel
        
    def _create_right_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        
        # 模式切换标签页
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet(self._tab_style())
        
        # 直线模板页
        self.line_tab = QWidget()
        self._setup_line_template_tab()
        self.tab_widget.addTab(self.line_tab, "直线模板")
        
        # 外形模板页
        self.shape_tab = QWidget()
        self._setup_shape_template_tab()
        self.tab_widget.addTab(self.shape_tab, "外形模板")
        
        # 贴料参数页
        self.paste_tab = QWidget()
        self._setup_paste_params_tab()
        self.tab_widget.addTab(self.paste_tab, "贴料参数")
        
        layout.addWidget(self.tab_widget)
        layout.addStretch()
        
        return panel
        
    def _setup_line_template_tab(self):
        layout = QVBoxLayout(self.line_tab)
        layout.setSpacing(10)
        
        # 左模板设置
        left_group = QGroupBox("左模板设置")
        left_group.setStyleSheet(self._group_style())
        left_layout = QVBoxLayout(left_group)
        
        self.btn_clear_left_line = QPushButton("清除左模板")
        self.btn_clear_left_line.setStyleSheet(self._button_style("#FF6666"))
        self.btn_clear_left_line.clicked.connect(self._clear_left_line)
        left_layout.addWidget(self.btn_clear_left_line)
        
        self.btn_draw_left_line = QPushButton("画左模板")
        self.btn_draw_left_line.setStyleSheet(self._button_style("#00AA66"))
        self.btn_draw_left_line.clicked.connect(lambda: self.image_canvas.start_drawing("line"))
        left_layout.addWidget(self.btn_draw_left_line)
        
        self.btn_left_roi = QPushButton("左监测区域")
        self.btn_left_roi.setStyleSheet(self._button_style())
        self.btn_left_roi.clicked.connect(lambda: self.image_canvas.start_drawing("roi"))
        left_layout.addWidget(self.btn_left_roi)
        
        self.left_line_info = QLabel("状态: 未设置")
        self.left_line_info.setStyleSheet("color: #00FF96; font: 10px Consolas;")
        left_layout.addWidget(self.left_line_info)
        
        layout.addWidget(left_group)
        
        # 右模板设置
        right_group = QGroupBox("右模板设置")
        right_group.setStyleSheet(self._group_style())
        right_layout = QVBoxLayout(right_group)
        
        self.btn_clear_right_line = QPushButton("清除右模板")
        self.btn_clear_right_line.setStyleSheet(self._button_style("#FF6666"))
        self.btn_clear_right_line.clicked.connect(self._clear_right_line)
        right_layout.addWidget(self.btn_clear_right_line)
        
        self.btn_draw_right_line = QPushButton("画右模板")
        self.btn_draw_right_line.setStyleSheet(self._button_style("#00AA66"))
        self.btn_draw_right_line.clicked.connect(lambda: self.image_canvas.start_drawing("line"))
        right_layout.addWidget(self.btn_draw_right_line)
        
        self.btn_right_roi = QPushButton("右监测区域")
        self.btn_right_roi.setStyleSheet(self._button_style())
        self.btn_right_roi.clicked.connect(lambda: self.image_canvas.start_drawing("roi"))
        right_layout.addWidget(self.btn_right_roi)
        
        self.right_line_info = QLabel("状态: 未设置")
        self.right_line_info.setStyleSheet("color: #FF9600; font: 10px Consolas;")
        right_layout.addWidget(self.right_line_info)
        
        layout.addWidget(right_group)
        
        # 操作按钮
        action_group = QGroupBox("操作")
        action_group.setStyleSheet(self._group_style())
        action_layout = QVBoxLayout(action_group)
        
        self.btn_save_base = QPushButton("面纸A基准")
        self.btn_save_base.setStyleSheet(self._button_style("#0088FF"))
        self.btn_save_base.clicked.connect(self._save_base_template)
        action_layout.addWidget(self.btn_save_base)
        
        self.btn_test_line = QPushButton("手动测试")
        self.btn_test_line.setStyleSheet(self._button_style("#AA00AA"))
        self.btn_test_line.clicked.connect(self._test_line_detection)
        action_layout.addWidget(self.btn_test_line)
        
        self.btn_save_line_template = QPushButton("保存模板")
        self.btn_save_line_template.setStyleSheet(self._button_style())
        self.btn_save_line_template.clicked.connect(self._save_line_template)
        action_layout.addWidget(self.btn_save_line_template)
        
        self.btn_load_line_template = QPushButton("加载模板")
        self.btn_load_line_template.setStyleSheet(self._button_style())
        self.btn_load_line_template.clicked.connect(self._load_line_template)
        action_layout.addWidget(self.btn_load_line_template)
        
        layout.addWidget(action_group)
        layout.addStretch()
        
    def _setup_shape_template_tab(self):
        layout = QVBoxLayout(self.shape_tab)
        layout.setSpacing(10)
        
        # 参数设置
        param_group = QGroupBox("轮廓参数")
        param_group.setStyleSheet(self._group_style())
        param_layout = QVBoxLayout(param_group)
        
        param_layout.addWidget(QLabel("Canny低阈值:"))
        self.canny_low_slider = QSlider(Qt.Horizontal)
        self.canny_low_slider.setRange(10, 200)
        self.canny_low_slider.setValue(50)
        self.canny_low_slider.setStyleSheet(self._slider_style())
        param_layout.addWidget(self.canny_low_slider)
        
        param_layout.addWidget(QLabel("Canny高阈值:"))
        self.canny_high_slider = QSlider(Qt.Horizontal)
        self.canny_high_slider.setRange(50, 300)
        self.canny_high_slider.setValue(150)
        self.canny_high_slider.setStyleSheet(self._slider_style())
        param_layout.addWidget(self.canny_high_slider)
        
        param_layout.addWidget(QLabel("匹配阈值:"))
        self.match_threshold_spin = QDoubleSpinBox()
        self.match_threshold_spin.setRange(0.1, 1.0)
        self.match_threshold_spin.setValue(0.7)
        self.match_threshold_spin.setStyleSheet(self._spinbox_style())
        param_layout.addWidget(self.match_threshold_spin)
        
        layout.addWidget(param_group)
        
        # 左外形模板
        left_group = QGroupBox("左外形模板")
        left_group.setStyleSheet(self._group_style())
        left_layout = QVBoxLayout(left_group)
        
        self.btn_select_left_shape = QPushButton("框选左特征区")
        self.btn_select_left_shape.setStyleSheet(self._button_style("#00AA66"))
        self.btn_select_left_shape.clicked.connect(lambda: self.image_canvas.start_drawing("template"))
        left_layout.addWidget(self.btn_select_left_shape)
        
        self.btn_clear_left_shape = QPushButton("清除左模板")
        self.btn_clear_left_shape.setStyleSheet(self._button_style("#FF6666"))
        self.btn_clear_left_shape.clicked.connect(self._clear_left_shape)
        left_layout.addWidget(self.btn_clear_left_shape)
        
        self.left_shape_info = QLabel("状态: 未设置")
        self.left_shape_info.setStyleSheet("color: #00FF96; font: 10px Consolas;")
        left_layout.addWidget(self.left_shape_info)
        
        layout.addWidget(left_group)
        
        # 右外形模板
        right_group = QGroupBox("右外形模板")
        right_group.setStyleSheet(self._group_style())
        right_layout = QVBoxLayout(right_group)
        
        self.btn_select_right_shape = QPushButton("框选右特征区")
        self.btn_select_right_shape.setStyleSheet(self._button_style("#00AA66"))
        self.btn_select_right_shape.clicked.connect(lambda: self.image_canvas.start_drawing("template"))
        right_layout.addWidget(self.btn_select_right_shape)
        
        self.btn_clear_right_shape = QPushButton("清除右模板")
        self.btn_clear_right_shape.setStyleSheet(self._button_style("#FF6666"))
        self.btn_clear_right_shape.clicked.connect(self._clear_right_shape)
        right_layout.addWidget(self.btn_clear_right_shape)
        
        self.right_shape_info = QLabel("状态: 未设置")
        self.right_shape_info.setStyleSheet("color: #FF9600; font: 10px Consolas;")
        right_layout.addWidget(self.right_shape_info)
        
        layout.addWidget(right_group)
        
        # 操作按钮
        action_group = QGroupBox("操作")
        action_group.setStyleSheet(self._group_style())
        action_layout = QVBoxLayout(action_group)
        
        self.btn_test_shape = QPushButton("测试识别")
        self.btn_test_shape.setStyleSheet(self._button_style("#AA00AA"))
        self.btn_test_shape.clicked.connect(self._test_shape_detection)
        action_layout.addWidget(self.btn_test_shape)
        
        self.btn_save_shape_template = QPushButton("保存异形模板")
        self.btn_save_shape_template.setStyleSheet(self._button_style())
        self.btn_save_shape_template.clicked.connect(self._save_shape_template)
        action_layout.addWidget(self.btn_save_shape_template)
        
        layout.addWidget(action_group)
        layout.addStretch()
        
    def _setup_paste_params_tab(self):
        layout = QVBoxLayout(self.paste_tab)
        layout.setSpacing(10)
        
        # 偏移补偿
        offset_group = QGroupBox("偏移补偿")
        offset_group.setStyleSheet(self._group_style())
        offset_layout = QGridLayout(offset_group)
        
        offset_layout.addWidget(QLabel("X偏移(mm):"), 0, 0)
        self.offset_x_spin = QDoubleSpinBox()
        self.offset_x_spin.setRange(-100, 100)
        self.offset_x_spin.setSingleStep(0.1)
        self.offset_x_spin.setStyleSheet(self._spinbox_style())
        offset_layout.addWidget(self.offset_x_spin, 0, 1)
        
        offset_layout.addWidget(QLabel("Y偏移(mm):"), 1, 0)
        self.offset_y_spin = QDoubleSpinBox()
        self.offset_y_spin.setRange(-100, 100)
        self.offset_y_spin.setSingleStep(0.1)
        self.offset_y_spin.setStyleSheet(self._spinbox_style())
        offset_layout.addWidget(self.offset_y_spin, 1, 1)
        
        offset_layout.addWidget(QLabel("Z偏移(mm):"), 2, 0)
        self.offset_z_spin = QDoubleSpinBox()
        self.offset_z_spin.setRange(-100, 100)
        self.offset_z_spin.setSingleStep(0.1)
        self.offset_z_spin.setStyleSheet(self._spinbox_style())
        offset_layout.addWidget(self.offset_z_spin, 2, 1)
        
        offset_layout.addWidget(QLabel("A偏移(°):"), 3, 0)
        self.offset_a_spin = QDoubleSpinBox()
        self.offset_a_spin.setRange(-180, 180)
        self.offset_a_spin.setSingleStep(0.1)
        self.offset_a_spin.setStyleSheet(self._spinbox_style())
        offset_layout.addWidget(self.offset_a_spin, 3, 1)
        
        layout.addWidget(offset_group)
        
        # 阈值设置
        threshold_group = QGroupBox("阈值设置")
        threshold_group.setStyleSheet(self._group_style())
        threshold_layout = QGridLayout(threshold_group)
        
        threshold_layout.addWidget(QLabel("匹配分数阈值:"), 0, 0)
        self.score_threshold_spin = QDoubleSpinBox()
        self.score_threshold_spin.setRange(0.1, 1.0)
        self.score_threshold_spin.setValue(0.7)
        self.score_threshold_spin.setStyleSheet(self._spinbox_style())
        threshold_layout.addWidget(self.score_threshold_spin, 0, 1)
        
        threshold_layout.addWidget(QLabel("X报警限值(mm):"), 1, 0)
        self.alarm_x_spin = QDoubleSpinBox()
        self.alarm_x_spin.setRange(0, 50)
        self.alarm_x_spin.setValue(5)
        self.alarm_x_spin.setStyleSheet(self._spinbox_style())
        threshold_layout.addWidget(self.alarm_x_spin, 1, 1)
        
        threshold_layout.addWidget(QLabel("Y报警限值(mm):"), 2, 0)
        self.alarm_y_spin = QDoubleSpinBox()
        self.alarm_y_spin.setRange(0, 50)
        self.alarm_y_spin.setValue(5)
        self.alarm_y_spin.setStyleSheet(self._spinbox_style())
        threshold_layout.addWidget(self.alarm_y_spin, 2, 1)
        
        threshold_layout.addWidget(QLabel("A报警限值(°):"), 3, 0)
        self.alarm_a_spin = QDoubleSpinBox()
        self.alarm_a_spin.setRange(0, 45)
        self.alarm_a_spin.setValue(3)
        self.alarm_a_spin.setStyleSheet(self._spinbox_style())
        threshold_layout.addWidget(self.alarm_a_spin, 3, 1)
        
        layout.addWidget(threshold_group)
        
        # 时序设置
        timing_group = QGroupBox("时序设置")
        timing_group.setStyleSheet(self._group_style())
        timing_layout = QGridLayout(timing_group)
        
        timing_layout.addWidget(QLabel("识别超时(ms):"), 0, 0)
        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(100, 10000)
        self.timeout_spin.setValue(3000)
        self.timeout_spin.setStyleSheet(self._spinbox_style())
        timing_layout.addWidget(self.timeout_spin, 0, 1)
        
        timing_layout.addWidget(QLabel("拍照延时(ms):"), 1, 0)
        self.photo_delay_spin = QSpinBox()
        self.photo_delay_spin.setRange(0, 1000)
        self.photo_delay_spin.setValue(100)
        self.photo_delay_spin.setStyleSheet(self._spinbox_style())
        timing_layout.addWidget(self.photo_delay_spin, 1, 1)
        
        timing_layout.addWidget(QLabel("真空延时(ms):"), 2, 0)
        self.vacuum_delay_spin = QSpinBox()
        self.vacuum_delay_spin.setRange(0, 1000)
        self.vacuum_delay_spin.setValue(200)
        self.vacuum_delay_spin.setStyleSheet(self._spinbox_style())
        timing_layout.addWidget(self.vacuum_delay_spin, 2, 1)
        
        layout.addWidget(timing_group)
        
        # 失败处理
        fail_group = QGroupBox("失败处理")
        fail_group.setStyleSheet(self._group_style())
        fail_layout = QVBoxLayout(fail_group)
        
        fail_layout.addWidget(QLabel("重试次数:"))
        self.retry_count_spin = QSpinBox()
        self.retry_count_spin.setRange(0, 10)
        self.retry_count_spin.setValue(3)
        self.retry_count_spin.setStyleSheet(self._spinbox_style())
        fail_layout.addWidget(self.retry_count_spin)
        
        fail_layout.addWidget(QLabel("失败模式:"))
        self.fail_mode_combo = QComboBox()
        self.fail_mode_combo.addItems(["报警停止", "自动重试", "跳过继续"])
        self.fail_mode_combo.setStyleSheet(self._combo_style())
        fail_layout.addWidget(self.fail_mode_combo)
        
        layout.addWidget(fail_group)
        
        # 保存/加载
        btn_layout = QHBoxLayout()
        self.btn_save_params = QPushButton("保存参数")
        self.btn_save_params.setStyleSheet(self._button_style())
        self.btn_save_params.clicked.connect(self._save_paste_params)
        btn_layout.addWidget(self.btn_save_params)
        
        self.btn_load_params = QPushButton("加载参数")
        self.btn_load_params.setStyleSheet(self._button_style())
        self.btn_load_params.clicked.connect(self._load_paste_params)
        btn_layout.addWidget(self.btn_load_params)
        
        layout.addLayout(btn_layout)
        layout.addStretch()
        
    def _setup_statusbar(self):
        status_bar = QStatusBar()
        self.setStatusBar(status_bar)
        
        self.status_light = QLabel()
        self.status_light.setStyleSheet("background-color: #FF6666; border-radius: 8px;")
        self.status_light.setFixedSize(16, 16)
        status_bar.addWidget(self.status_light)
        
        self.status_label = QLabel("系统就绪")
        self.status_label.setStyleSheet("color: #00FFFF; font: 11px Consolas;")
        status_bar.addWidget(self.status_label)
        
        self.fps_label = QLabel("FPS: 0")
        self.fps_label.setStyleSheet("color: #00FF96; font: 11px Consolas;")
        status_bar.addPermanentWidget(self.fps_label)
        
    def _setup_connections(self):
        self.image_canvas.roi_selected.connect(self._on_roi_selected)
        self.image_canvas.line_drawn.connect(self._on_line_drawn)
        self.image_canvas.template_selected.connect(self._on_template_selected)
        
    # ==================== 样式方法 ====================
    def _group_style(self):
        return """
            QGroupBox {
                color: #00FFFF;
                font: bold 11px Consolas;
                border: 2px solid #0066CC;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """
        
    def _button_style(self, color="#0088FF"):
        return f"""
            QPushButton {{
                background-color: rgba(0, 50, 100, 150);
                color: #00FFFF;
                border: 2px solid {color};
                border-radius: 5px;
                padding: 8px;
                font: bold 10px Consolas;
            }}
            QPushButton:hover {{
                background-color: rgba(0, 100, 200, 200);
            }}
        """
        
    def _combo_style(self):
        return """
            QComboBox {
                background-color: rgba(0, 50, 100, 150);
                color: #00FFFF;
                border: 2px solid #0088FF;
                border-radius: 5px;
                padding: 5px;
                font: 10px Consolas;
            }
            QComboBox QAbstractItemView {
                background-color: rgba(10, 20, 40, 230);
                color: #00FFFF;
            }
        """
        
    def _table_style(self):
        return """
            QTableWidget {
                background-color: rgba(10, 20, 40, 200);
                color: #00FFFF;
                border: 1px solid #0066CC;
                font: 10px Consolas;
            }
            QHeaderView::section {
                background-color: rgba(0, 50, 100, 200);
                color: #00FFFF;
                border: 1px solid #0066CC;
                padding: 3px;
            }
        """
        
    def _slider_style(self):
        return """
            QSlider::groove:horizontal {
                background: rgba(0, 50, 100, 150);
                height: 8px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #00FFFF;
                width: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }
        """
        
    def _spinbox_style(self):
        return """
            QSpinBox, QDoubleSpinBox {
                background-color: rgba(0, 50, 100, 150);
                color: #00FFFF;
                border: 2px solid #0088FF;
                border-radius: 5px;
                padding: 5px;
            }
        """
        
    def _tab_style(self):
        return """
            QTabWidget::pane {
                border: 2px solid #0066CC;
                border-radius: 8px;
                background-color: rgba(10, 20, 40, 200);
            }
            QTabBar::tab {
                background-color: rgba(0, 50, 100, 150);
                color: #00FFFF;
                border: 2px solid #0066CC;
                border-radius: 5px;
                padding: 8px 15px;
                margin-right: 5px;
            }
            QTabBar::tab:selected {
                background-color: rgba(0, 100, 200, 200);
                border-color: #00FFFF;
            }
        """
        
    # ==================== 摄像头控制 ====================
    def _detect_cameras(self):
        self.camera_combo.clear()
        self.camera_combo.addItem("请选择摄像头")
        
        for i in range(10):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.camera_combo.addItem(f"摄像头 {i} ({w}x{h})")
                cap.release()
                
    def _toggle_camera(self):
        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.stop()
            self.btn_open_cam.setText("📷 打开摄像头")
            self.status_label.setText("摄像头已关闭")
            self.status_light.setStyleSheet("background-color: #FF6666; border-radius: 8px;")
        else:
            if self.camera_combo.currentIndex() == 0:
                QMessageBox.warning(self, "警告", "请先选择摄像头")
                return
                
            camera_index = self.camera_combo.currentIndex() - 1
            res = self.resolution_combo.currentText().split('x')
            width, height = int(res[0]), int(res[1])
            
            self.camera_thread = CameraThread(camera_index, (width, height))
            self.camera_thread.frame_ready.connect(self._on_frame_ready)
            self.camera_thread.error_occurred.connect(self._on_camera_error)
            self.camera_thread.start()
            
            self.btn_open_cam.setText("⏹ 关闭摄像头")
            self.status_label.setText("摄像头运行中")
            self.status_light.setStyleSheet("background-color: #00FF96; border-radius: 8px;")
            
    def _on_frame_ready(self, frame):
        self.image_canvas.set_frame(frame)
        self.vision_processor.current_frame = frame
        
    def _on_camera_error(self, error):
        QMessageBox.critical(self, "摄像头错误", error)
        
    # ==================== 标定功能 ====================
    def _add_calib_point(self):
        row = self.calib_table.rowCount()
        self.calib_table.insertRow(row)
        self.calib_table.setItem(row, 0, QTableWidgetItem(str(row+1)))
        self.calib_table.setItem(row, 1, QTableWidgetItem("0.0"))
        self.calib_table.setItem(row, 2, QTableWidgetItem("0.0"))
        self.calib_table.setItem(row, 3, QTableWidgetItem("0.0"))
        
    def _clear_calib_points(self):
        self.calib_table.setRowCount(0)
        
    def _calculate_calibration(self):
        if self.calib_table.rowCount() < 2:
            QMessageBox.warning(self, "警告", "至少需要2个标定点")
            return
            
        try:
            px_list = []
            phys_list = []
            
            for row in range(self.calib_table.rowCount()):
                px = float(self.calib_table.item(row, 1).text())
                py = float(self.calib_table.item(row, 2).text())
                phys = float(self.calib_table.item(row, 3).text())
                px_list.append((px, py))
                phys_list.append(phys)
                
            if len(px_list) >= 2:
                dx_pix = abs(px_list[1][0] - px_list[0][0])
                dx_phys = abs(phys_list[1] - phys_list[0])
                if dx_phys > 0:
                    self.calibration_data.pixel_per_mm_x = dx_pix / dx_phys
                    
                dy_pix = abs(px_list[1][1] - px_list[0][1])
                dy_phys = abs(phys_list[1] - phys_list[0])
                if dy_phys > 0:
                    self.calibration_data.pixel_per_mm_y = dy_pix / dy_phys
                    
            self.calibration_data.is_calibrated = True
            self.vision_processor.calibration_data = self.calibration_data
            
            self.calib_info.setText(f"状态: 已标定 ✓\n像素/mm: X={self.calibration_data.pixel_per_mm_x:.4f}, Y={self.calibration_data.pixel_per_mm_y:.4f}")
            self.calib_info.setStyleSheet("color: #00FF96; font: 11px Consolas;")
            
            QMessageBox.information(self, "成功", "标定计算完成！")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"计算失败: {str(e)}")
            
    # ==================== 模式切换 ====================
    def _change_mode(self, index):
        self.current_mode = "shape" if index == 1 else "line"
        self.vision_processor.current_mode = self.current_mode
        
        if self.current_mode == "line":
            self.tab_widget.setCurrentIndex(0)
        else:
            self.tab_widget.setCurrentIndex(1)
            
    # ==================== 直线模板 ====================
    def _clear_left_line(self):
        self.image_canvas.left_line = None
        self.left_line_info.setText("状态: 未设置")
        
    def _clear_right_line(self):
        self.image_canvas.right_line = None
        self.right_line_info.setText("状态: 未设置")