#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
贴内衬机视觉定位系统 - 优化版GUI主程序
工业级、科技风、HUD风格界面

优化内容：
1. 添加撤销功能 - 区域和直线选择可撤销
2. 优化手动测试 - 隐藏模板线条，显示识别结果，计算X/Y/Z/A偏移量
3. 调整整体UI字体大小
4. 添加自动保存/加载功能 - 启动时加载上次模板，保存时选择位置
"""

import sys
import os
import math
import json
import time
import cv2
import numpy as np
from collections import deque

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFrame, QGridLayout, QSlider,
    QComboBox, QGroupBox, QTabWidget, QTableWidget,
    QTableWidgetItem, QHeaderView, QSpinBox, QDoubleSpinBox,
    QFileDialog, QMessageBox, QStatusBar, QTextEdit, QCheckBox,
    QProgressBar, QScrollArea, QRadioButton, QButtonGroup, QSplitter
)
from PyQt5.QtCore import (
    Qt, pyqtSignal, QThread, QRect, QTimer, QPointF
)
from PyQt5.QtGui import (
    QPainter, QColor, QPen, QBrush, QFont, QLinearGradient,
    QRadialGradient, QPolygonF, QImage, QPixmap, QCursor
)

from vision_utils_v2 import (
    CalibrationData, DualLineTemplate, ShapeTemplate, DetectionResult,
    CameraCalibration, VisionAlgorithm, TemplateManager, CameraUtils
)
from simulation_module import (
    SimulationConfig, SceneGenerator, SimulatedCameraThread,
    WorkConditionPresets, SimulationTestEngine, TestCaseResult,
    BatchTestThread
)


# 默认配置路径
DEFAULT_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_FILE = os.path.join(DEFAULT_CONFIG_DIR, "system_config.json")
DEFAULT_LINE_TEMPLATE_FILE = os.path.join(DEFAULT_CONFIG_DIR, "last_line_template.json")
DEFAULT_SHAPE_TEMPLATE_FILE = os.path.join(DEFAULT_CONFIG_DIR, "last_shape_template.json")
DEFAULT_CALIB_FILE = os.path.join(DEFAULT_CONFIG_DIR, "last_calibration.json")


class NeonStyle:
    """霓虹风格颜色定义"""
    NEON_BLUE = QColor(0, 150, 255)
    NEON_CYAN = QColor(0, 255, 255)
    NEON_PURPLE = QColor(150, 0, 255)
    NEON_GREEN = QColor(0, 255, 150)
    NEON_RED = QColor(255, 50, 100)
    NEON_ORANGE = QColor(255, 150, 0)
    NEON_YELLOW = QColor(255, 255, 0)
    DARK_BG = QColor(5, 10, 25)
    PANEL_BG = QColor(10, 20, 40, 200)


class CameraThread(QThread):
    """摄像头采集线程"""
    frame_ready = pyqtSignal(np.ndarray)
    error_occurred = pyqtSignal(str)

    def __init__(self, camera_index=0, resolution=(1280, 720)):
        super().__init__()
        self.camera_index = camera_index
        self.resolution = resolution
        self.running = False
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


class ImageCanvas(QWidget):
    """
    图像画布控件
    支持绘制直线、矩形、轮廓等，带撤销功能
    支持框选区域时实时预览检测到的边缘线
    """
    roi_selected = pyqtSignal(tuple)
    line_drawn = pyqtSignal(tuple)
    contour_selected = pyqtSignal(tuple)
    contour_drawn = pyqtSignal(np.ndarray)  # 新增：手绘轮廓完成信号
    auto_line_roi_selected = pyqtSignal(tuple)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.setMouseTracking(True)

        # 图像数据
        self.current_frame = None
        self.display_frame = None
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0

        # 绘制模式
        self.drawing_mode = None
        self.drawing_start = None
        self.drawing_end = None

        # 手绘轮廓点（用于实时绘制）
        self.contour_points = []

        # 双直线模板数据
        self.left_line_long = None
        self.left_line_short = None
        self.right_line_long = None
        self.right_line_short = None
        self.left_roi = None
        self.right_roi = None

        # 异形模板数据
        self.left_contour = None
        self.right_contour = None
        self.left_contour_roi = None
        self.right_contour_roi = None

        # 检测结果
        self.detected_lines = []
        self.detected_contours = []
        self.detected_rois = []
        self.detected_center = None
        self.detected_angle = 0.0

        # 预览检测结果（用于框选时实时显示）
        self.preview_lines = []
        self.preview_rect = None

        # 撤销栈
        self.undo_stack = deque(maxlen=20)

        # 测试模式标志
        self.test_mode = False

        # 动画效果
        self.scan_phase = 0
        self.pulse_phase = 0

    def save_state(self, action_type, old_value, description):
        """保存撤销状态"""
        self.undo_stack.append({
            'type': action_type,
            'old_value': old_value,
            'description': description
        })

    def undo(self):
        """撤销上一步操作"""
        if not self.undo_stack:
            return None

        state = self.undo_stack.pop()
        action_type = state['type']
        old_value = state['old_value']
        description = state['description']

        if action_type == 'line_long':
            if state['old_value'] is None:
                self.left_line_long = None
            else:
                self.left_line_long = state['old_value']
        elif action_type == 'line_short':
            if state['old_value'] is None:
                self.left_line_short = None
            else:
                self.left_line_short = state['old_value']
        elif action_type == 'right_line_long':
            if state['old_value'] is None:
                self.right_line_long = None
            else:
                self.right_line_long = state['old_value']
        elif action_type == 'right_line_short':
            if state['old_value'] is None:
                self.right_line_short = None
            else:
                self.right_line_short = state['old_value']
        elif action_type == 'left_roi':
            self.left_roi = state['old_value']
        elif action_type == 'right_roi':
            self.right_roi = state['old_value']
        elif action_type == 'left_contour':
            self.left_contour = state['old_value']
        elif action_type == 'right_contour':
            self.right_contour = state['old_value']
        elif action_type == 'left_contour_roi':
            self.left_contour_roi = state['old_value']
        elif action_type == 'right_contour_roi':
            self.right_contour_roi = state['old_value']

        self.update()
        return description

    def can_undo(self):
        """检查是否可以撤销"""
        return len(self.undo_stack) > 0

    def set_frame(self, frame):
        """设置当前帧"""
        self.current_frame = frame.copy()
        self._update_display()
        self.update()

    def _update_display(self):
        """更新显示"""
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
        """绘制事件"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 绘制背景
        self._draw_background(painter)

        # 绘制图像
        if self.display_frame is not None:
            h, w = self.display_frame.shape[:2]
            q_img = QImage(self.display_frame.data, w, h, 3*w, QImage.Format_BGR888)
            painter.drawImage(self.offset_x, self.offset_y, q_img)
        else:
            self._draw_placeholder(painter)

        # 绘制扫描效果
        self._draw_scan_effect(painter)

        # 非测试模式下绘制ROI和模板
        if not self.test_mode:
            self._draw_rois(painter)
            self._draw_template_lines(painter)
            self._draw_template_contours(painter)

        # 绘制检测结果（测试模式下也显示）
        self._draw_detection_results(painter)

        # 绘制十字准心
        self._draw_crosshair(painter)

        # 绘制绘制预览
        if self.drawing_mode:
            self._draw_drawing_preview(painter)

        # 绘制HUD信息
        self._draw_hud_overlay(painter)

        # 绘制角落装饰
        self._draw_corner_decorations(painter)

        # 更新动画相位
        self.scan_phase += 0.05
        self.pulse_phase += 0.1

    def _draw_background(self, painter):
        """绘制背景"""
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0, QColor(5, 10, 30))
        gradient.setColorAt(0.5, QColor(10, 20, 50))
        gradient.setColorAt(1, QColor(5, 15, 35))
        painter.fillRect(self.rect(), gradient)

    def _draw_placeholder(self, painter):
        """绘制占位符"""
        cx, cy = self.width()//2, self.height()//2
        gradient = QRadialGradient(cx, cy, 200)
        gradient.setColorAt(0, QColor(0, 100, 200, 50))
        gradient.setColorAt(1, QColor(0, 20, 80, 10))
        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(cx-200, cy-200, 400, 400)
        pen = QPen(NeonStyle.NEON_BLUE, 2)
        painter.setPen(pen)
        painter.setFont(QFont("Consolas", 20, QFont.Bold))
        painter.drawText(self.rect(), Qt.AlignCenter, "等待摄像头连接...")

    def _draw_scan_effect(self, painter):
        """绘制扫描效果"""
        scan_x = int((self.scan_phase * 100) % self.width())
        gradient = QLinearGradient(scan_x-30, 0, scan_x+30, 0)
        gradient.setColorAt(0, QColor(0, 255, 255, 0))
        gradient.setColorAt(0.5, QColor(0, 255, 255, 40))
        gradient.setColorAt(1, QColor(0, 255, 255, 0))
        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.NoPen)
        painter.drawRect(scan_x-30, 0, 60, self.height())

    def _draw_rois(self, painter):
        """绘制ROI区域"""
        painter.setOpacity(0.6)

        if self.left_roi:
            x1, y1, x2, y2 = self._scale_rect(self.left_roi)
            pen = QPen(NeonStyle.NEON_GREEN, 2, Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(QBrush(QColor(0, 255, 150, 30)))
            painter.drawRect(x1, y1, x2-x1, y2-y1)
            painter.setFont(QFont("Consolas", 12, QFont.Bold))
            painter.drawText(x1+5, y1+20, "左ROI")

        if self.right_roi:
            x1, y1, x2, y2 = self._scale_rect(self.right_roi)
            pen = QPen(NeonStyle.NEON_ORANGE, 2, Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(QBrush(QColor(255, 150, 0, 30)))
            painter.drawRect(x1, y1, x2-x1, y2-y1)
            painter.drawText(x1+5, y1+20, "右ROI")

        if self.left_contour_roi:
            x1, y1, x2, y2 = self._scale_rect(self.left_contour_roi)
            pen = QPen(NeonStyle.NEON_GREEN, 2, Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(QBrush(QColor(0, 255, 150, 30)))
            painter.drawRect(x1, y1, x2-x1, y2-y1)
            painter.drawText(x1+5, y1+20, "左轮廓ROI")

        if self.right_contour_roi:
            x1, y1, x2, y2 = self._scale_rect(self.right_contour_roi)
            pen = QPen(NeonStyle.NEON_ORANGE, 2, Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(QBrush(QColor(255, 150, 0, 30)))
            painter.drawRect(x1, y1, x2-x1, y2-y1)
            painter.drawText(x1+5, y1+20, "右轮廓ROI")

        painter.setOpacity(1.0)

    def _draw_template_lines(self, painter):
        """绘制模板直线"""
        # 左侧长边
        if self.left_line_long:
            x1, y1 = self._scale_point(self.left_line_long[0])
            x2, y2 = self._scale_point(self.left_line_long[1])
            pen = QPen(NeonStyle.NEON_GREEN, 4)
            painter.setPen(pen)
            painter.drawLine(x1, y1, x2, y2)
            mid_x, mid_y = (x1+x2)//2, (y1+y2)//2
            painter.setFont(QFont("Consolas", 12, QFont.Bold))
            painter.drawText(mid_x+5, mid_y-5, "左长边")

        # 左侧短边
        if self.left_line_short:
            x1, y1 = self._scale_point(self.left_line_short[0])
            x2, y2 = self._scale_point(self.left_line_short[1])
            pen = QPen(NeonStyle.NEON_GREEN, 4)
            painter.setPen(pen)
            painter.drawLine(x1, y1, x2, y2)
            mid_x, mid_y = (x1+x2)//2, (y1+y2)//2
            painter.drawText(mid_x+5, mid_y-5, "左短边")

        # 右侧长边
        if self.right_line_long:
            x1, y1 = self._scale_point(self.right_line_long[0])
            x2, y2 = self._scale_point(self.right_line_long[1])
            pen = QPen(NeonStyle.NEON_ORANGE, 4)
            painter.setPen(pen)
            painter.drawLine(x1, y1, x2, y2)
            mid_x, mid_y = (x1+x2)//2, (y1+y2)//2
            painter.setFont(QFont("Consolas", 12, QFont.Bold))
            painter.drawText(mid_x+5, mid_y-5, "右长边")

        # 右侧短边
        if self.right_line_short:
            x1, y1 = self._scale_point(self.right_line_short[0])
            x2, y2 = self._scale_point(self.right_line_short[1])
            pen = QPen(NeonStyle.NEON_ORANGE, 4)
            painter.setPen(pen)
            painter.drawLine(x1, y1, x2, y2)
            mid_x, mid_y = (x1+x2)//2, (y1+y2)//2
            painter.drawText(mid_x+5, mid_y-5, "右短边")

    def _draw_template_contours(self, painter):
        """绘制模板轮廓"""
        if self.left_contour is not None and len(self.left_contour) >= 3:
            self._draw_single_contour(painter, self.left_contour, NeonStyle.NEON_GREEN, "左轮廓")

        if self.right_contour is not None and len(self.right_contour) >= 3:
            self._draw_single_contour(painter, self.right_contour, NeonStyle.NEON_ORANGE, "右轮廓")

    def _draw_single_contour(self, painter, contour, color, label):
        """绘制单个轮廓（不闭合）"""
        points = [self._scale_point((int(p[0][0]), int(p[0][1]))) for p in contour]
        if len(points) < 2:
            return
        
        # 绘制折线（不闭合），不填充
        polyline = QPolygonF([QPointF(x, y) for x, y in points])
        pen = QPen(color, 3, Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawPolyline(polyline)

        if len(points) > 0:
            cx = sum(p[0] for p in points) / len(points)
            cy = sum(p[1] for p in points) / len(points)
            painter.setFont(QFont("Consolas", 12, QFont.Bold))
            painter.drawText(int(cx)+5, int(cy)-5, label)

    def _draw_detection_results(self, painter):
        """绘制检测结果"""
        if self.test_mode:
            print(f"调试: 绘制检测结果 - 线条数: {len(self.detected_lines)}, ROI数: {len(self.detected_rois)}")
        
        # 检测到的ROI区域
        for roi, color in self.detected_rois:
            x1, y1, x2, y2 = self._scale_rect(roi)
            pen = QPen(color, 2, Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 30)))
            painter.drawRect(x1, y1, x2-x1, y2-y1)

        # 检测到的直线
        for line, color in self.detected_lines:
            x1, y1 = self._scale_point(line[0])
            x2, y2 = self._scale_point(line[1])
            pen = QPen(color, 5)
            painter.setPen(pen)
            painter.drawLine(x1, y1, x2, y2)

        # 检测到的轮廓
        for contour, color in self.detected_contours:
            if len(contour) >= 3:
                points = [self._scale_point((int(p[0][0]), int(p[0][1]))) for p in contour]
                polygon = QPolygonF([QPointF(x, y) for x, y in points])
                pen = QPen(color, 3)
                painter.setPen(pen)
                painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 60)))
                painter.drawPolygon(polygon)

        # 检测中心
        if self.detected_center:
            cx, cy = self._scale_point(self.detected_center)
            pulse = int(8 * math.sin(self.pulse_phase))
            pen = QPen(NeonStyle.NEON_RED, 3)
            painter.setPen(pen)
            size = 25 + pulse
            painter.drawLine(cx-size, cy, cx-8, cy)
            painter.drawLine(cx+8, cy, cx+size, cy)
            painter.drawLine(cx, cy-size, cx, cy-8)
            painter.drawLine(cx, cy+8, cx, cy+size)
            painter.drawEllipse(cx-10, cy-10, 20, 20)

            # 角度线
            if abs(self.detected_angle) > 0.1:
                angle_rad = math.radians(self.detected_angle)
                length = 80
                dx = length * math.cos(angle_rad)
                dy = length * math.sin(angle_rad)
                pen = QPen(NeonStyle.NEON_YELLOW, 3)
                painter.setPen(pen)
                painter.drawLine(int(cx-dx), int(cy-dy), int(cx+dx), int(cy+dy))
                painter.setFont(QFont("Consolas", 14, QFont.Bold))
                painter.drawText(int(cx+dx)+5, int(cy+dy)-5, f"{self.detected_angle:.1f}°")

    def _draw_crosshair(self, painter):
        """绘制十字准心"""
        cx, cy = self.width()//2, self.height()//2
        size = 30
        gap = 10
        pulse = int(4 * math.sin(self.pulse_phase))
        pen = QPen(NeonStyle.NEON_CYAN, 2)
        painter.setPen(pen)
        painter.drawLine(cx-size-pulse, cy, cx-gap, cy)
        painter.drawLine(cx+gap, cy, cx+size+pulse, cy)
        painter.drawLine(cx, cy-size-pulse, cx, cy-gap)
        painter.drawLine(cx, cy+gap, cx, cy+size+pulse)

    def _draw_drawing_preview(self, painter):
        """绘制绘制预览"""
        if self.drawing_start and self.drawing_end:
            sx, sy = self.drawing_start
            ex, ey = self.drawing_end
            if self.drawing_mode == "roi":
                pen = QPen(NeonStyle.NEON_YELLOW, 3, Qt.DashLine)
                painter.setPen(pen)
                painter.setBrush(QBrush(QColor(255, 255, 0, 60)))
                painter.drawRect(sx, sy, ex-sx, ey-sy)
            elif self.drawing_mode == "line":
                pen = QPen(NeonStyle.NEON_YELLOW, 4)
                painter.setPen(pen)
                painter.drawLine(sx, sy, ex, ey)
            elif self.drawing_mode == "contour":
                # 绘制手绘轮廓轨迹（不是矩形）
                if hasattr(self, 'contour_points') and len(self.contour_points) > 1:
                    pen = QPen(NeonStyle.NEON_PURPLE, 3)
                    painter.setPen(pen)
                    painter.setBrush(Qt.NoBrush)
                    # 将轮廓点转换为屏幕坐标并绘制折线
                    scaled_points = [self._scale_point((p[0], p[1])) for p in self.contour_points]
                    for i in range(len(scaled_points) - 1):
                        painter.drawLine(scaled_points[i][0], scaled_points[i][1],
                                        scaled_points[i+1][0], scaled_points[i+1][1])
                else:
                    # 如果点太少，显示矩形作为提示
                    pen = QPen(NeonStyle.NEON_PURPLE, 3, Qt.DashLine)
                    painter.setPen(pen)
                    painter.setBrush(QBrush(QColor(150, 0, 255, 60)))
                    painter.drawRect(sx, sy, ex-sx, ey-sy)
            elif self.drawing_mode == "auto_line_roi":
                # 框选区域自动检测直线时，显示矩形和预览线条
                pen = QPen(NeonStyle.NEON_CYAN, 3, Qt.DashLine)
                painter.setPen(pen)
                painter.setBrush(QBrush(QColor(0, 255, 255, 40)))
                painter.drawRect(sx, sy, ex-sx, ey-sy)
                
                # 绘制预览线条
                if self.preview_lines:
                    for line in self.preview_lines:
                        x1, y1 = self._scale_point(line[0])
                        x2, y2 = self._scale_point(line[1])
                        pen = QPen(QColor(0, 255, 255), 3, Qt.DotLine)
                        painter.setPen(pen)
                        painter.drawLine(x1, y1, x2, y2)

    def _draw_hud_overlay(self, painter):
        """绘制HUD信息"""
        pen = QPen(NeonStyle.NEON_CYAN, 1)
        painter.setPen(pen)
        painter.setFont(QFont("Consolas", 14, QFont.Bold))
        mode_text = "测试模式" if self.test_mode else (self.drawing_mode or "观察")
        painter.drawText(20, 30, f"模式: {mode_text}")
        if self.current_frame is not None:
            h, w = self.current_frame.shape[:2]
            painter.drawText(20, 55, f"分辨率: {w}x{h}")

    def _draw_corner_decorations(self, painter):
        """绘制角落装饰"""
        pen = QPen(NeonStyle.NEON_BLUE, 4)
        painter.setPen(pen)
        size = 40
        painter.drawLine(0, size, 0, 0)
        painter.drawLine(0, 0, size, 0)
        painter.drawLine(self.width()-size, 0, self.width(), 0)
        painter.drawLine(self.width(), 0, self.width(), size)
        painter.drawLine(0, self.height()-size, 0, self.height())
        painter.drawLine(0, self.height(), size, self.height())
        painter.drawLine(self.width()-size, self.height(), self.width(), self.height())
        painter.drawLine(self.width(), self.height()-size, self.width(), self.height())

    def _scale_point(self, point):
        """缩放点坐标"""
        x = int(point[0] * self.scale_factor + self.offset_x)
        y = int(point[1] * self.scale_factor + self.offset_y)
        return (x, y)

    def _scale_rect(self, rect):
        """缩放矩形坐标"""
        x1 = int(rect[0] * self.scale_factor + self.offset_x)
        y1 = int(rect[1] * self.scale_factor + self.offset_y)
        x2 = int(rect[2] * self.scale_factor + self.offset_x)
        y2 = int(rect[3] * self.scale_factor + self.offset_y)
        return (x1, y1, x2, y2)

    def _unscale_point(self, point):
        """反向缩放点坐标"""
        x = (point[0] - self.offset_x) / self.scale_factor
        y = (point[1] - self.offset_y) / self.scale_factor
        return (x, y)

    def mousePressEvent(self, event):
        """鼠标按下事件"""
        if event.button() == Qt.LeftButton and self.drawing_mode:
            self.drawing_start = (event.x(), event.y())
            self.drawing_end = (event.x(), event.y())
            
            # 如果是手绘轮廓模式，初始化轮廓点列表
            if self.drawing_mode == "contour":
                sx, sy = self._unscale_point(self.drawing_start)
                self.contour_points = [[int(sx), int(sy)]]

    def mouseMoveEvent(self, event):
        """鼠标移动事件 - 实时预览检测"""
        if self.drawing_start:
            self.drawing_end = (event.x(), event.y())

            # 如果是自动检测直线模式，实时预览检测结果
            if self.drawing_mode == "auto_line_roi" and self.current_frame is not None:
                try:
                    sx, sy = self._unscale_point(self.drawing_start)
                    ex, ey = self._unscale_point(self.drawing_end)
                    rect = (min(sx, ex), min(sy, ey), max(sx, ex), max(sy, ey))

                    # 使用低阈值Canny算法检测直线（与手动测试保持一致）
                    preview_lines = VisionAlgorithm.detect_lines_single_algorithm(
                        self.current_frame, rect, 'canny_low', []
                    )
                    self.preview_lines = preview_lines
                except Exception as e:
                    self.preview_lines = []
            
            # 如果是手绘轮廓模式，记录轨迹点
            elif self.drawing_mode == "contour":
                ex, ey = self._unscale_point(self.drawing_end)
                self.contour_points.append([int(ex), int(ey)])

            # 如果是绘制ROI模式，实时预览矩形
            elif self.drawing_mode == "roi":
                sx, sy = self._unscale_point(self.drawing_start)
                ex, ey = self._unscale_point(self.drawing_end)
                self.preview_rect = (min(sx, ex), min(sy, ey), max(sx, ex), max(sy, ey))

            self.update()

    def mouseReleaseEvent(self, event):
        """鼠标释放事件"""
        if event.button() == Qt.LeftButton and self.drawing_start and self.drawing_end:
            sx, sy = self._unscale_point(self.drawing_start)
            ex, ey = self._unscale_point(self.drawing_end)

            if self.drawing_mode == "roi":
                rect = (min(sx, ex), min(sy, ey), max(sx, ex), max(sy, ey))
                self.roi_selected.emit(rect)
            elif self.drawing_mode == "line":
                self.line_drawn.emit((sx, sy, ex, ey))
            elif self.drawing_mode == "contour":
                # 发送手绘的轮廓点而不是矩形
                if hasattr(self, 'contour_points') and len(self.contour_points) > 2:
                    contour = np.array(self.contour_points, dtype=np.int32).reshape((-1, 1, 2))
                    self.contour_drawn.emit(contour)
                else:
                    # 如果点太少，发送矩形区域进行自动识别
                    rect = (min(sx, ex), min(sy, ey), max(sx, ex), max(sy, ey))
                    self.contour_selected.emit(rect)
            elif self.drawing_mode == "auto_line_roi":
                # 框选区域后自动检测直线
                rect = (min(sx, ex), min(sy, ey), max(sx, ex), max(sy, ey))
                self.auto_line_roi_selected.emit(rect)

            self.drawing_start = None
            self.drawing_end = None
            self.drawing_mode = None
            self.contour_points = []
            self.clear_preview()
            self.update()

    def start_drawing(self, mode):
        """开始绘制"""
        self.drawing_mode = mode
        self.setCursor(QCursor(Qt.CrossCursor))

    def clear_preview(self):
        """清除预览数据"""
        self.preview_lines = []
        self.preview_rect = None
        self.update()

    def stop_drawing(self):
        """停止绘制"""
        self.drawing_mode = None
        self.setCursor(QCursor(Qt.ArrowCursor))
        self.clear_preview()

    def clear_all(self):
        """清除所有数据"""
        self.left_line_long = None
        self.left_line_short = None
        self.right_line_long = None
        self.right_line_short = None
        self.left_roi = None
        self.right_roi = None
        self.left_contour = None
        self.right_contour = None
        self.left_contour_roi = None
        self.right_contour_roi = None
        self.detected_lines = []
        self.detected_contours = []
        self.detected_center = None
        self.detected_angle = 0.0
        self.preview_lines = []
        self.preview_rect = None
        self.undo_stack.clear()
        self.test_mode = False
        self.update()

    def enter_test_mode(self):
        """进入测试模式"""
        self.test_mode = True
        print("调试: 进入测试模式")
        self.update()

    def exit_test_mode(self):
        """退出测试模式"""
        self.test_mode = False
        self.detected_lines = []
        self.detected_contours = []
        self.detected_rois = []
        self.detected_center = None
        self.detected_angle = 0.0
        print("调试: 退出测试模式")
        self.update()


class MainWindow(QMainWindow):
    """
    主窗口类

    优化内容：
1. 添加撤销功能 - 区域和直线选择可撤销
2. 优化手动测试 - 隐藏模板线条，显示识别结果，计算X/Y/Z/A偏移量
3. 调整整体UI字体大小
4. 添加自动保存/加载功能 - 启动时加载上次模板，保存时选择位置
5. DPI屏幕适配 - 支持不同分辨率和DPI设置
"""

    def __init__(self):
        super().__init__()

        # 计算DPI缩放因子
        self._calculate_dpi_scale()

        self.setWindowTitle("贴内衬机视觉定位系统 - 优化版")

        # 根据屏幕尺寸设置窗口大小
        screen = QApplication.primaryScreen()
        if screen:
            screen_geo = screen.geometry()
            # 使用屏幕尺寸的85%
            width = int(screen_geo.width() * 0.85)
            height = int(screen_geo.height() * 0.85)
            # 确保最小尺寸
            width = max(1200, min(width, screen_geo.width() - 100))
            height = max(800, min(height, screen_geo.height() - 100))
            self.setGeometry(50, 50, width, height)
        else:
            self.setGeometry(50, 50, 1600, 900)

        self.setStyleSheet(
            "QMainWindow { background-color: #0A0F1E; }"
            "QSplitter::handle { background-color: #0A1428; }"
            "QSplitter::handle:hover { background-color: #0066CC; }"
        )

        # 初始化变量
        self.camera_thread = None
        self.sim_camera_thread = None       # 模拟摄像头线程
        self.sim_config = SimulationConfig()
        self.sim_test_engine = None         # 批量测试引擎
        self.sim_last_truth = None          # 最近一帧真实位姿
        self._batch_display_active = False  # 批量可视化期间主画面由批量帧接管
        self.current_mode = "line"
        self.calibration_data = CalibrationData()
        self.dual_line_template = DualLineTemplate()
        self.shape_template = ShapeTemplate()

        # 绘制状态跟踪
        self.line_drawing_side = None
        self.roi_drawing_side = None
        self.contour_drawing_side = None
        self.contour_roi_drawing_side = None

        # 标定图像列表
        self.calib_images = []

        # 动态检测定时器
        self.detection_timer = QTimer()
        self.detection_timer.setInterval(100)  # 100ms间隔，约10fps
        self.detection_timer.timeout.connect(self._perform_detection_dispatcher)

        # 创建UI
        self.image_canvas = ImageCanvas()
        self._setup_ui()
        self._setup_connections()

        # 加载上次保存的配置和模板
        self._load_last_session()

    def _calculate_dpi_scale(self):
        """计算DPI缩放因子"""
        app = QApplication.instance()
        if app:
            # 获取DPI缩放因子
            desktop = app.desktop()
            if desktop:
                # 尝试获取高DPI缩放
                try:
                    self.dpi_scale = desktop.logicalDpiX() / 96.0
                except:
                    self.dpi_scale = 1.0
            else:
                self.dpi_scale = 1.0
        else:
            self.dpi_scale = 1.0
        # 确保缩放因子在合理范围内
        self.dpi_scale = max(0.8, min(self.dpi_scale, 2.0))

    def _get_scaled_font_size(self, base_size):
        """根据DPI缩放字体大小"""
        return int(base_size * self.dpi_scale)

    def _load_last_session(self):
        """加载上次会话的数据 - 只加载标定和贴料参数，不加载模板"""
        try:
            # 加载标定数据
            if os.path.exists(DEFAULT_CALIB_FILE):
                try:
                    self.calibration_data = TemplateManager.load_calibration(DEFAULT_CALIB_FILE)
                    self._update_calib_display()
                    self.status_label.setText("已加载上次标定数据")
                except Exception as e:
                    print(f"加载标定数据失败: {e}")
                    # 初始化空的标定数据
                    self.calibration_data = CalibrationData()

            # 不加载双直线模板 - 保持为空
            self.dual_line_template = DualLineTemplate()

            # 不加载异形模板 - 保持为空
            self.shape_template = ShapeTemplate()

            # 加载贴料参数
            if os.path.exists(DEFAULT_CONFIG_FILE):
                try:
                    with open(DEFAULT_CONFIG_FILE, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    self._apply_paste_params(config)
                except Exception as e:
                    print(f"加载贴料参数失败: {e}")

            # 更新状态，不显示模板
            self._update_line_template_status()
            self._update_shape_template_status()

        except Exception as e:
            print(f"加载上次会话失败: {e}")

    def _repair_line_template_file(self):
        """修复损坏的模板文件"""
        try:
            default_content = {
                "left_line_long": None,
                "left_line_short": None,
                "right_line_long": None,
                "right_line_short": None,
                "left_roi": None,
                "right_roi": None,
                "valid": False
            }
            with open(DEFAULT_LINE_TEMPLATE_FILE, 'w', encoding='utf-8') as f:
                json.dump(default_content, f, indent=2, ensure_ascii=False)
            print("已修复损坏的模板文件")
        except Exception as e:
            print(f"修复模板文件失败: {e}")

    def _apply_paste_params(self, config):
        """应用贴料参数"""
        self.offset_x_spin.setValue(config.get('offset_x', 0))
        self.offset_y_spin.setValue(config.get('offset_y', 0))
        self.offset_z_spin.setValue(config.get('offset_z', 0))
        self.offset_a_spin.setValue(config.get('offset_a', 0))
        self.score_threshold_spin.setValue(config.get('score_threshold', 0.7))
        self.alarm_x_spin.setValue(config.get('alarm_x', 5))
        self.alarm_y_spin.setValue(config.get('alarm_y', 5))
        self.alarm_a_spin.setValue(config.get('alarm_a', 3))
        self.timeout_spin.setValue(config.get('timeout', 3000))
        self.photo_delay_spin.setValue(config.get('photo_delay', 100))
        self.vacuum_delay_spin.setValue(config.get('vacuum_delay', 200))
        self.retry_count_spin.setValue(config.get('retry_count', 3))

    def _save_current_session(self):
        """保存当前会话"""
        try:
            # 保存标定数据
            if self.calibration_data.is_calibrated:
                TemplateManager.save_calibration(DEFAULT_CALIB_FILE, self.calibration_data)

            # 保存双直线模板
            if self.dual_line_template.valid:
                TemplateManager.save_dual_line_template(DEFAULT_LINE_TEMPLATE_FILE, self.dual_line_template)

            # 保存异形模板
            if self.shape_template.valid:
                params = {
                    'canny_low': self.canny_low_slider.value(),
                    'canny_high': self.canny_high_slider.value(),
                    'match_threshold': self.match_threshold_spin.value()
                }
                TemplateManager.save_shape_template(DEFAULT_SHAPE_TEMPLATE_FILE, self.shape_template, params)

            # 保存贴料参数
            config = {
                'offset_x': self.offset_x_spin.value(),
                'offset_y': self.offset_y_spin.value(),
                'offset_z': self.offset_z_spin.value(),
                'offset_a': self.offset_a_spin.value(),
                'score_threshold': self.score_threshold_spin.value(),
                'alarm_x': self.alarm_x_spin.value(),
                'alarm_y': self.alarm_y_spin.value(),
                'alarm_a': self.alarm_a_spin.value(),
                'timeout': self.timeout_spin.value(),
                'photo_delay': self.photo_delay_spin.value(),
                'vacuum_delay': self.vacuum_delay_spin.value(),
                'retry_count': self.retry_count_spin.value(),
                'fail_mode': self.fail_mode_combo.currentText()
            }
            with open(DEFAULT_CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"保存会话失败: {e}")

    def _setup_ui(self):
        """设置UI"""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # 三栏可拖拽分割：用户可自由调节左/中/右面板宽度
        left = self._create_left_panel()
        center = self._create_center_panel()
        right = self._create_right_panel()
        left.setMinimumWidth(230)
        center.setMinimumWidth(380)
        right.setMinimumWidth(290)

        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setHandleWidth(6)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.addWidget(left)
        self.main_splitter.addWidget(center)
        self.main_splitter.addWidget(right)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setStretchFactor(2, 0)
        self.main_splitter.setSizes([320, 860, 380])
        main_layout.addWidget(self.main_splitter)

        self._setup_statusbar()

    def _create_left_panel(self):
        """创建左侧面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(15)

        # 摄像头控制组
        camera_group = QGroupBox("摄像头控制")
        camera_group.setStyleSheet(self._get_group_style())
        camera_layout = QVBoxLayout(camera_group)

        self.btn_detect_cam = QPushButton("🔍 检测摄像头")
        self.btn_detect_cam.setStyleSheet(self._get_button_style())
        self.btn_detect_cam.clicked.connect(self._detect_cameras)
        camera_layout.addWidget(self.btn_detect_cam)

        camera_layout.addWidget(QLabel("选择摄像头:"))
        self.camera_combo = QComboBox()
        self.camera_combo.addItem("请检测摄像头")
        self.camera_combo.setStyleSheet(self._get_combo_style())
        camera_layout.addWidget(self.camera_combo)

        camera_layout.addWidget(QLabel("分辨率:"))
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["640x480", "800x600", "1280x720", "1920x1080"])
        self.resolution_combo.setStyleSheet(self._get_combo_style())
        camera_layout.addWidget(self.resolution_combo)

        self.btn_open_cam = QPushButton("📷 打开摄像头")
        self.btn_open_cam.setStyleSheet(self._get_button_style("#00AA66"))
        self.btn_open_cam.clicked.connect(self._toggle_camera)
        camera_layout.addWidget(self.btn_open_cam)

        layout.addWidget(camera_group)

        # 相机标定组
        calib_group = QGroupBox("相机标定")
        calib_group.setStyleSheet(self._get_group_style())
        calib_layout = QVBoxLayout(calib_group)

        calib_layout.addWidget(QLabel("标定方法:"))
        self.calib_method = QComboBox()
        self.calib_method.addItems(["棋盘格标定", "点对应标定", "手动输入"])
        self.calib_method.setStyleSheet(self._get_combo_style())
        self.calib_method.currentIndexChanged.connect(self._on_calib_method_changed)
        calib_layout.addWidget(self.calib_method)

        # 棋盘格参数
        self.checkerboard_widget = QWidget()
        cb_layout = QVBoxLayout(self.checkerboard_widget)

        cb_layout.addWidget(QLabel("棋盘格内角点:"))
        cb_size_layout = QHBoxLayout()
        self.cb_cols_spin = QSpinBox()
        self.cb_cols_spin.setRange(3, 20)
        self.cb_cols_spin.setValue(9)
        self.cb_cols_spin.setStyleSheet(self._get_spinbox_style())
        cb_size_layout.addWidget(self.cb_cols_spin)
        cb_size_layout.addWidget(QLabel("x"))
        self.cb_rows_spin = QSpinBox()
        self.cb_rows_spin.setRange(3, 20)
        self.cb_rows_spin.setValue(6)
        self.cb_rows_spin.setStyleSheet(self._get_spinbox_style())
        cb_size_layout.addWidget(self.cb_rows_spin)
        cb_layout.addLayout(cb_size_layout)

        cb_layout.addWidget(QLabel("方格尺寸(mm):"))
        self.square_size_spin = QDoubleSpinBox()
        self.square_size_spin.setRange(1, 100)
        self.square_size_spin.setValue(25.0)
        self.square_size_spin.setStyleSheet(self._get_spinbox_style())
        cb_layout.addWidget(self.square_size_spin)

        self.btn_capture_calib = QPushButton("采集标定图像")
        self.btn_capture_calib.setStyleSheet(self._get_button_style())
        self.btn_capture_calib.clicked.connect(self._capture_calib_image)
        cb_layout.addWidget(self.btn_capture_calib)

        self.calib_image_count = QLabel("已采集: 0 张")
        self.calib_image_count.setStyleSheet(self._get_label_style(12, "#00FFFF"))
        cb_layout.addWidget(self.calib_image_count)

        self.btn_calibrate = QPushButton("执行标定")
        self.btn_calibrate.setStyleSheet(self._get_button_style("#00AA66"))
        self.btn_calibrate.clicked.connect(self._perform_calibration)
        cb_layout.addWidget(self.btn_calibrate)

        calib_layout.addWidget(self.checkerboard_widget)

        # 点对应标定参数
        self.points_widget = QWidget()
        points_layout = QVBoxLayout(self.points_widget)
        self.points_widget.setVisible(False)

        self.calib_table = QTableWidget()
        self.calib_table.setColumnCount(4)
        self.calib_table.setHorizontalHeaderLabels(["#", "像素X", "像素Y", "物理值(mm)"])
        self.calib_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.calib_table.setStyleSheet(self._get_table_style())
        self.calib_table.setMinimumHeight(150)
        points_layout.addWidget(self.calib_table)

        calib_btn_layout = QHBoxLayout()
        self.btn_add_point = QPushButton("添加点")
        self.btn_add_point.setStyleSheet(self._get_button_style())
        self.btn_add_point.clicked.connect(self._add_calib_point)
        calib_btn_layout.addWidget(self.btn_add_point)

        self.btn_clear_points = QPushButton("清除")
        self.btn_clear_points.setStyleSheet(self._get_button_style("#FF6666"))
        self.btn_clear_points.clicked.connect(self._clear_calib_points)
        calib_btn_layout.addWidget(self.btn_clear_points)

        points_layout.addLayout(calib_btn_layout)

        self.btn_calc_calib = QPushButton("计算标定")
        self.btn_calc_calib.setStyleSheet(self._get_button_style("#00AA66"))
        self.btn_calc_calib.clicked.connect(self._calculate_calibration_from_points)
        points_layout.addWidget(self.btn_calc_calib)

        calib_layout.addWidget(self.points_widget)

        self.calib_info = QLabel("状态: 未标定")
        self.calib_info.setStyleSheet(self._get_label_style(12, "#FF6666"))
        calib_layout.addWidget(self.calib_info)

        self.btn_show_calib_info = QPushButton("查看标定原理")
        self.btn_show_calib_info.setStyleSheet(self._get_button_style())
        self.btn_show_calib_info.clicked.connect(self._show_calibration_info)
        calib_layout.addWidget(self.btn_show_calib_info)

        calib_io_layout = QHBoxLayout()
        self.btn_save_calib = QPushButton("保存标定")
        self.btn_save_calib.setStyleSheet(self._get_button_style())
        self.btn_save_calib.clicked.connect(self._save_calibration)
        calib_io_layout.addWidget(self.btn_save_calib)

        self.btn_load_calib = QPushButton("加载标定")
        self.btn_load_calib.setStyleSheet(self._get_button_style())
        self.btn_load_calib.clicked.connect(self._load_calibration)
        calib_io_layout.addWidget(self.btn_load_calib)

        calib_layout.addLayout(calib_io_layout)

        layout.addWidget(calib_group)
        layout.addStretch()

        return panel

    def _create_center_panel(self):
        """创建中间面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(15)

        # 标题栏
        title_bar = QWidget()
        title_layout = QHBoxLayout(title_bar)

        title_label = QLabel("贴内衬机视觉定位系统")
        title_label.setFont(QFont("Consolas", 18, QFont.Bold))
        title_label.setStyleSheet("color: #00FFFF;")
        title_layout.addWidget(title_label)

        self.mode_switch = QComboBox()
        self.mode_switch.addItems(["双直线模式(矩形)", "双外形模式(异形)"])
        self.mode_switch.setStyleSheet(self._get_combo_style())
        self.mode_switch.currentIndexChanged.connect(self._change_mode)
        title_layout.addWidget(self.mode_switch)

        layout.addWidget(title_bar)

        # 图像画布
        self.image_canvas.setStyleSheet("border: 3px solid #0066CC; border-radius: 15px;")
        layout.addWidget(self.image_canvas, stretch=1)

        # 实时检测数据面板
        data_panel = QGroupBox("实时检测数据")
        data_panel.setStyleSheet(self._get_group_style())
        data_layout = QGridLayout(data_panel)

        data_layout.addWidget(QLabel("X(像素):"), 0, 0)
        self.data_x_pixel = QLabel("--")
        self.data_x_pixel.setStyleSheet(self._get_label_style(12, "#00FF96", True))
        data_layout.addWidget(self.data_x_pixel, 0, 1)

        data_layout.addWidget(QLabel("Y(像素):"), 0, 2)
        self.data_y_pixel = QLabel("--")
        self.data_y_pixel.setStyleSheet(self._get_label_style(12, "#00FF96", True))
        data_layout.addWidget(self.data_y_pixel, 0, 3)

        data_layout.addWidget(QLabel("X(mm):"), 1, 0)
        self.data_x_mm = QLabel("--")
        self.data_x_mm.setStyleSheet(self._get_label_style(12, "#00FFFF", True))
        data_layout.addWidget(self.data_x_mm, 1, 1)

        data_layout.addWidget(QLabel("Y(mm):"), 1, 2)
        self.data_y_mm = QLabel("--")
        self.data_y_mm.setStyleSheet(self._get_label_style(12, "#00FFFF", True))
        data_layout.addWidget(self.data_y_mm, 1, 3)

        data_layout.addWidget(QLabel("Z(mm):"), 2, 0)
        self.data_z = QLabel("0.00")
        self.data_z.setStyleSheet(self._get_label_style(12, "#FF9600", True))
        data_layout.addWidget(self.data_z, 2, 1)

        data_layout.addWidget(QLabel("角度(A):"), 2, 2)
        self.data_angle = QLabel("--")
        self.data_angle.setStyleSheet(self._get_label_style(12, "#FF9600", True))
        data_layout.addWidget(self.data_angle, 2, 3)

        data_layout.addWidget(QLabel("匹配分数:"), 3, 0)
        self.data_score = QLabel("--")
        self.data_score.setStyleSheet(self._get_label_style(12, "#AA00AA", True))
        data_layout.addWidget(self.data_score, 3, 1)

        data_layout.addWidget(QLabel("状态:"), 3, 2)
        self.data_status = QLabel("等待")
        self.data_status.setStyleSheet(self._get_label_style(12, "#FF6666", True))
        data_layout.addWidget(self.data_status, 3, 3)

        # 偏移量显示（新增）
        offset_panel = QGroupBox("机械臂偏移参数")
        offset_panel.setStyleSheet(self._get_group_style())
        offset_layout = QGridLayout(offset_panel)

        offset_layout.addWidget(QLabel("X偏移:"), 0, 0)
        self.offset_x_value = QLabel("0.00 mm")
        self.offset_x_value.setStyleSheet(self._get_value_style(13, "#00FF96"))
        offset_layout.addWidget(self.offset_x_value, 0, 1)

        offset_layout.addWidget(QLabel("Y偏移:"), 0, 2)
        self.offset_y_value = QLabel("0.00 mm")
        self.offset_y_value.setStyleSheet(self._get_value_style(13, "#00FF96"))
        offset_layout.addWidget(self.offset_y_value, 0, 3)

        offset_layout.addWidget(QLabel("Z偏移:"), 1, 0)
        self.offset_z_value = QLabel("0.00 mm")
        self.offset_z_value.setStyleSheet(self._get_value_style(13, "#FF9600"))
        offset_layout.addWidget(self.offset_z_value, 1, 1)

        offset_layout.addWidget(QLabel("A偏移:"), 1, 2)
        self.offset_a_value = QLabel("0.00 °")
        self.offset_a_value.setStyleSheet(self._get_value_style(13, "#FF9600"))
        offset_layout.addWidget(self.offset_a_value, 1, 3)

        layout.addWidget(offset_panel)
        layout.addWidget(data_panel)

        return panel

    def _create_right_panel(self):
        """创建右侧面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(15)

        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet(self._get_tab_style())

        self.line_tab = QWidget()
        self._setup_line_template_tab()
        self.tab_widget.addTab(self.line_tab, "双直线模板")

        self.shape_tab = QWidget()
        self._setup_shape_template_tab()
        self.tab_widget.addTab(self.shape_tab, "外形模板")

        self.paste_tab = QWidget()
        self._setup_paste_params_tab()
        self.tab_widget.addTab(self.paste_tab, "贴料参数")

        self.test_tab = QWidget()
        self._setup_interactive_test_tab()
        self.tab_widget.addTab(self.test_tab, "交互测试")

        self.sim_tab = QWidget()
        self._setup_simulation_tab()
        self.tab_widget.addTab(self.sim_tab, "模拟测试")

        layout.addWidget(self.tab_widget)
        layout.addStretch()

        return panel

    def _setup_line_template_tab(self):
        """设置双直线模板选项卡"""
        layout = QVBoxLayout(self.line_tab)
        layout.setSpacing(15)

        # 撤销按钮
        self.btn_undo_line = QPushButton("↩ 撤销上一步")
        self.btn_undo_line.setStyleSheet(self._get_button_style("#FF9600"))
        self.btn_undo_line.clicked.connect(self._undo_line_operation)
        layout.addWidget(self.btn_undo_line)

        # 左侧模板组
        left_group = QGroupBox("左侧模板（长边+短边）")
        left_group.setStyleSheet(self._get_group_style())
        left_layout = QVBoxLayout(left_group)

        left_long_layout = QHBoxLayout()
        self.btn_draw_left_long = QPushButton("框选长边")
        self.btn_draw_left_long.setStyleSheet(self._get_button_style("#00AA66"))
        self.btn_draw_left_long.clicked.connect(lambda: self._start_auto_detect_line("left_long"))
        left_long_layout.addWidget(self.btn_draw_left_long)

        self.btn_clear_left_long = QPushButton("清除")
        self.btn_clear_left_long.setStyleSheet(self._get_button_style("#FF6666"))
        self.btn_clear_left_long.clicked.connect(lambda: self._clear_line("left_long"))
        left_long_layout.addWidget(self.btn_clear_left_long)

        left_layout.addLayout(left_long_layout)

        self.left_long_info = QLabel("长边: 未设置")
        self.left_long_info.setStyleSheet(self._get_label_style(14, "#00FF96"))
        left_layout.addWidget(self.left_long_info)

        left_short_layout = QHBoxLayout()
        self.btn_draw_left_short = QPushButton("框选短边")
        self.btn_draw_left_short.setStyleSheet(self._get_button_style("#00AA66"))
        self.btn_draw_left_short.clicked.connect(lambda: self._start_auto_detect_line("left_short"))
        left_short_layout.addWidget(self.btn_draw_left_short)

        self.btn_clear_left_short = QPushButton("清除")
        self.btn_clear_left_short.setStyleSheet(self._get_button_style("#FF6666"))
        self.btn_clear_left_short.clicked.connect(lambda: self._clear_line("left_short"))
        left_short_layout.addWidget(self.btn_clear_left_short)

        left_layout.addLayout(left_short_layout)

        self.left_short_info = QLabel("短边: 未设置")
        self.left_short_info.setStyleSheet(self._get_label_style(14, "#00FF96"))
        left_layout.addWidget(self.left_short_info)

        self.btn_draw_left_roi = QPushButton("设置左监测区域")
        self.btn_draw_left_roi.setStyleSheet(self._get_button_style())
        self.btn_draw_left_roi.clicked.connect(lambda: self._start_draw_roi("left"))
        left_layout.addWidget(self.btn_draw_left_roi)

        self.left_roi_info = QLabel("区域: 未设置")
        self.left_roi_info.setStyleSheet(self._get_label_style(14, "#00FF96"))
        left_layout.addWidget(self.left_roi_info)

        layout.addWidget(left_group)

        # 右侧模板组
        right_group = QGroupBox("右侧模板（长边+短边）")
        right_group.setStyleSheet(self._get_group_style())
        right_layout = QVBoxLayout(right_group)

        right_long_layout = QHBoxLayout()
        self.btn_draw_right_long = QPushButton("框选长边")
        self.btn_draw_right_long.setStyleSheet(self._get_button_style("#00AA66"))
        self.btn_draw_right_long.clicked.connect(lambda: self._start_auto_detect_line("right_long"))
        right_long_layout.addWidget(self.btn_draw_right_long)

        self.btn_clear_right_long = QPushButton("清除")
        self.btn_clear_right_long.setStyleSheet(self._get_button_style("#FF6666"))
        self.btn_clear_right_long.clicked.connect(lambda: self._clear_line("right_long"))
        right_long_layout.addWidget(self.btn_clear_right_long)

        right_layout.addLayout(right_long_layout)

        self.right_long_info = QLabel("长边: 未设置")
        self.right_long_info.setStyleSheet(self._get_label_style(14, "#FF9600"))
        right_layout.addWidget(self.right_long_info)

        right_short_layout = QHBoxLayout()
        self.btn_draw_right_short = QPushButton("框选短边")
        self.btn_draw_right_short.setStyleSheet(self._get_button_style("#00AA66"))
        self.btn_draw_right_short.clicked.connect(lambda: self._start_auto_detect_line("right_short"))
        right_short_layout.addWidget(self.btn_draw_right_short)

        self.btn_clear_right_short = QPushButton("清除")
        self.btn_clear_right_short.setStyleSheet(self._get_button_style("#FF6666"))
        self.btn_clear_right_short.clicked.connect(lambda: self._clear_line("right_short"))
        right_short_layout.addWidget(self.btn_clear_right_short)

        right_layout.addLayout(right_short_layout)

        self.right_short_info = QLabel("短边: 未设置")
        self.right_short_info.setStyleSheet(self._get_label_style(14, "#FF9600"))
        right_layout.addWidget(self.right_short_info)

        self.btn_draw_right_roi = QPushButton("设置右监测区域")
        self.btn_draw_right_roi.setStyleSheet(self._get_button_style())
        self.btn_draw_right_roi.clicked.connect(lambda: self._start_draw_roi("right"))
        right_layout.addWidget(self.btn_draw_right_roi)

        self.right_roi_info = QLabel("区域: 未设置")
        self.right_roi_info.setStyleSheet(self._get_label_style(14, "#FF9600"))
        right_layout.addWidget(self.right_roi_info)

        layout.addWidget(right_group)

        # 操作组
        action_group = QGroupBox("操作")
        action_group.setStyleSheet(self._get_group_style())
        action_layout = QVBoxLayout(action_group)

        self.line_template_status = QLabel("模板状态: 未完成")
        self.line_template_status.setStyleSheet(self._get_label_style(12, "#FF6666", True))
        action_layout.addWidget(self.line_template_status)

        self.btn_save_line_template = QPushButton("💾 保存当前模板")
        self.btn_save_line_template.setStyleSheet(self._get_button_style("#0088FF"))
        self.btn_save_line_template.clicked.connect(self._save_line_template)
        self.btn_save_line_template.setEnabled(False)
        action_layout.addWidget(self.btn_save_line_template)

        self.btn_test_line = QPushButton("🔍 手动测试")
        self.btn_test_line.setStyleSheet(self._get_button_style("#AA00AA"))
        self.btn_test_line.clicked.connect(self._test_line_detection)
        self.btn_test_line.setEnabled(False)
        action_layout.addWidget(self.btn_test_line)

        # 算法选择下拉菜单
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.setStyleSheet(self._get_button_style())
        self.algorithm_combo.setFixedWidth(150)
        self._populate_algorithm_combo()
        action_layout.addWidget(self.algorithm_combo)

        # 算法描述标签
        self.algorithm_desc = QLabel("选择检测算法")
        self.algorithm_desc.setStyleSheet(self._get_label_style(size=10, color="#88CCFF"))
        self.algorithm_desc.setWordWrap(True)
        action_layout.addWidget(self.algorithm_desc)

        # 算法选择事件
        self.algorithm_combo.currentIndexChanged.connect(self._on_algorithm_changed)

        self.btn_load_line_template = QPushButton("📂 加载模板")
        self.btn_load_line_template.setStyleSheet(self._get_button_style())
        self.btn_load_line_template.clicked.connect(self._load_line_template)
        action_layout.addWidget(self.btn_load_line_template)

        self.btn_clear_all_lines = QPushButton("🗑 清除所有")
        self.btn_clear_all_lines.setStyleSheet(self._get_button_style("#FF6666"))
        self.btn_clear_all_lines.clicked.connect(self._clear_all_line_template)
        action_layout.addWidget(self.btn_clear_all_lines)

        layout.addWidget(action_group)
        layout.addStretch()

    def _setup_shape_template_tab(self):
        """设置异形模板选项卡"""
        layout = QVBoxLayout(self.shape_tab)
        layout.setSpacing(15)

        # 撤销按钮
        self.btn_undo_shape = QPushButton("↩ 撤销上一步")
        self.btn_undo_shape.setStyleSheet(self._get_button_style("#FF9600"))
        self.btn_undo_shape.clicked.connect(self._undo_shape_operation)
        layout.addWidget(self.btn_undo_shape)

        # 参数设置组
        param_group = QGroupBox("轮廓参数")
        param_group.setStyleSheet(self._get_group_style())
        param_layout = QVBoxLayout(param_group)

        param_layout.addWidget(QLabel("Canny低阈值:"))
        self.canny_low_slider = QSlider(Qt.Horizontal)
        self.canny_low_slider.setRange(10, 200)
        self.canny_low_slider.setValue(50)
        self.canny_low_slider.setStyleSheet(self._get_slider_style())
        param_layout.addWidget(self.canny_low_slider)

        param_layout.addWidget(QLabel("Canny高阈值:"))
        self.canny_high_slider = QSlider(Qt.Horizontal)
        self.canny_high_slider.setRange(50, 300)
        self.canny_high_slider.setValue(150)
        self.canny_high_slider.setStyleSheet(self._get_slider_style())
        param_layout.addWidget(self.canny_high_slider)

        param_layout.addWidget(QLabel("匹配阈值:"))
        self.match_threshold_spin = QDoubleSpinBox()
        self.match_threshold_spin.setRange(0.1, 1.0)
        self.match_threshold_spin.setValue(0.7)
        self.match_threshold_spin.setSingleStep(0.05)
        self.match_threshold_spin.setStyleSheet(self._get_spinbox_style())
        param_layout.addWidget(self.match_threshold_spin)

        layout.addWidget(param_group)

        # 左侧轮廓组
        left_group = QGroupBox("左侧轮廓")
        left_group.setStyleSheet(self._get_group_style())
        left_layout = QGridLayout(left_group)

        # 左上：手绘左轮廓
        self.btn_draw_left_contour = QPushButton("手绘左轮廓")
        self.btn_draw_left_contour.setStyleSheet(self._get_button_style("#00AA66"))
        self.btn_draw_left_contour.clicked.connect(lambda: self._start_draw_contour("left"))
        left_layout.addWidget(self.btn_draw_left_contour, 0, 0)

        # 右上：自动识别区域（需要先设置轮廓）
        self.btn_auto_detect_left_roi = QPushButton("自动识别区域")
        self.btn_auto_detect_left_roi.setStyleSheet(self._get_button_style("#00AAFF"))
        self.btn_auto_detect_left_roi.clicked.connect(self._auto_detect_left_contour_roi)
        left_layout.addWidget(self.btn_auto_detect_left_roi, 0, 1)

        # 左下：手绘左区域
        self.btn_draw_left_contour_roi = QPushButton("手绘左区域")
        self.btn_draw_left_contour_roi.setStyleSheet(self._get_button_style())
        self.btn_draw_left_contour_roi.clicked.connect(lambda: self._start_draw_contour_roi("left"))
        left_layout.addWidget(self.btn_draw_left_contour_roi, 1, 0)

        # 右下：自动识别左轮廓（需要先设置区域）
        self.btn_auto_detect_left_contour = QPushButton("自动识别轮廓")
        self.btn_auto_detect_left_contour.setStyleSheet(self._get_button_style("#00AAFF"))
        self.btn_auto_detect_left_contour.clicked.connect(self._auto_detect_left_contour)
        left_layout.addWidget(self.btn_auto_detect_left_contour, 1, 1)

        self.left_contour_info = QLabel("轮廓: 未设置")
        self.left_contour_info.setStyleSheet(self._get_label_style(14, "#00FF96"))
        left_layout.addWidget(self.left_contour_info, 2, 0, 1, 2)

        self.left_contour_roi_info = QLabel("区域: 未设置")
        self.left_contour_roi_info.setStyleSheet(self._get_label_style(14, "#00FF96"))
        left_layout.addWidget(self.left_contour_roi_info, 3, 0, 1, 2)

        layout.addWidget(left_group)

        # 右侧轮廓组
        right_group = QGroupBox("右侧轮廓")
        right_group.setStyleSheet(self._get_group_style())
        right_layout = QGridLayout(right_group)

        # 左上：手绘右轮廓
        self.btn_draw_right_contour = QPushButton("手绘右轮廓")
        self.btn_draw_right_contour.setStyleSheet(self._get_button_style("#00AA66"))
        self.btn_draw_right_contour.clicked.connect(lambda: self._start_draw_contour("right"))
        right_layout.addWidget(self.btn_draw_right_contour, 0, 0)

        # 右上：自动识别区域（需要先设置轮廓）
        self.btn_auto_detect_right_roi = QPushButton("自动识别区域")
        self.btn_auto_detect_right_roi.setStyleSheet(self._get_button_style("#00AAFF"))
        self.btn_auto_detect_right_roi.clicked.connect(self._auto_detect_right_contour_roi)
        right_layout.addWidget(self.btn_auto_detect_right_roi, 0, 1)

        # 左下：手绘右区域
        self.btn_draw_right_contour_roi = QPushButton("手绘右区域")
        self.btn_draw_right_contour_roi.setStyleSheet(self._get_button_style())
        self.btn_draw_right_contour_roi.clicked.connect(lambda: self._start_draw_contour_roi("right"))
        right_layout.addWidget(self.btn_draw_right_contour_roi, 1, 0)

        # 右下：自动识别右轮廓（需要先设置区域）
        self.btn_auto_detect_right_contour = QPushButton("自动识别轮廓")
        self.btn_auto_detect_right_contour.setStyleSheet(self._get_button_style("#00AAFF"))
        self.btn_auto_detect_right_contour.clicked.connect(self._auto_detect_right_contour)
        right_layout.addWidget(self.btn_auto_detect_right_contour, 1, 1)

        self.right_contour_info = QLabel("轮廓: 未设置")
        self.right_contour_info.setStyleSheet(self._get_label_style(14, "#FF9600"))
        right_layout.addWidget(self.right_contour_info, 2, 0, 1, 2)

        self.right_contour_roi_info = QLabel("区域: 未设置")
        self.right_contour_roi_info.setStyleSheet(self._get_label_style(14, "#FF9600"))
        right_layout.addWidget(self.right_contour_roi_info, 3, 0, 1, 2)

        layout.addWidget(right_group)

        # 操作组
        action_group = QGroupBox("操作")
        action_group.setStyleSheet(self._get_group_style())
        action_layout = QVBoxLayout(action_group)

        self.shape_template_status = QLabel("模板状态: 未完成")
        self.shape_template_status.setStyleSheet(self._get_label_style(12, "#FF6666", True))
        action_layout.addWidget(self.shape_template_status)

        self.btn_save_shape_template = QPushButton("💾 保存当前模板")
        self.btn_save_shape_template.setStyleSheet(self._get_button_style("#0088FF"))
        self.btn_save_shape_template.clicked.connect(self._save_shape_template)
        self.btn_save_shape_template.setEnabled(False)
        action_layout.addWidget(self.btn_save_shape_template)

        self.btn_test_shape = QPushButton("🔍 手动测试")
        self.btn_test_shape.setStyleSheet(self._get_button_style("#AA00AA"))
        self.btn_test_shape.clicked.connect(self._test_shape_detection)
        self.btn_test_shape.setEnabled(False)
        action_layout.addWidget(self.btn_test_shape)

        self.btn_load_shape_template = QPushButton("📂 加载模板")
        self.btn_load_shape_template.setStyleSheet(self._get_button_style())
        self.btn_load_shape_template.clicked.connect(self._load_shape_template)
        action_layout.addWidget(self.btn_load_shape_template)

        self.btn_clear_all_shapes = QPushButton("🗑 清除所有")
        self.btn_clear_all_shapes.setStyleSheet(self._get_button_style("#FF6666"))
        self.btn_clear_all_shapes.clicked.connect(self._clear_all_shape_template)
        action_layout.addWidget(self.btn_clear_all_shapes)

        layout.addWidget(action_group)
        layout.addStretch()


    def _setup_paste_params_tab(self):
        """设置贴料参数选项卡"""
        layout = QVBoxLayout(self.paste_tab)
        layout.setSpacing(15)

        # 偏移补偿组
        offset_group = QGroupBox("偏移补偿")
        offset_group.setStyleSheet(self._get_group_style())
        offset_layout = QGridLayout(offset_group)

        offset_layout.addWidget(QLabel("X偏移(mm):"), 0, 0)
        self.offset_x_spin = QDoubleSpinBox()
        self.offset_x_spin.setRange(-100, 100)
        self.offset_x_spin.setSingleStep(0.1)
        self.offset_x_spin.setStyleSheet(self._get_spinbox_style())
        offset_layout.addWidget(self.offset_x_spin, 0, 1)

        offset_layout.addWidget(QLabel("Y偏移(mm):"), 1, 0)
        self.offset_y_spin = QDoubleSpinBox()
        self.offset_y_spin.setRange(-100, 100)
        self.offset_y_spin.setSingleStep(0.1)
        self.offset_y_spin.setStyleSheet(self._get_spinbox_style())
        offset_layout.addWidget(self.offset_y_spin, 1, 1)

        offset_layout.addWidget(QLabel("Z偏移(mm):"), 2, 0)
        self.offset_z_spin = QDoubleSpinBox()
        self.offset_z_spin.setRange(-100, 100)
        self.offset_z_spin.setSingleStep(0.1)
        self.offset_z_spin.setStyleSheet(self._get_spinbox_style())
        offset_layout.addWidget(self.offset_z_spin, 2, 1)

        offset_layout.addWidget(QLabel("A偏移(°):"), 3, 0)
        self.offset_a_spin = QDoubleSpinBox()
        self.offset_a_spin.setRange(-180, 180)
        self.offset_a_spin.setSingleStep(0.1)
        self.offset_a_spin.setStyleSheet(self._get_spinbox_style())
        offset_layout.addWidget(self.offset_a_spin, 3, 1)

        layout.addWidget(offset_group)

        # 阈值设置组
        threshold_group = QGroupBox("阈值设置")
        threshold_group.setStyleSheet(self._get_group_style())
        threshold_layout = QGridLayout(threshold_group)

        threshold_layout.addWidget(QLabel("匹配分数阈值:"), 0, 0)
        self.score_threshold_spin = QDoubleSpinBox()
        self.score_threshold_spin.setRange(0.1, 1.0)
        self.score_threshold_spin.setValue(0.7)
        self.score_threshold_spin.setStyleSheet(self._get_spinbox_style())
        threshold_layout.addWidget(self.score_threshold_spin, 0, 1)

        threshold_layout.addWidget(QLabel("X报警限值(mm):"), 1, 0)
        self.alarm_x_spin = QDoubleSpinBox()
        self.alarm_x_spin.setRange(0, 50)
        self.alarm_x_spin.setValue(5)
        self.alarm_x_spin.setStyleSheet(self._get_spinbox_style())
        threshold_layout.addWidget(self.alarm_x_spin, 1, 1)

        threshold_layout.addWidget(QLabel("Y报警限值(mm):"), 2, 0)
        self.alarm_y_spin = QDoubleSpinBox()
        self.alarm_y_spin.setRange(0, 50)
        self.alarm_y_spin.setValue(5)
        self.alarm_y_spin.setStyleSheet(self._get_spinbox_style())
        threshold_layout.addWidget(self.alarm_y_spin, 2, 1)

        threshold_layout.addWidget(QLabel("A报警限值(°):"), 3, 0)
        self.alarm_a_spin = QDoubleSpinBox()
        self.alarm_a_spin.setRange(0, 45)
        self.alarm_a_spin.setValue(3)
        self.alarm_a_spin.setStyleSheet(self._get_spinbox_style())
        threshold_layout.addWidget(self.alarm_a_spin, 3, 1)

        layout.addWidget(threshold_group)

        # 时序设置组
        timing_group = QGroupBox("时序设置")
        timing_group.setStyleSheet(self._get_group_style())
        timing_layout = QGridLayout(timing_group)

        timing_layout.addWidget(QLabel("识别超时(ms):"), 0, 0)
        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(100, 10000)
        self.timeout_spin.setValue(3000)
        self.timeout_spin.setStyleSheet(self._get_spinbox_style())
        timing_layout.addWidget(self.timeout_spin, 0, 1)

        timing_layout.addWidget(QLabel("拍照延时(ms):"), 1, 0)
        self.photo_delay_spin = QSpinBox()
        self.photo_delay_spin.setRange(0, 1000)
        self.photo_delay_spin.setValue(100)
        self.photo_delay_spin.setStyleSheet(self._get_spinbox_style())
        timing_layout.addWidget(self.photo_delay_spin, 1, 1)

        timing_layout.addWidget(QLabel("真空延时(ms):"), 2, 0)
        self.vacuum_delay_spin = QSpinBox()
        self.vacuum_delay_spin.setRange(0, 1000)
        self.vacuum_delay_spin.setValue(200)
        self.vacuum_delay_spin.setStyleSheet(self._get_spinbox_style())
        timing_layout.addWidget(self.vacuum_delay_spin, 2, 1)

        layout.addWidget(timing_group)

        # 失败处理组
        fail_group = QGroupBox("失败处理")
        fail_group.setStyleSheet(self._get_group_style())
        fail_layout = QVBoxLayout(fail_group)

        fail_layout.addWidget(QLabel("重试次数:"))
        self.retry_count_spin = QSpinBox()
        self.retry_count_spin.setRange(0, 10)
        self.retry_count_spin.setValue(3)
        self.retry_count_spin.setStyleSheet(self._get_spinbox_style())
        fail_layout.addWidget(self.retry_count_spin)

        fail_layout.addWidget(QLabel("失败模式:"))
        self.fail_mode_combo = QComboBox()
        self.fail_mode_combo.addItems(["报警停止", "自动重试", "跳过继续"])
        self.fail_mode_combo.setStyleSheet(self._get_combo_style())
        fail_layout.addWidget(self.fail_mode_combo)

        layout.addWidget(fail_group)

        # 保存/加载按钮
        btn_layout = QHBoxLayout()
        self.btn_save_params = QPushButton("💾 保存参数")
        self.btn_save_params.setStyleSheet(self._get_button_style())
        self.btn_save_params.clicked.connect(self._save_paste_params)
        btn_layout.addWidget(self.btn_save_params)

        self.btn_load_params = QPushButton("📂 加载参数")
        self.btn_load_params.setStyleSheet(self._get_button_style())
        self.btn_load_params.clicked.connect(self._load_paste_params)
        btn_layout.addWidget(self.btn_load_params)

        layout.addLayout(btn_layout)
        layout.addStretch()

    def _setup_statusbar(self):
        """设置状态栏"""
        status_bar = QStatusBar()
        self.setStatusBar(status_bar)

        self.status_light = QLabel()
        self.status_light.setStyleSheet("background-color: #FF6666; border-radius: 10px; min-width: 20px; min-height: 20px;")
        self.status_light.setFixedSize(20, 20)
        status_bar.addWidget(self.status_light)

        self.status_label = QLabel("系统就绪")
        self.status_label.setStyleSheet(self._get_label_style(12, "#00FFFF"))
        status_bar.addWidget(self.status_label)

        self.fps_label = QLabel("FPS: 0")
        self.fps_label.setStyleSheet(self._get_label_style(14, "#00FF96"))
        status_bar.addPermanentWidget(self.fps_label)

    def _setup_connections(self):
        """设置信号连接"""
        self.image_canvas.roi_selected.connect(self._on_roi_selected)
        self.image_canvas.line_drawn.connect(self._on_line_drawn)
        self.image_canvas.contour_selected.connect(self._on_contour_selected)
        self.image_canvas.contour_drawn.connect(self._on_contour_drawn)
        self.image_canvas.auto_line_roi_selected.connect(self._on_auto_line_roi_selected)

    # ==================== 样式定义 ====================

    def _get_group_style(self):
        fs = self._get_scaled_font_size
        return f"""
            QGroupBox {{
                color: #00FFFF;
                font: bold {fs(12)}px Consolas;
                border: 2px solid #0066CC;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 5px;
            }}
        """

    def _get_button_style(self, color="#0088FF"):
        fs = self._get_scaled_font_size
        return f"""
            QPushButton {{
                background-color: rgba(0, 50, 100, 150);
                color: #00FFFF;
                border: 2px solid {color};
                border-radius: 6px;
                padding: 8px;
                font: bold {fs(11)}px Consolas;
                min-height: 28px;
            }}
            QPushButton:hover {{
                background-color: rgba(0, 100, 200, 200);
            }}
            QPushButton:disabled {{
                background-color: rgba(50, 50, 50, 150);
                color: #666666;
                border-color: #333333;
            }}
        """

    def _get_combo_style(self):
        fs = self._get_scaled_font_size
        return f"""
            QComboBox {{
                background-color: rgba(0, 50, 100, 150);
                color: #00FFFF;
                border: 2px solid #0088FF;
                border-radius: 6px;
                padding: 6px;
                font: {fs(11)}px Consolas;
                min-height: 28px;
            }}
            QComboBox QAbstractItemView {{
                background-color: rgba(10, 20, 40, 230);
                color: #00FFFF;
                font: {fs(11)}px Consolas;
            }}
        """

    def _get_table_style(self):
        fs = self._get_scaled_font_size
        return f"""
            QTableWidget {{
                background-color: rgba(10, 20, 40, 200);
                color: #00FFFF;
                border: 1px solid #0066CC;
                font: {fs(11)}px Consolas;
            }}
            QHeaderView::section {{
                background-color: rgba(0, 50, 100, 200);
                color: #00FFFF;
                border: 1px solid #0066CC;
                padding: 4px;
                font: bold {fs(11)}px Consolas;
            }}
        """

    def _get_slider_style(self):
        return """
            QSlider::groove:horizontal {
                background: rgba(0, 50, 100, 150);
                height: 10px;
                border-radius: 5px;
            }
            QSlider::handle:horizontal {
                background: #00FFFF;
                width: 20px;
                margin: -5px 0;
                border-radius: 10px;
            }
        """

    def _get_spinbox_style(self):
        fs = self._get_scaled_font_size
        return f"""
            QSpinBox, QDoubleSpinBox {{
                background-color: rgba(0, 50, 100, 150);
                color: #00FFFF;
                border: 2px solid #0088FF;
                border-radius: 6px;
                padding: 6px;
                font: {fs(11)}px Consolas;
            }}
        """

    def _get_tab_style(self):
        fs = self._get_scaled_font_size
        return f"""
            QTabWidget::pane {{
                border: 2px solid #0066CC;
                border-radius: 8px;
                background-color: rgba(10, 20, 40, 200);
            }}
            QTabBar::tab {{
                background-color: rgba(0, 50, 100, 150);
                color: #00FFFF;
                border: 2px solid #0066CC;
                border-radius: 6px;
                padding: 10px 15px;
                margin-right: 5px;
                font: bold {fs(11)}px Consolas;
            }}
            QTabBar::tab:selected {{
                background-color: rgba(0, 100, 200, 200);
                border-color: #00FFFF;
            }}
        """

    def _get_label_style(self, size=12, color="#00FFFF", bold=False):
        fs = self._get_scaled_font_size
        weight = "bold" if bold else "normal"
        return f"color: {color}; font: {weight} {fs(size)}px Consolas;"

    def _get_value_style(self, size=13, color="#00FF96", bold=True):
        fs = self._get_scaled_font_size
        weight = "bold" if bold else "normal"
        return f"color: {color}; font: {weight} {fs(size)}px Consolas; background: rgba(0,50,100,150); padding: 5px; border-radius: 5px;"

    def _get_text_edit_style(self):
        """获取文本编辑框样式"""
        fs = self._get_scaled_font_size
        return f"""
            QTextEdit {{
                color: #00FFFF;
                background-color: rgba(10, 20, 40, 200);
                border: 2px solid #0088FF;
                border-radius: 5px;
                padding: 8px;
                font: {fs(12)}px Consolas;
            }}
            QTextEdit:focus {{
                border-color: #00FFFF;
            }}
        """

    # ==================== 算法选择 ====================

    def _populate_algorithm_combo(self):
        """填充算法选择下拉菜单"""
        algorithms = VisionAlgorithm.get_available_algorithms()
        self.algorithm_combo.clear()
        for key, info in algorithms.items():
            self.algorithm_combo.addItem(info['name'], key)
            if key == 'fusion':
                self.algorithm_combo.setCurrentIndex(self.algorithm_combo.count() - 1)
        self._update_algorithm_desc()

    def _on_algorithm_changed(self):
        """算法选择改变时更新描述并刷新检测结果"""
        self._update_algorithm_desc()
        # 切换算法时进行预览
        self._preview_with_selected_algorithm()
        # 如果正在测试模式，立即使用新算法重新检测
        if hasattr(self, 'is_testing') and self.is_testing and hasattr(self, 'detection_timer') and self.detection_timer.isActive():
            self._perform_dynamic_detection()

    def _preview_with_selected_algorithm(self):
        """使用当前选择的算法进行预览检测"""
        try:
            if not hasattr(self, 'dual_line_template') or not hasattr(self, 'image_canvas'):
                return
            
            template_lines_left = []
            if self.dual_line_template.left_line_long:
                template_lines_left.append(self.dual_line_template.left_line_long)
            if self.dual_line_template.left_line_short:
                template_lines_left.append(self.dual_line_template.left_line_short)
            
            template_lines_right = []
            if self.dual_line_template.right_line_long:
                template_lines_right.append(self.dual_line_template.right_line_long)
            if self.dual_line_template.right_line_short:
                template_lines_right.append(self.dual_line_template.right_line_short)
            
            if not template_lines_left or not template_lines_right:
                return
                
            if not self.image_canvas.current_frame:
                return
            
            frame = self.image_canvas.current_frame
            selected_algorithm = self._get_selected_algorithm()
            
            left_lines = VisionAlgorithm.detect_lines_single_algorithm(
                frame, self.dual_line_template.left_roi, selected_algorithm, template_lines_left
            )
            right_lines = VisionAlgorithm.detect_lines_single_algorithm(
                frame, self.dual_line_template.right_roi, selected_algorithm, template_lines_right
            )
            
            matched_left = VisionAlgorithm.find_best_matching_lines(left_lines, template_lines_left)
            matched_right = VisionAlgorithm.find_best_matching_lines(right_lines, template_lines_right)
            
            self.image_canvas.detected_lines = []
            self.image_canvas.detected_rois = []
            
            if self.dual_line_template.left_roi:
                self.image_canvas.detected_rois.append((self.dual_line_template.left_roi, NeonStyle.NEON_GREEN))
            if self.dual_line_template.right_roi:
                self.image_canvas.detected_rois.append((self.dual_line_template.right_roi, NeonStyle.NEON_ORANGE))
            
            display_left = matched_left if len(matched_left) > 0 else left_lines
            display_right = matched_right if len(matched_right) > 0 else right_lines
            
            for line in display_left[:2]:
                self.image_canvas.detected_lines.append((line, NeonStyle.NEON_GREEN))
            for line in display_right[:2]:
                self.image_canvas.detected_lines.append((line, NeonStyle.NEON_ORANGE))
            
            self.image_canvas.update()
            
        except Exception as e:
            print(f"预览检测错误: {str(e)}")

    def _update_algorithm_desc(self):
        """更新算法描述"""
        if not hasattr(self, 'algorithm_desc') or not hasattr(self, 'algorithm_combo'):
            return
        algorithms = VisionAlgorithm.get_available_algorithms()
        current_key = self.algorithm_combo.currentData()
        if current_key in algorithms:
            self.algorithm_desc.setText(algorithms[current_key]['description'])

    def _get_selected_algorithm(self):
        """获取当前选择的算法"""
        return self.algorithm_combo.currentData()

    # ==================== 摄像头控制 ====================

    def _detect_cameras(self):
        """检测摄像头"""
        cameras = CameraUtils.detect_cameras()
        self.camera_combo.clear()
        self.camera_combo.addItem("请选择摄像头")
        for idx, w, h in cameras:
            self.camera_combo.addItem(f"摄像头 {idx} ({w}x{h})")

    def _toggle_camera(self):
        """切换摄像头"""
        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.stop()
            self.btn_open_cam.setText("📷 打开摄像头")
            self.status_label.setText("摄像头已关闭")
            self.status_light.setStyleSheet("background-color: #FF6666; border-radius: 10px; min-width: 20px; min-height: 20px;")
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
            self.status_light.setStyleSheet("background-color: #00FF96; border-radius: 10px; min-width: 20px; min-height: 20px;")

    def _on_frame_ready(self, frame):
        """帧就绪回调"""
        self.image_canvas.set_frame(frame)

    def _on_camera_error(self, error):
        """摄像头错误回调"""
        QMessageBox.critical(self, "摄像头错误", error)

    # ==================== 相机标定 ====================

    def _on_calib_method_changed(self, index):
        """标定方法改变"""
        self.checkerboard_widget.setVisible(index == 0)
        self.points_widget.setVisible(index == 1)

    def _capture_calib_image(self):
        """采集标定图像"""
        if self.image_canvas.current_frame is None:
            QMessageBox.warning(self, "警告", "请先打开摄像头")
            return

        pattern_size = (self.cb_cols_spin.value(), self.cb_rows_spin.value())
        ret, corners = CameraCalibration.detect_checkerboard(
            self.image_canvas.current_frame, pattern_size
        )

        if ret:
            self.calib_images.append(self.image_canvas.current_frame.copy())
            self.calib_image_count.setText(f"已采集: {len(self.calib_images)} 张")
            self.status_label.setText(f"标定图像采集成功 ({len(self.calib_images)}/15)")
        else:
            QMessageBox.warning(self, "警告", "未检测到棋盘格，请调整角度或距离")

    def _perform_calibration(self):
        """执行标定"""
        if len(self.calib_images) < 10:
            QMessageBox.warning(self, "警告", "至少需要10张标定图像")
            return

        try:
            pattern_size = (self.cb_cols_spin.value(), self.cb_rows_spin.value())
            square_size = self.square_size_spin.value()

            self.calibration_data, rms_error = CameraCalibration.calibrate_camera(
                self.calib_images, pattern_size, square_size
            )

            self._update_calib_display()
            self._save_current_session()

            QMessageBox.information(
                self, "标定成功",
                f"相机标定完成！\n重投影误差: {rms_error:.4f} 像素\n\n"
                f"标定精度评估：\n"
                f"- 误差 < 0.5px: 优秀\n"
                f"- 误差 < 1.0px: 良好\n"
                f"- 误差 > 1.0px: 需重新标定"
            )

        except Exception as e:
            QMessageBox.critical(self, "标定失败", f"错误: {str(e)}")

    def _update_calib_display(self):
        """更新标定显示"""
        if self.calibration_data.is_calibrated:
            self.calib_info.setText(
                f"状态: 已标定 ✓\n"
                f"重投影误差: {self.calibration_data.reprojection_error:.4f} px\n"
                f"像素/mm: X={self.calibration_data.pixel_per_mm_x:.2f}, Y={self.calibration_data.pixel_per_mm_y:.2f}"
            )
            self.calib_info.setStyleSheet(self._get_label_style(14, "#00FF96"))

    def _add_calib_point(self):
        """添加标定点"""
        row = self.calib_table.rowCount()
        self.calib_table.insertRow(row)
        self.calib_table.setItem(row, 0, QTableWidgetItem(str(row+1)))
        self.calib_table.setItem(row, 1, QTableWidgetItem("0.0"))
        self.calib_table.setItem(row, 2, QTableWidgetItem("0.0"))
        self.calib_table.setItem(row, 3, QTableWidgetItem("0.0"))

    def _clear_calib_points(self):
        """清除标定点"""
        self.calib_table.setRowCount(0)

    def _calculate_calibration_from_points(self):
        """从点对应计算标定"""
        if self.calib_table.rowCount() < 2:
            QMessageBox.warning(self, "警告", "至少需要2个标定点")
            return

        try:
            pixel_points = []
            physical_points = []

            for row in range(self.calib_table.rowCount()):
                px = float(self.calib_table.item(row, 1).text())
                py = float(self.calib_table.item(row, 2).text())
                phys = float(self.calib_table.item(row, 3).text())
                pixel_points.append((px, py))
                physical_points.append((phys, 0))

            self.calibration_data = CameraCalibration.calibrate_from_points(
                pixel_points, physical_points
            )

            self._update_calib_display()
            self._save_current_session()

            QMessageBox.information(self, "成功", "标定计算完成！")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"计算失败: {str(e)}")

    def _show_calibration_info(self):
        """显示标定原理说明"""
        info_text = """
        <h2>相机标定原理与流程</h2>

        <h3>一、针孔相机模型</h3>
        <p>相机成像遵循针孔模型，3D空间点通过投影变换映射到2D图像平面。</p>

        <h3>二、内参矩阵</h3>
        <p>K = | fx  0  cx |<br>
           &nbsp;&nbsp;&nbsp;&nbsp;|  0 fy  cy |<br>
           &nbsp;&nbsp;&nbsp;&nbsp;|  0  0   1 |</p>
        <ul>
            <li>fx, fy: 焦距（像素单位）</li>
            <li>cx, cy: 主点坐标（图像中心）</li>
        </ul>

        <h3>三、标定流程</h3>
        <ol>
            <li>准备棋盘格标定板（推荐9x6内角点）</li>
            <li>从不同角度拍摄至少15-20张图像</li>
            <li>检测每张图像中的角点</li>
            <li>求解相机内参和畸变系数</li>
            <li>计算重投影误差评估精度</li>
        </ol>

        <h3>四、精度评估</h3>
        <ul>
            <li>重投影误差 < 0.5px: 优秀</li>
            <li>重投影误差 < 1.0px: 良好</li>
            <li>重投影误差 > 1.0px: 需重新标定</li>
        </ul>
        """

        msg = QMessageBox(self)
        msg.setWindowTitle("相机标定原理说明")
        msg.setTextFormat(Qt.RichText)
        msg.setText(info_text)
        fs = self._get_scaled_font_size
        msg.setStyleSheet(f"""
            QMessageBox {{
                background-color: #0A0F1E;
            }}
            QMessageBox QLabel {{
                color: #00FFFF;
                font: {fs(14)}px Consolas;
            }}
        """)
        msg.exec_()

    def _save_calibration(self):
        """保存标定数据"""
        if not self.calibration_data.is_calibrated:
            QMessageBox.warning(self, "警告", "请先完成标定")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "保存标定数据", "", "JSON (*.json)"
        )
        if filename:
            TemplateManager.save_calibration(filename, self.calibration_data)
            QMessageBox.information(self, "成功", f"标定数据已保存到:\n{filename}")

    def _load_calibration(self):
        """加载标定数据"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "加载标定数据", "", "JSON (*.json)"
        )
        if filename:
            self.calibration_data = TemplateManager.load_calibration(filename)
            self._update_calib_display()
            self._save_current_session()
            QMessageBox.information(self, "成功", "标定数据已加载")

    # ==================== 模式切换 ====================

    def _change_mode(self, index):
        """切换模式"""
        self.current_mode = "shape" if index == 1 else "line"
        if self.current_mode == "line":
            self.tab_widget.setCurrentIndex(0)
        else:
            self.tab_widget.setCurrentIndex(1)

    # ==================== 双直线模板设置 ====================

    def _start_draw_line(self, side):
        """开始绘制直线"""
        self.line_drawing_side = side
        self.image_canvas.start_drawing("line")
        self.status_label.setText(f"绘制直线: {side}")

    def _start_auto_detect_line(self, side):
        """开始框选区域以自动检测直线"""
        self.line_drawing_side = side
        self.image_canvas.start_drawing("auto_line_roi")
        self.status_label.setText(f"框选区域以自动检测: {side}")

    def _clear_line(self, side):
        """清除直线"""
        if side == "left_long":
            self.image_canvas.save_state("line_long", self.image_canvas.left_line_long, "清除左长边")
            self.image_canvas.left_line_long = None
            self.left_long_info.setText("长边: 未设置")
        elif side == "left_short":
            self.image_canvas.save_state("line_short", self.image_canvas.left_line_short, "清除左短边")
            self.image_canvas.left_line_short = None
            self.left_short_info.setText("短边: 未设置")
        elif side == "right_long":
            self.image_canvas.save_state("right_line_long", self.image_canvas.right_line_long, "清除右长边")
            self.image_canvas.right_line_long = None
            self.right_long_info.setText("长边: 未设置")
        elif side == "right_short":
            self.image_canvas.save_state("right_line_short", self.image_canvas.right_line_short, "清除右短边")
            self.image_canvas.right_line_short = None
            self.right_short_info.setText("短边: 未设置")

        self._update_line_template_status()
        self.image_canvas.update()

    def _start_draw_roi(self, side):
        """开始绘制ROI区域"""
        self.roi_drawing_side = side
        self.image_canvas.start_drawing("roi")
        self.status_label.setText(f"绘制ROI: {side}")

    def _undo_line_operation(self):
        """撤销上一步操作"""
        result = self.image_canvas.undo()
        if result:
            self.status_label.setText(f"已撤销: {result}")
            self._update_line_ui()
            self._update_line_template_status()
        else:
            self.status_label.setText("没有可撤销的操作")

    def _update_line_ui(self):
        """更新直线UI显示"""
        try:
            # 更新左侧长边
            if self.image_canvas.left_line_long and self._is_valid_line(self.image_canvas.left_line_long):
                angle = VisionAlgorithm.calculate_line_angle(self.image_canvas.left_line_long)
                self.left_long_info.setText(f"长边: θ={angle:.1f}°")
            else:
                self.left_long_info.setText("长边: 未设置")

            # 更新左侧短边
            if self.image_canvas.left_line_short and self._is_valid_line(self.image_canvas.left_line_short):
                angle = VisionAlgorithm.calculate_line_angle(self.image_canvas.left_line_short)
                self.left_short_info.setText(f"短边: θ={angle:.1f}°")
            else:
                self.left_short_info.setText("短边: 未设置")

            # 更新右侧长边
            if self.image_canvas.right_line_long and self._is_valid_line(self.image_canvas.right_line_long):
                angle = VisionAlgorithm.calculate_line_angle(self.image_canvas.right_line_long)
                self.right_long_info.setText(f"长边: θ={angle:.1f}°")
            else:
                self.right_long_info.setText("长边: 未设置")

            # 更新右侧短边
            if self.image_canvas.right_line_short and self._is_valid_line(self.image_canvas.right_line_short):
                angle = VisionAlgorithm.calculate_line_angle(self.image_canvas.right_line_short)
                self.right_short_info.setText(f"短边: θ={angle:.1f}°")
            else:
                self.right_short_info.setText("短边: 未设置")

            # 更新左侧ROI
            if self.image_canvas.left_roi and self._is_valid_roi(self.image_canvas.left_roi):
                rect = self.image_canvas.left_roi
                self.left_roi_info.setText(f"区域: ({rect[0]:.0f},{rect[1]:.0f})-({rect[2]:.0f},{rect[3]:.0f})")
            else:
                self.left_roi_info.setText("区域: 未设置")

            # 更新右侧ROI
            if self.image_canvas.right_roi and self._is_valid_roi(self.image_canvas.right_roi):
                rect = self.image_canvas.right_roi
                self.right_roi_info.setText(f"区域: ({rect[0]:.0f},{rect[1]:.0f})-({rect[2]:.0f},{rect[3]:.0f})")
            else:
                self.right_roi_info.setText("区域: 未设置")
                
        except Exception as e:
            import traceback
            error_msg = f"更新直线UI出错: {str(e)}\n\n{traceback.format_exc()}"
            QMessageBox.critical(self, "错误", error_msg)

    def _is_valid_line(self, line):
        """检查直线数据是否有效"""
        if not line:
            return False
        if not isinstance(line, tuple) or len(line) != 2:
            return False
        start, end = line
        if not isinstance(start, (tuple, list)) or len(start) != 2:
            return False
        if not isinstance(end, (tuple, list)) or len(end) != 2:
            return False
        return True

    def _is_valid_roi(self, roi):
        """检查ROI数据是否有效"""
        if not roi:
            return False
        if not isinstance(roi, (tuple, list)) or len(roi) != 4:
            return False
        return True

    def _on_line_drawn(self, line_data):
        """直线绘制完成回调"""
        sx, sy, ex, ey = line_data
        line = ((sx, sy), (ex, ey))

        if self.line_drawing_side == "left_long":
            self.image_canvas.save_state("line_long", self.image_canvas.left_line_long, "绘制左长边")
            self.image_canvas.left_line_long = line
            angle = VisionAlgorithm.calculate_line_angle(line)
            length = math.sqrt((ex-sx)**2 + (ey-sy)**2)
            self.left_long_info.setText(f"长边: θ={angle:.1f}°, L={length:.1f}px")

        elif self.line_drawing_side == "left_short":
            self.image_canvas.save_state("line_short", self.image_canvas.left_line_short, "绘制左短边")
            self.image_canvas.left_line_short = line
            angle = VisionAlgorithm.calculate_line_angle(line)
            length = math.sqrt((ex-sx)**2 + (ey-sy)**2)
            self.left_short_info.setText(f"短边: θ={angle:.1f}°, L={length:.1f}px")

        elif self.line_drawing_side == "right_long":
            self.image_canvas.save_state("right_line_long", self.image_canvas.right_line_long, "绘制右长边")
            self.image_canvas.right_line_long = line
            angle = VisionAlgorithm.calculate_line_angle(line)
            length = math.sqrt((ex-sx)**2 + (ey-sy)**2)
            self.right_long_info.setText(f"长边: θ={angle:.1f}°, L={length:.1f}px")

        elif self.line_drawing_side == "right_short":
            self.image_canvas.save_state("right_line_short", self.image_canvas.right_line_short, "绘制右短边")
            self.image_canvas.right_line_short = line
            angle = VisionAlgorithm.calculate_line_angle(line)
            length = math.sqrt((ex-sx)**2 + (ey-sy)**2)
            self.right_short_info.setText(f"短边: θ={angle:.1f}°, L={length:.1f}px")

        self.line_drawing_side = None
        self._update_line_template_status()
        self.image_canvas.update()

    def _on_auto_line_roi_selected(self, rect):
        """框选区域后自动检测直线"""
        try:
            # 检查帧是否存在（使用 is None 而不是 not，避免 numpy 数组判断问题）
            if self.image_canvas.current_frame is None:
                QMessageBox.warning(self, "警告", "请先打开摄像头")
                return

            frame = self.image_canvas.current_frame

            if frame is None or frame.size == 0:
                QMessageBox.warning(self, "警告", "当前帧无效")
                return

            x1, y1, x2, y2 = rect

            # 检查区域是否有效
            if x1 >= x2 or y1 >= y2:
                QMessageBox.warning(self, "警告", "请选择有效的区域")
                return

            # 检查区域是否在图像范围内
            h, w = frame.shape[:2]
            if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                QMessageBox.warning(self, "警告", "选择的区域超出图像范围")
                return

            # 使用低阈值Canny算法检测直线（与手动测试保持一致）
            detected_lines = VisionAlgorithm.detect_lines_single_algorithm(frame, rect, 'canny_low', [])

            if not detected_lines:
                QMessageBox.warning(self, "警告", "在该区域未检测到直线，请重新框选")
                return

            # 选择最长的直线作为结果
            best_line = detected_lines[0]
            max_length = 0
            for line in detected_lines:
                if not line or len(line) != 2:
                    continue
                (lx1, ly1), (lx2, ly2) = line
                length = math.sqrt((lx2-lx1)**2 + (ly2-ly1)**2)
                if length > max_length:
                    max_length = length
                    best_line = line

            # 设置检测到的直线
            sx, sy = best_line[0]
            ex, ey = best_line[1]

            if self.line_drawing_side == "left_long":
                self.image_canvas.save_state("line_long", self.image_canvas.left_line_long, "自动检测左长边")
                self.image_canvas.left_line_long = best_line
                angle = VisionAlgorithm.calculate_line_angle(best_line)
                self.left_long_info.setText(f"长边: θ={angle:.1f}°, L={max_length:.1f}px")
            elif self.line_drawing_side == "left_short":
                self.image_canvas.save_state("line_short", self.image_canvas.left_line_short, "自动检测左短边")
                self.image_canvas.left_line_short = best_line
                angle = VisionAlgorithm.calculate_line_angle(best_line)
                self.left_short_info.setText(f"短边: θ={angle:.1f}°, L={max_length:.1f}px")
            elif self.line_drawing_side == "right_long":
                self.image_canvas.save_state("right_line_long", self.image_canvas.right_line_long, "自动检测右长边")
                self.image_canvas.right_line_long = best_line
                angle = VisionAlgorithm.calculate_line_angle(best_line)
                self.right_long_info.setText(f"长边: θ={angle:.1f}°, L={max_length:.1f}px")
            elif self.line_drawing_side == "right_short":
                self.image_canvas.save_state("right_line_short", self.image_canvas.right_line_short, "自动检测右短边")
                self.image_canvas.right_line_short = best_line
                angle = VisionAlgorithm.calculate_line_angle(best_line)
                self.right_short_info.setText(f"短边: θ={angle:.1f}°, L={max_length:.1f}px")
            
            self.line_drawing_side = None
            self._update_line_template_status()
            self.image_canvas.update()
            self.status_label.setText("自动检测直线完成")
            
        except Exception as e:
            import traceback
            error_msg = f"自动检测直线出错: {str(e)}\n\n{traceback.format_exc()}"
            QMessageBox.critical(self, "错误", error_msg)
            self.line_drawing_side = None
            self.status_label.setText("检测出错")

    def _on_roi_selected(self, rect):
        """ROI区域选择完成回调"""
        # 双直线模板的ROI
        if self.roi_drawing_side == "left":
            self.image_canvas.save_state("left_roi", self.image_canvas.left_roi, "设置左ROI")
            self.image_canvas.left_roi = rect
            self.left_roi_info.setText(f"区域: ({rect[0]:.0f},{rect[1]:.0f})-({rect[2]:.0f},{rect[3]:.0f})")
            self.roi_drawing_side = None
            self._update_line_template_status()
        elif self.roi_drawing_side == "right":
            self.image_canvas.save_state("right_roi", self.image_canvas.right_roi, "设置右ROI")
            self.image_canvas.right_roi = rect
            self.right_roi_info.setText(f"区域: ({rect[0]:.0f},{rect[1]:.0f})-({rect[2]:.0f},{rect[3]:.0f})")
            self.roi_drawing_side = None
            self._update_line_template_status()
        # 外形模板的ROI
        elif self.contour_roi_drawing_side == "left":
            self.image_canvas.save_state("left_contour_roi", self.image_canvas.left_contour_roi, "设置左轮廓ROI")
            self.image_canvas.left_contour_roi = rect
            self.left_contour_roi_info.setText(f"区域: ({rect[0]:.0f},{rect[1]:.0f})-({rect[2]:.0f},{rect[3]:.0f})")
            self.contour_roi_drawing_side = None
            self._update_shape_template_status()
        elif self.contour_roi_drawing_side == "right":
            self.image_canvas.save_state("right_contour_roi", self.image_canvas.right_contour_roi, "设置右轮廓ROI")
            self.image_canvas.right_contour_roi = rect
            self.right_contour_roi_info.setText(f"区域: ({rect[0]:.0f},{rect[1]:.0f})-({rect[2]:.0f},{rect[3]:.0f})")
            self.contour_roi_drawing_side = None
            self._update_shape_template_status()

        self.image_canvas.update()

    def _update_line_template_status(self):
        """更新双直线模板状态"""
        has_all_lines = (
            self.image_canvas.left_line_long is not None and
            self.image_canvas.left_line_short is not None and
            self.image_canvas.right_line_long is not None and
            self.image_canvas.right_line_short is not None
        )

        has_all_rois = (
            self.image_canvas.left_roi is not None and
            self.image_canvas.right_roi is not None
        )

        if has_all_lines and has_all_rois:
            self.line_template_status.setText("模板状态: 已完成 ✓")
            self.line_template_status.setStyleSheet(self._get_label_style(12, "#00FF96", True))
            self.btn_save_line_template.setEnabled(True)
        elif has_all_lines:
            self.line_template_status.setText("模板状态: 需设置区域")
            self.line_template_status.setStyleSheet(self._get_label_style(12, "#FF9600", True))
            self.btn_save_line_template.setEnabled(False)
        else:
            self.line_template_status.setText("模板状态: 未完成")
            self.line_template_status.setStyleSheet(self._get_label_style(12, "#FF6666", True))
            self.btn_save_line_template.setEnabled(False)

        # 只要画布上有线条和区域，就可以启动测试（不需要先保存）
        self.btn_test_line.setEnabled(has_all_lines and has_all_rois)

    def _save_line_template(self):
        """保存双直线模板（支持选择路径和重命名）"""
        if not self.image_canvas.left_line_long or not self.image_canvas.left_line_short:
            QMessageBox.warning(self, "警告", "请先设置左侧的长边和短边")
            return

        if not self.image_canvas.right_line_long or not self.image_canvas.right_line_short:
            QMessageBox.warning(self, "警告", "请先设置右侧的长边和短边")
            return

        if not self.image_canvas.left_roi or not self.image_canvas.right_roi:
            QMessageBox.warning(self, "警告", "请先设置左右监测区域")
            return

        # 保存模板到内存
        self.dual_line_template.left_line_long = self.image_canvas.left_line_long
        self.dual_line_template.left_line_short = self.image_canvas.left_line_short
        self.dual_line_template.right_line_long = self.image_canvas.right_line_long
        self.dual_line_template.right_line_short = self.image_canvas.right_line_short
        self.dual_line_template.left_roi = self.image_canvas.left_roi
        self.dual_line_template.right_roi = self.image_canvas.right_roi
        self.dual_line_template.valid = True

        # 计算基准数据
        left_intersection = VisionAlgorithm.calculate_line_intersection(
            self.image_canvas.left_line_long, self.image_canvas.left_line_short
        )
        right_intersection = VisionAlgorithm.calculate_line_intersection(
            self.image_canvas.right_line_long, self.image_canvas.right_line_short
        )

        base_x = base_y = base_angle = 0
        if left_intersection and right_intersection:
            base_x = (left_intersection[0] + right_intersection[0]) / 2
            base_y = (left_intersection[1] + right_intersection[1]) / 2
            left_angle = VisionAlgorithm.calculate_line_angle(self.image_canvas.left_line_long)
            right_angle = VisionAlgorithm.calculate_line_angle(self.image_canvas.right_line_long)
            base_angle = (left_angle + right_angle) / 2

        # 弹出保存对话框
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存双直线模板",
            DEFAULT_LINE_TEMPLATE_FILE,
            "JSON文件 (*.json);;所有文件 (*.*)"
        )

        if file_path:
            # 确保文件扩展名正确
            if not file_path.lower().endswith('.json'):
                file_path += '.json'

            # 保存到用户选择的路径
            TemplateManager.save_dual_line_template(file_path, self.dual_line_template)

            # 同时保存到默认位置（用于下次启动加载）
            TemplateManager.save_dual_line_template(DEFAULT_LINE_TEMPLATE_FILE, self.dual_line_template)

            self.btn_test_line.setEnabled(True)

            QMessageBox.information(
                self, "保存成功",
                f"双直线模板已保存！\n\n"
                f"保存位置: {file_path}\n\n"
                f"基准数据：\n"
                f"中心: ({base_x:.2f}, {base_y:.2f}) px\n"
                f"角度: {base_angle:.2f}°\n\n"
                f"同时已备份到默认位置供下次启动使用"
            )

    def _load_line_template_to_canvas(self):
        """加载模板到画布"""
        self.image_canvas.left_line_long = self.dual_line_template.left_line_long
        self.image_canvas.left_line_short = self.dual_line_template.left_line_short
        self.image_canvas.right_line_long = self.dual_line_template.right_line_long
        self.image_canvas.right_line_short = self.dual_line_template.right_line_short
        self.image_canvas.left_roi = self.dual_line_template.left_roi
        self.image_canvas.right_roi = self.dual_line_template.right_roi
        self._update_line_ui()

    def _load_line_template(self):
        """加载双直线模板"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "加载双直线模板", "", "JSON (*.json)"
        )
        if filename:
            try:
                # 验证文件是否存在
                if not os.path.exists(filename):
                    QMessageBox.warning(self, "警告", "文件不存在")
                    return
                
                # 验证文件是否可读
                try:
                    with open(filename, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if not content.strip():
                            QMessageBox.warning(self, "警告", "文件为空")
                            return
                except Exception as e:
                    QMessageBox.warning(self, "警告", f"无法读取文件: {str(e)}")
                    return
                
                # 加载模板
                self.dual_line_template = TemplateManager.load_dual_line_template(filename)
                
                # 检查模板是否有效
                if self.dual_line_template is None:
                    QMessageBox.warning(self, "警告", "模板加载失败")
                    return
                
                # 加载到画布
                self._load_line_template_to_canvas()
                self._update_line_template_status()
                self.image_canvas.update()
                QMessageBox.information(self, "成功", f"双直线模板已加载:\n{filename}")
                
            except Exception as e:
                import traceback
                error_msg = f"加载失败: {str(e)}\n\n{traceback.format_exc()}"
                QMessageBox.critical(self, "错误", error_msg)

    def _clear_all_line_template(self):
        """清除所有双直线模板"""
        self.image_canvas.clear_all()
        self.dual_line_template = DualLineTemplate()
        self.left_long_info.setText("长边: 未设置")
        self.left_short_info.setText("短边: 未设置")
        self.right_long_info.setText("长边: 未设置")
        self.right_short_info.setText("短边: 未设置")
        self.left_roi_info.setText("区域: 未设置")
        self.right_roi_info.setText("区域: 未设置")
        self._update_line_template_status()

    def _test_line_detection(self):
        """测试双直线检测 - 使用画布上的当前设置（不需要先保存）"""
        print("\n" + "="*60)
        print("[手动测试启动]")
        print("="*60)
        
        # 检查画布上是否有设置
        has_all_lines = (
            self.image_canvas.left_line_long is not None and
            self.image_canvas.left_line_short is not None and
            self.image_canvas.right_line_long is not None and
            self.image_canvas.right_line_short is not None
        )
        has_all_rois = (
            self.image_canvas.left_roi is not None and
            self.image_canvas.right_roi is not None
        )
        
        print(f"[检查] 所有线条已设置: {has_all_lines}")
        print(f"[检查] 所有ROI已设置: {has_all_rois}")
        
        if has_all_lines:
            print(f"[模板] 左长边: {self.image_canvas.left_line_long}")
            print(f"[模板] 左短边: {self.image_canvas.left_line_short}")
            print(f"[模板] 右长边: {self.image_canvas.right_line_long}")
            print(f"[模板] 右短边: {self.image_canvas.right_line_short}")
        
        if has_all_rois:
            print(f"[模板] 左ROI: {self.image_canvas.left_roi}")
            print(f"[模板] 右ROI: {self.image_canvas.right_roi}")
        
        if not has_all_lines or not has_all_rois:
            QMessageBox.warning(self, "警告", "请先设置左右侧的长短边和检测区域")
            print("[错误] 模板设置不完整，无法启动测试")
            return

        if self.image_canvas.current_frame is None:
            QMessageBox.warning(self, "警告", "请先打开摄像头")
            print("[错误] 没有当前帧，无法启动测试")
            return
        
        print(f"[帧信息] 帧大小: {self.image_canvas.current_frame.shape}")

        # 清除之前的基准值
        if hasattr(self, 'test_base_x'):
            del self.test_base_x
        if hasattr(self, 'test_base_y'):
            del self.test_base_y
        if hasattr(self, 'test_base_angle'):
            del self.test_base_angle

        # 计算基准值：使用画布上的模板设置作为基准
        left_center = None
        right_center = None
        left_angle = 0
        right_angle = 0
        
        # 计算左侧交点或中心
        if self.image_canvas.left_line_long and self.image_canvas.left_line_short:
            left_intersection = VisionAlgorithm.calculate_line_intersection(
                self.image_canvas.left_line_long, self.image_canvas.left_line_short
            )
            if left_intersection:
                left_center = left_intersection
            else:
                left_center = VisionAlgorithm.calculate_line_center(self.image_canvas.left_line_long)
        elif self.image_canvas.left_line_long:
            left_center = VisionAlgorithm.calculate_line_center(self.image_canvas.left_line_long)
        
        # 计算右侧交点或中心
        if self.image_canvas.right_line_long and self.image_canvas.right_line_short:
            right_intersection = VisionAlgorithm.calculate_line_intersection(
                self.image_canvas.right_line_long, self.image_canvas.right_line_short
            )
            if right_intersection:
                right_center = right_intersection
            else:
                right_center = VisionAlgorithm.calculate_line_center(self.image_canvas.right_line_long)
        elif self.image_canvas.right_line_long:
            right_center = VisionAlgorithm.calculate_line_center(self.image_canvas.right_line_long)
        
        # 计算基准中心点
        if left_center and right_center:
            self.test_base_x = (left_center[0] + right_center[0]) / 2
            self.test_base_y = (left_center[1] + right_center[1]) / 2
        elif left_center:
            self.test_base_x = left_center[0]
            self.test_base_y = left_center[1]
        elif right_center:
            self.test_base_x = right_center[0]
            self.test_base_y = right_center[1]
        else:
            self.test_base_x = 0
            self.test_base_y = 0
        
        # 计算基准角度
        if self.image_canvas.left_line_long:
            left_angle = VisionAlgorithm.calculate_line_angle(self.image_canvas.left_line_long)
        if self.image_canvas.right_line_long:
            right_angle = VisionAlgorithm.calculate_line_angle(self.image_canvas.right_line_long)
        self.test_base_angle = (left_angle + right_angle) / 2
        
        print(f"[基准值] X: {self.test_base_x:.1f}, Y: {self.test_base_y:.1f}, Angle: {self.test_base_angle:.1f}°")

        # 默认使用第一个算法
        try:
            self.algorithm_combo.setCurrentIndex(0)
            print(f"[算法] 默认使用: {self.algorithm_combo.currentText()}")
        except Exception as e:
            print(f"[警告] 设置默认算法失败: {e}")

        self.current_mode = "line"
        # 进入测试模式
        self.image_canvas.enter_test_mode()

        # 启动动态检测
        self.detection_timer.start()
        print(f"[启动] 动态检测定时器已启动 (间隔: {self.detection_timer.interval()}ms)")

        # 更新按钮状态
        self.btn_test_line.setText("⏹ 停止检测")
        self.btn_test_line.clicked.disconnect()
        self.btn_test_line.clicked.connect(self._exit_test_mode)
        
        print("[完成] 手动测试模式已启动")
        print("="*60)

    def _perform_detection_dispatcher(self):
        """动态检测调度器 - 根据当前模式调用相应检测方法"""
        if self.current_mode == "line":
            self._perform_dynamic_detection()
        elif self.current_mode == "shape":
            self._perform_shape_dynamic_detection()

    def _perform_dynamic_detection(self):
        """动态实时检测 - 区域随纸张变换，保持相对位置不变"""
        try:
            print("\n" + "-"*40)
            print("[动态检测] 开始检测")
            
            # 检查当前帧是否有效
            if self.image_canvas.current_frame is None or self.image_canvas.current_frame.size == 0:
                print("[动态检测] 错误: 没有当前帧或帧为空")
                return

            frame = self.image_canvas.current_frame
            h, w = frame.shape[:2]
            print(f"[动态检测] 帧大小: {frame.shape}")

            # 获取模板设置
            template_lines_left = []
            template_lines_right = []
            if self.image_canvas.left_line_long:
                template_lines_left.append(('long', self.image_canvas.left_line_long))
            if self.image_canvas.left_line_short:
                template_lines_left.append(('short', self.image_canvas.left_line_short))
            if self.image_canvas.right_line_long:
                template_lines_right.append(('long', self.image_canvas.right_line_long))
            if self.image_canvas.right_line_short:
                template_lines_right.append(('short', self.image_canvas.right_line_short))

            # 获取模板区域
            template_left_roi = self.image_canvas.left_roi
            template_right_roi = self.image_canvas.right_roi
            
            # 检查是否有完整的设置
            if not template_lines_left or not template_lines_right or not template_left_roi or not template_right_roi:
                self.status_label.setText("模板未设置完整，无法检测")
                print("[动态检测] 错误: 模板未设置完整")
                return

            print(f"[动态检测] 模板左ROI: {template_left_roi}")
            print(f"[动态检测] 模板右ROI: {template_right_roi}")
            print(f"[动态检测] 模板左直线: {len(template_lines_left)} 条")
            print(f"[动态检测] 模板右直线: {len(template_lines_right)} 条")
            
            # 获取当前选择的算法
            selected_algorithm = self._get_selected_algorithm()
            print(f"[动态检测] 使用算法: {selected_algorithm}")
            
            # 计算模板中心点（基准位置）
            template_center_x = 0
            template_center_y = 0
            template_line_count = 0
            
            for _, line in template_lines_left + template_lines_right:
                center = VisionAlgorithm.calculate_line_center(line)
                template_center_x += center[0]
                template_center_y += center[1]
                template_line_count += 1
            
            if template_line_count > 0:
                template_center_x /= template_line_count
                template_center_y /= template_line_count
            print(f"[动态检测] 模板中心点: ({template_center_x:.1f}, {template_center_y:.1f})")
            
            # 第一步：使用模板区域进行初始检测
            detected_left_lines_raw = []
            detected_right_lines_raw = []
            
            if template_left_roi:
                try:
                    detected_left_lines_raw = VisionAlgorithm.detect_lines_single_algorithm(
                        frame, template_left_roi, selected_algorithm, [line for _, line in template_lines_left]
                    )
                    print(f"[动态检测] 左侧区域检测完成，发现 {len(detected_left_lines_raw)} 条原始直线")
                except Exception as e:
                    print(f"[动态检测] 左检测异常: {e}")
            
            if template_right_roi:
                try:
                    detected_right_lines_raw = VisionAlgorithm.detect_lines_single_algorithm(
                        frame, template_right_roi, selected_algorithm, [line for _, line in template_lines_right]
                    )
                    print(f"[动态检测] 右侧区域检测完成，发现 {len(detected_right_lines_raw)} 条原始直线")
                except Exception as e:
                    print(f"[动态检测] 右检测异常: {e}")
            
            # 第二步：模板匹配找到最佳匹配线条
            detected_left_lines = []
            detected_right_lines = []
            
            for line_type, template_line in template_lines_left:
                if detected_left_lines_raw:
                    matched_lines = VisionAlgorithm.find_best_matching_lines(
                        detected_left_lines_raw, [template_line]
                    )
                    if matched_lines:
                        detected_left_lines.append((line_type, matched_lines[0]))
            
            for line_type, template_line in template_lines_right:
                if detected_right_lines_raw:
                    matched_lines = VisionAlgorithm.find_best_matching_lines(
                        detected_right_lines_raw, [template_line]
                    )
                    if matched_lines:
                        detected_right_lines.append((line_type, matched_lines[0]))
            
            print(f"[动态检测] 左侧匹配到 {len(detected_left_lines)} 条直线")
            for line_type, line in detected_left_lines:
                angle = VisionAlgorithm.calculate_line_angle(line)
                length = math.sqrt((line[1][0]-line[0][0])**2 + (line[1][1]-line[0][1])**2)
                print(f"  - {line_type}: {line}, 角度: {angle:.1f}°, 长度: {length:.1f}px")
            
            print(f"[动态检测] 右侧匹配到 {len(detected_right_lines)} 条直线")
            for line_type, line in detected_right_lines:
                angle = VisionAlgorithm.calculate_line_angle(line)
                length = math.sqrt((line[1][0]-line[0][0])**2 + (line[1][1]-line[0][1])**2)
                print(f"  - {line_type}: {line}, 角度: {angle:.1f}°, 长度: {length:.1f}px")
            
            # 第三步：计算纸张偏移量（基于检测结果）
            all_detected_lines = [line for _, line in detected_left_lines + detected_right_lines]
            
            # 计算检测中心点
            if all_detected_lines:
                det_center_x = sum(VisionAlgorithm.calculate_line_center(line)[0] for line in all_detected_lines) / len(all_detected_lines)
                det_center_y = sum(VisionAlgorithm.calculate_line_center(line)[1] for line in all_detected_lines) / len(all_detected_lines)
                
                # 计算偏移量
                dx = det_center_x - template_center_x
                dy = det_center_y - template_center_y
                print(f"[动态检测] 纸张偏移: dx={dx:.2f}px, dy={dy:.2f}px")
            else:
                dx = 0
                dy = 0
                det_center_x = template_center_x
                det_center_y = template_center_y
                print(f"[动态检测] 未检测到线条，偏移量为0")
            
            # 第四步：变换检测区域（随纸张移动）
            # 保持区域与线条的相对位置不变
            transformed_left_roi = None
            transformed_right_roi = None
            
            if template_left_roi:
                x1, y1, x2, y2 = template_left_roi
                transformed_left_roi = (
                    x1 + dx, y1 + dy,
                    x2 + dx, y2 + dy
                )
                # 限制在图像范围内
                transformed_left_roi = (
                    max(0, transformed_left_roi[0]),
                    max(0, transformed_left_roi[1]),
                    min(w, transformed_left_roi[2]),
                    min(h, transformed_left_roi[3])
                )
            
            if template_right_roi:
                x1, y1, x2, y2 = template_right_roi
                transformed_right_roi = (
                    x1 + dx, y1 + dy,
                    x2 + dx, y2 + dy
                )
                transformed_right_roi = (
                    max(0, transformed_right_roi[0]),
                    max(0, transformed_right_roi[1]),
                    min(w, transformed_right_roi[2]),
                    min(h, transformed_right_roi[3])
                )
            
            print(f"[动态检测] 变换后左ROI: {transformed_left_roi}")
            print(f"[动态检测] 变换后右ROI: {transformed_right_roi}")
            
            # 第五步：使用变换后的区域进行二次检测（提高精度）
            refined_left_lines = []
            refined_right_lines = []
            
            if transformed_left_roi and len(detected_left_lines) < len(template_lines_left):
                try:
                    refined_lines = VisionAlgorithm.detect_lines_single_algorithm(
                        frame, transformed_left_roi, selected_algorithm, [line for _, line in template_lines_left]
                    )
                    print(f"[动态检测] 左侧二次检测发现 {len(refined_lines)} 条直线")
                    for line_type, template_line in template_lines_left:
                        found = False
                        for dt, dl in detected_left_lines:
                            if dt == line_type:
                                found = True
                                break
                        if not found and refined_lines:
                            matched_lines = VisionAlgorithm.find_best_matching_lines(refined_lines, [template_line])
                            if matched_lines:
                                refined_left_lines.append((line_type, matched_lines[0]))
                except Exception as e:
                    print(f"[动态检测] 左侧二次检测异常: {e}")
            
            if transformed_right_roi and len(detected_right_lines) < len(template_lines_right):
                try:
                    refined_lines = VisionAlgorithm.detect_lines_single_algorithm(
                        frame, transformed_right_roi, selected_algorithm, [line for _, line in template_lines_right]
                    )
                    print(f"[动态检测] 右侧二次检测发现 {len(refined_lines)} 条直线")
                    for line_type, template_line in template_lines_right:
                        found = False
                        for dt, dl in detected_right_lines:
                            if dt == line_type:
                                found = True
                                break
                        if not found and refined_lines:
                            matched_lines = VisionAlgorithm.find_best_matching_lines(refined_lines, [template_line])
                            if matched_lines:
                                refined_right_lines.append((line_type, matched_lines[0]))
                except Exception as e:
                    print(f"[动态检测] 右侧二次检测异常: {e}")
            
            # 合并检测结果
            final_left_lines = []
            final_right_lines = []
            
            # 左侧：先使用检测结果，再补充二次检测结果，最后使用模板
            detected_types = {dt for dt, _ in detected_left_lines}
            for line_type, template_line in template_lines_left:
                if line_type in detected_types:
                    for dt, dl in detected_left_lines:
                        if dt == line_type:
                            # 对检测到的线条应用偏移，保持与区域的相对位置
                            offset_line = (
                                (dl[0][0], dl[0][1]),
                                (dl[1][0], dl[1][1])
                            )
                            final_left_lines.append((line_type, offset_line))
                            break
                else:
                    # 检查二次检测结果
                    found = False
                    for rt, rl in refined_left_lines:
                        if rt == line_type:
                            final_left_lines.append((line_type, rl))
                            found = True
                            break
                    if not found:
                        # 使用模板位置（偏移后）
                        offset_line = (
                            (template_line[0][0] + dx, template_line[0][1] + dy),
                            (template_line[1][0] + dx, template_line[1][1] + dy)
                        )
                        final_left_lines.append((line_type, offset_line))
            
            # 右侧同理
            detected_types = {dt for dt, _ in detected_right_lines}
            for line_type, template_line in template_lines_right:
                if line_type in detected_types:
                    for dt, dl in detected_right_lines:
                        if dt == line_type:
                            offset_line = (
                                (dl[0][0], dl[0][1]),
                                (dl[1][0], dl[1][1])
                            )
                            final_right_lines.append((line_type, offset_line))
                            break
                else:
                    found = False
                    for rt, rl in refined_right_lines:
                        if rt == line_type:
                            final_right_lines.append((line_type, rl))
                            found = True
                            break
                    if not found:
                        offset_line = (
                            (template_line[0][0] + dx, template_line[0][1] + dy),
                            (template_line[1][0] + dx, template_line[1][1] + dy)
                        )
                        final_right_lines.append((line_type, offset_line))
            
            print(f"[动态检测] 最终左侧线条: {len(final_left_lines)} 条")
            print(f"[动态检测] 最终右侧线条: {len(final_right_lines)} 条")
            
            # 更新检测结果
            self.image_canvas.detected_lines = []
            self.image_canvas.detected_rois = []
            
            # 显示变换后的检测区域
            display_left_roi = transformed_left_roi if transformed_left_roi else template_left_roi
            display_right_roi = transformed_right_roi if transformed_right_roi else template_right_roi
            
            if display_left_roi:
                self.image_canvas.detected_rois.append((display_left_roi, NeonStyle.NEON_GREEN))
            if display_right_roi:
                self.image_canvas.detected_rois.append((display_right_roi, NeonStyle.NEON_ORANGE))
            
            # 显示最终线条
            for _, line in final_left_lines[:2]:
                self.image_canvas.detected_lines.append((line, NeonStyle.NEON_GREEN))
            for _, line in final_right_lines[:2]:
                self.image_canvas.detected_lines.append((line, NeonStyle.NEON_ORANGE))
            
            print(f"[动态检测] 显示检测区域: 左={display_left_roi}, 右={display_right_roi}")
            print(f"[动态检测] 显示线条数: {len(self.image_canvas.detected_lines)} 条")
            
            # 计算中心位置和角度
            current_lines_left = [line for _, line in final_left_lines]
            current_lines_right = [line for _, line in final_right_lines]
            
            if len(current_lines_left) >= 1 and len(current_lines_right) >= 1:
                left_center = VisionAlgorithm.calculate_line_center(current_lines_left[0])
                right_center = VisionAlgorithm.calculate_line_center(current_lines_right[0])
                cx = (left_center[0] + right_center[0]) / 2
                cy = (left_center[1] + right_center[1]) / 2
                
                self.image_canvas.detected_center = (cx, cy)
                
                left_angle = VisionAlgorithm.calculate_line_angle(current_lines_left[0])
                right_angle = VisionAlgorithm.calculate_line_angle(current_lines_right[0])
                avg_angle = (left_angle + right_angle) / 2
                self.image_canvas.detected_angle = avg_angle
                
                print(f"[动态检测] 检测中心: ({cx:.1f}, {cy:.1f}), 平均角度: {avg_angle:.1f}°")
                
                # 计算并显示偏移（同步到机械臂参数）
                self._calculate_and_display_offsets_with_base(cx, cy, avg_angle)
                
                # 更新实时检测数据到UI
                self._sync_detection_data_to_ui(current_lines_left, current_lines_right, dx, dy, avg_angle)
                
                self.data_status.setText(f"检测中... 左:{len(current_lines_left)}条 右:{len(current_lines_right)}条")
            else:
                self.data_status.setText(f"未检测到直线 左:{len(detected_left_lines)}条 右:{len(detected_right_lines)}条")
                print(f"[动态检测] 警告: 检测直线数量不足")
            
            self.image_canvas.update()
            print("[动态检测] 完成")
            print("-"*40)
        except Exception as e:
            print(f"[动态检测] 错误: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _update_realtime_detection_data(self, left_lines, right_lines, dx, dy):
        """更新实时检测数据参数（兼容旧接口）"""
        self._sync_detection_data_to_ui(left_lines, right_lines, dx, dy, 0)
    
    def _sync_detection_data_to_ui(self, left_lines, right_lines, dx, dy, angle):
        """同步检测数据到UI显示（机械臂偏移参数和实时检测数据）"""
        try:
            # 计算左侧线条角度和长度
            if left_lines:
                left_angle = VisionAlgorithm.calculate_line_angle(left_lines[0])
                left_length = math.sqrt((left_lines[0][1][0]-left_lines[0][0][0])**2 + 
                                      (left_lines[0][1][1]-left_lines[0][0][1])**2)
            else:
                left_angle = 0
                left_length = 0
            
            # 计算右侧线条角度和长度
            if right_lines:
                right_angle = VisionAlgorithm.calculate_line_angle(right_lines[0])
                right_length = math.sqrt((right_lines[0][1][0]-right_lines[0][0][0])**2 + 
                                       (right_lines[0][1][1]-right_lines[0][0][1])**2)
            else:
                right_angle = 0
                right_length = 0
            
            # 计算中心点
            if left_lines and right_lines:
                left_center = VisionAlgorithm.calculate_line_center(left_lines[0])
                right_center = VisionAlgorithm.calculate_line_center(right_lines[0])
                center_x = (left_center[0] + right_center[0]) / 2
                center_y = (left_center[1] + right_center[1]) / 2
            elif left_lines:
                center_x, center_y = VisionAlgorithm.calculate_line_center(left_lines[0])
            elif right_lines:
                center_x, center_y = VisionAlgorithm.calculate_line_center(right_lines[0])
            else:
                center_x = 0
                center_y = 0
            
            # 同步到机械臂偏移参数（如果已标定）
            if self.calibration_data.is_calibrated and hasattr(self.calibration_data, 'pixel_to_mm') and self.calibration_data.pixel_to_mm > 0:
                dx_mm = dx * self.calibration_data.pixel_to_mm
                dy_mm = dy * self.calibration_data.pixel_to_mm
                
                # 更新UI显示 - 机械臂偏移参数
                if hasattr(self, 'offset_x_value'):
                    self.offset_x_value.setText(f"{dx_mm:.2f}")
                if hasattr(self, 'offset_y_value'):
                    self.offset_y_value.setText(f"{dy_mm:.2f}")
                if hasattr(self, 'offset_a_value'):
                    self.offset_a_value.setText(f"{angle:.2f}")
                
                print(f"[实时数据] 机械臂偏移 - X: {dx_mm:.2f}mm, Y: {dy_mm:.2f}mm, A: {angle:.2f}°")
            else:
                # 如果未标定，显示像素单位
                if hasattr(self, 'offset_x_value'):
                    self.offset_x_value.setText(f"{dx:.1f}")
                if hasattr(self, 'offset_y_value'):
                    self.offset_y_value.setText(f"{dy:.1f}")
                if hasattr(self, 'offset_a_value'):
                    self.offset_a_value.setText(f"{angle:.2f}")
                
                print(f"[实时数据] 偏移(像素) - X: {dx:.1f}px, Y: {dy:.1f}px, A: {angle:.2f}°")
            
            # 更新实时检测数据面板
            if hasattr(self, 'left_angle_value'):
                self.left_angle_value.setText(f"{left_angle:.1f}")
            if hasattr(self, 'left_length_value'):
                self.left_length_value.setText(f"{left_length:.1f}")
            if hasattr(self, 'right_angle_value'):
                self.right_angle_value.setText(f"{right_angle:.1f}")
            if hasattr(self, 'right_length_value'):
                self.right_length_value.setText(f"{right_length:.1f}")
            if hasattr(self, 'center_x_value'):
                self.center_x_value.setText(f"{center_x:.1f}")
            if hasattr(self, 'center_y_value'):
                self.center_y_value.setText(f"{center_y:.1f}")
            
            # 记录实时检测参数
            self.last_detection_data = {
                'left_angle': left_angle,
                'left_length': left_length,
                'right_angle': right_angle,
                'right_length': right_length,
                'center_x': center_x,
                'center_y': center_y,
                'dx': dx,
                'dy': dy,
                'angle': angle,
                'timestamp': time.time()
            }
            
            print(f"[实时数据] 左角度: {left_angle:.1f}°, 左长度: {left_length:.1f}px")
            print(f"[实时数据] 右角度: {right_angle:.1f}°, 右长度: {right_length:.1f}px")
            print(f"[实时数据] 中心点: ({center_x:.1f}, {center_y:.1f})")
            
        except Exception as e:
            print(f"[实时数据同步] 错误: {e}")
            import traceback
            traceback.print_exc()

    def _calculate_and_display_offsets_with_base(self, current_x, current_y, current_angle):
        """使用测试时保存的基准值计算偏移并显示"""
        try:
            if not hasattr(self, 'test_base_x') or not hasattr(self, 'test_base_y') or not hasattr(self, 'test_base_angle'):
                print("[偏移计算] 错误: 基准值未设置")
                return
            
            # 计算像素偏移
            dx_pixel = current_x - self.test_base_x
            dy_pixel = current_y - self.test_base_y
            d_angle = current_angle - self.test_base_angle
            
            # 转换角度到 -180 到 180 范围
            while d_angle > 180:
                d_angle -= 360
            while d_angle < -180:
                d_angle += 360
            
            # 如果已标定，转换为物理坐标
            if self.calibration_data.is_calibrated and self.calibration_data.pixel_to_mm > 0:
                dx_mm = dx_pixel * self.calibration_data.pixel_to_mm
                dy_mm = dy_pixel * self.calibration_data.pixel_to_mm
                
                # 更新显示
                self.offset_x_label.setText(f"X偏移: {dx_mm:.2f} mm")
                self.offset_y_label.setText(f"Y偏移: {dy_mm:.2f} mm")
                self.offset_angle_label.setText(f"角度: {d_angle:.1f} °")
                
                # 设置颜色：如果在容差内为绿色，否则为红色
                alarm_x = self.alarm_x_spin.value()
                alarm_y = self.alarm_y_spin.value()
                alarm_a = self.alarm_a_spin.value()
                
                if abs(dx_mm) <= alarm_x and abs(dy_mm) <= alarm_y and abs(d_angle) <= alarm_a:
                    self.offset_x_label.setStyleSheet("color: #00FF96; font-weight: bold;")
                    self.offset_y_label.setStyleSheet("color: #00FF96; font-weight: bold;")
                    self.offset_angle_label.setStyleSheet("color: #00FF96; font-weight: bold;")
                    self.status_label.setText("状态: 正常 ✓")
                    self.status_label.setStyleSheet("color: #00FF96; font-weight: bold;")
                else:
                    self.offset_x_label.setStyleSheet("color: #FF6666; font-weight: bold;")
                    self.offset_y_label.setStyleSheet("color: #FF6666; font-weight: bold;")
                    self.offset_angle_label.setStyleSheet("color: #FF6666; font-weight: bold;")
                    self.status_label.setText("状态: 偏移报警 ⚠")
                    self.status_label.setStyleSheet("color: #FF6666; font-weight: bold;")
                
                print(f"[偏移计算] 基准: ({self.test_base_x:.1f}, {self.test_base_y:.1f}, {self.test_base_angle:.1f}°)")
                print(f"[偏移计算] 当前: ({current_x:.1f}, {current_y:.1f}, {current_angle:.1f}°)")
                print(f"[偏移计算] 像素偏移: dx={dx_pixel:.2f}px, dy={dy_pixel:.2f}px, da={d_angle:.2f}°")
                print(f"[偏移计算] 物理偏移: dx={dx_mm:.2f}mm, dy={dy_mm:.2f}mm")
                print(f"[偏移计算] 标定状态: 已标定 (像素/mm: {self.calibration_data.pixel_to_mm:.4f})")
            else:
                # 未标定，只显示像素坐标
                self.offset_x_label.setText(f"X偏移: {dx_pixel:.1f} px")
                self.offset_y_label.setText(f"Y偏移: {dy_pixel:.1f} px")
                self.offset_angle_label.setText(f"角度: {d_angle:.1f} °")
                
                print(f"[偏移计算] 基准: ({self.test_base_x:.1f}, {self.test_base_y:.1f}, {self.test_base_angle:.1f}°)")
                print(f"[偏移计算] 当前: ({current_x:.1f}, {current_y:.1f}, {current_angle:.1f}°)")
                print(f"[偏移计算] 像素偏移: dx={dx_pixel:.2f}px, dy={dy_pixel:.2f}px, da={d_angle:.2f}°")
                print(f"[偏移计算] 标定状态: 未标定")
        
        except Exception as e:
            print(f"[偏移计算] 错误: {e}")
            import traceback
            traceback.print_exc()

    def _exit_test_mode(self):
        """退出测试模式"""
        # 停止动态检测
        self.detection_timer.stop()
        
        # 清除测试基准值
        if hasattr(self, 'test_base_x'):
            del self.test_base_x
        if hasattr(self, 'test_base_y'):
            del self.test_base_y
        if hasattr(self, 'test_base_angle'):
            del self.test_base_angle
        
        self.image_canvas.exit_test_mode()
        self.image_canvas.update()

        self.btn_test_line.setText("🔍 手动测试")
        self.btn_test_line.clicked.disconnect()
        self.btn_test_line.clicked.connect(self._test_line_detection)

        self.data_status.setText("已退出测试")
        self.data_status.setStyleSheet(self._get_label_style(12, "#FF9600", True))

    def _calculate_and_display_offsets(self, cx, cy, angle):
        """计算并显示偏移量（支持双直线模板和外形模板）"""
        # 根据当前模式选择模板数据
        if self.current_mode == "line":
            # 双直线模板
            if not self.dual_line_template.left_roi or not self.dual_line_template.left_line_long:
                return

            # 计算像素偏移
            pixel_offset_x = cx - self.dual_line_template.left_roi[0]
            pixel_offset_y = cy - self.dual_line_template.left_roi[1]

            # 计算角度偏移
            base_left_angle = VisionAlgorithm.calculate_line_angle(self.dual_line_template.left_line_long)
            angle_offset = angle - base_left_angle

        elif self.current_mode == "shape":
            # 外形模板
            if not self.shape_template.left_roi:
                return

            # 计算像素偏移（使用左ROI作为基准）
            pixel_offset_x = cx - self.shape_template.left_roi[0]
            pixel_offset_y = cy - self.shape_template.left_roi[1]

            # 计算角度偏移（外形模板使用检测到的角度）
            angle_offset = angle

        else:
            return

        # 转换为物理单位（如果有标定）
        if self.calibration_data.is_calibrated:
            offset_x_mm = pixel_offset_x / self.calibration_data.pixel_per_mm_x
            offset_y_mm = pixel_offset_y / self.calibration_data.pixel_per_mm_y
        else:
            offset_x_mm = pixel_offset_x
            offset_y_mm = pixel_offset_y

        # 应用用户设置的偏移补偿
        final_offset_x = offset_x_mm + self.offset_x_spin.value()
        final_offset_y = offset_y_mm + self.offset_y_spin.value()
        final_offset_z = self.offset_z_spin.value()
        final_offset_a = angle_offset + self.offset_a_spin.value()

        # 更新数据显示
        self._update_result_display(cx, cy, 0, angle, 0.95)

        # 更新机械臂偏移参数显示
        self.offset_x_value.setText(f"{final_offset_x:.4f} mm")
        self.offset_y_value.setText(f"{final_offset_y:.4f} mm")
        self.offset_z_value.setText(f"{final_offset_z:.4f} mm")
        self.offset_a_value.setText(f"{final_offset_a:.4f} °")

    # ==================== 异形模板设置 ====================

    def _start_draw_contour(self, side):
        """开始绘制轮廓"""
        self.contour_drawing_side = side
        self.image_canvas.start_drawing("contour")
        self.status_label.setText(f"框选轮廓: {side}")

    def _start_draw_contour_roi(self, side):
        """开始绘制轮廓ROI"""
        self.contour_roi_drawing_side = side
        self.image_canvas.start_drawing("roi")
        self.status_label.setText(f"绘制轮廓ROI: {side}")

    def _auto_detect_left_contour(self):
        """自动识别左侧轮廓"""
        if self.image_canvas.current_frame is None:
            QMessageBox.warning(self, "警告", "请先打开摄像头")
            return

        if self.image_canvas.left_contour_roi is None:
            QMessageBox.warning(self, "警告", "请先设置左侧监测区域，再进行自动识别")
            return

        frame = self.image_canvas.current_frame
        contours = VisionAlgorithm.detect_contours_multialgorithm(
            frame, self.image_canvas.left_contour_roi
        )

        if contours:
            best_contour = contours[0][0]
            self.image_canvas.save_state("left_contour", self.image_canvas.left_contour, "自动识别左轮廓")
            self.image_canvas.left_contour = best_contour
            self.image_canvas.update()
            self._update_shape_ui()
            self._update_shape_template_status()
            self.status_label.setText("左侧轮廓自动识别成功")
        else:
            QMessageBox.warning(self, "警告", "未检测到有效轮廓，请尝试调整参数或手动绘制")

    def _auto_detect_right_contour(self):
        """自动识别右侧轮廓"""
        if self.image_canvas.current_frame is None:
            QMessageBox.warning(self, "警告", "请先打开摄像头")
            return

        if self.image_canvas.right_contour_roi is None:
            QMessageBox.warning(self, "警告", "请先设置右侧监测区域，再进行自动识别")
            return

        frame = self.image_canvas.current_frame
        contours = VisionAlgorithm.detect_contours_multialgorithm(
            frame, self.image_canvas.right_contour_roi
        )

        if contours:
            best_contour = contours[0][0]
            self.image_canvas.save_state("right_contour", self.image_canvas.right_contour, "自动识别右轮廓")
            self.image_canvas.right_contour = best_contour
            self.image_canvas.update()
            self._update_shape_ui()
            self._update_shape_template_status()
            self.status_label.setText("右侧轮廓自动识别成功")
        else:
            QMessageBox.warning(self, "警告", "未检测到有效轮廓，请尝试调整参数或手动绘制")

    def _auto_detect_left_contour_roi(self):
        """自动识别左侧ROI区域"""
        if self.image_canvas.current_frame is None:
            QMessageBox.warning(self, "警告", "请先打开摄像头")
            return

        if self.image_canvas.left_contour is None:
            QMessageBox.warning(self, "警告", "请先设置或识别左侧轮廓，再设置ROI区域")
            return

        x, y, w, h = cv2.boundingRect(self.image_canvas.left_contour)
        padding = 20
        roi = (
            max(0, x - padding),
            max(0, y - padding),
            min(self.image_canvas.current_frame.shape[1], x + w + padding),
            min(self.image_canvas.current_frame.shape[0], y + h + padding)
        )

        self.image_canvas.save_state("left_contour_roi", self.image_canvas.left_contour_roi, "自动识别左ROI")
        self.image_canvas.left_contour_roi = roi
        self.image_canvas.update()
        self._update_shape_ui()
        self._update_shape_template_status()
        self.status_label.setText("左侧ROI区域自动识别成功")

    def _auto_detect_right_contour_roi(self):
        """自动识别右侧ROI区域"""
        if self.image_canvas.current_frame is None:
            QMessageBox.warning(self, "警告", "请先打开摄像头")
            return

        if self.image_canvas.right_contour is None:
            QMessageBox.warning(self, "警告", "请先设置或识别右侧轮廓，再设置ROI区域")
            return

        x, y, w, h = cv2.boundingRect(self.image_canvas.right_contour)
        padding = 20
        roi = (
            max(0, x - padding),
            max(0, y - padding),
            min(self.image_canvas.current_frame.shape[1], x + w + padding),
            min(self.image_canvas.current_frame.shape[0], y + h + padding)
        )

        self.image_canvas.save_state("right_contour_roi", self.image_canvas.right_contour_roi, "自动识别右ROI")
        self.image_canvas.right_contour_roi = roi
        self.image_canvas.update()
        self._update_shape_ui()
        self._update_shape_template_status()
        self.status_label.setText("右侧ROI区域自动识别成功")

    def _undo_shape_operation(self):
        """撤销上一步异形操作"""
        result = self.image_canvas.undo()
        if result:
            self.status_label.setText(f"已撤销: {result}")
            self._update_shape_ui()
            self._update_shape_template_status()
        else:
            self.status_label.setText("没有可撤销的操作")

    def _update_shape_ui(self):
        """更新异形UI显示"""
        if self.image_canvas.left_contour is not None:
            area = cv2.contourArea(self.image_canvas.left_contour)
            self.left_contour_info.setText(f"轮廓: 已设置 (面积={area:.0f})")
        else:
            self.left_contour_info.setText("轮廓: 未设置")

        if self.image_canvas.right_contour is not None:
            area = cv2.contourArea(self.image_canvas.right_contour)
            self.right_contour_info.setText(f"轮廓: 已设置 (面积={area:.0f})")
        else:
            self.right_contour_info.setText("轮廓: 未设置")

        if self.image_canvas.left_contour_roi:
            rect = self.image_canvas.left_contour_roi
            self.left_contour_roi_info.setText(f"区域: ({rect[0]:.0f},{rect[1]:.0f})-({rect[2]:.0f},{rect[3]:.0f})")
        else:
            self.left_contour_roi_info.setText("区域: 未设置")

        if self.image_canvas.right_contour_roi:
            rect = self.image_canvas.right_contour_roi
            self.right_contour_roi_info.setText(f"区域: ({rect[0]:.0f},{rect[1]:.0f})-({rect[2]:.0f},{rect[3]:.0f})")
        else:
            self.right_contour_roi_info.setText("区域: 未设置")

    def _on_contour_selected(self, rect):
        """轮廓选择完成回调"""
        if self.image_canvas.current_frame is None:
            return

        x1, y1, x2, y2 = rect
        frame = self.image_canvas.current_frame

        # 检测轮廓
        contours = VisionAlgorithm.detect_contours_in_roi(
            frame, rect,
            canny_low=self.canny_low_slider.value(),
            canny_high=self.canny_high_slider.value()
        )

        if contours:
            largest = max(contours, key=cv2.contourArea)

            if self.contour_drawing_side == "left":
                self.image_canvas.save_state("left_contour", self.image_canvas.left_contour, "框选左轮廓")
                self.image_canvas.left_contour = largest
                area = cv2.contourArea(largest)
                self.left_contour_info.setText(f"轮廓: 已设置 (面积={area:.0f})")
            elif self.contour_drawing_side == "right":
                self.image_canvas.save_state("right_contour", self.image_canvas.right_contour, "框选右轮廓")
                self.image_canvas.right_contour = largest
                area = cv2.contourArea(largest)
                self.right_contour_info.setText(f"轮廓: 已设置 (面积={area:.0f})")

        self.contour_drawing_side = None
        self._update_shape_template_status()
        self.image_canvas.update()

        # 处理轮廓ROI绘制
        if self.contour_roi_drawing_side:
            if self.contour_roi_drawing_side == "left":
                self.image_canvas.save_state("left_contour_roi", self.image_canvas.left_contour_roi, "设置左轮廓ROI")
                self.image_canvas.left_contour_roi = rect
                self.left_contour_roi_info.setText(f"区域: ({rect[0]:.0f},{rect[1]:.0f})-({rect[2]:.0f},{rect[3]:.0f})")
            elif self.contour_roi_drawing_side == "right":
                self.image_canvas.save_state("right_contour_roi", self.image_canvas.right_contour_roi, "设置右轮廓ROI")
                self.image_canvas.right_contour_roi = rect
                self.right_contour_roi_info.setText(f"区域: ({rect[0]:.0f},{rect[1]:.0f})-({rect[2]:.0f},{rect[3]:.0f})")
            self.contour_roi_drawing_side = None
            self._update_shape_template_status()
            self.image_canvas.update()

    def _on_contour_drawn(self, contour):
        """手绘轮廓完成回调"""
        if self.contour_drawing_side == "left":
            self.image_canvas.save_state("left_contour", self.image_canvas.left_contour, "手绘左轮廓")
            self.image_canvas.left_contour = contour
            area = cv2.contourArea(contour)
            self.left_contour_info.setText(f"轮廓: 手绘 (面积={area:.0f})")
        elif self.contour_drawing_side == "right":
            self.image_canvas.save_state("right_contour", self.image_canvas.right_contour, "手绘右轮廓")
            self.image_canvas.right_contour = contour
            area = cv2.contourArea(contour)
            self.right_contour_info.setText(f"轮廓: 手绘 (面积={area:.0f})")

        self.contour_drawing_side = None
        self._update_shape_template_status()
        self.image_canvas.update()

    def _update_shape_template_status(self):
        """更新异形模板状态"""
        has_all_contours = (
            self.image_canvas.left_contour is not None and
            self.image_canvas.right_contour is not None
        )

        has_all_rois = (
            self.image_canvas.left_contour_roi is not None and
            self.image_canvas.right_contour_roi is not None
        )

        if has_all_contours and has_all_rois:
            self.shape_template_status.setText("模板状态: 已完成 ✓")
            self.shape_template_status.setStyleSheet(self._get_label_style(12, "#00FF96", True))
            self.btn_save_shape_template.setEnabled(True)
        elif has_all_contours:
            self.shape_template_status.setText("模板状态: 需设置区域")
            self.shape_template_status.setStyleSheet(self._get_label_style(12, "#FF9600", True))
            self.btn_save_shape_template.setEnabled(False)
        else:
            self.shape_template_status.setText("模板状态: 未完成")
            self.shape_template_status.setStyleSheet(self._get_label_style(12, "#FF6666", True))
            self.btn_save_shape_template.setEnabled(False)

        self.btn_test_shape.setEnabled(self.shape_template.valid)

    def _save_shape_template(self):
        """保存异形模板（支持选择路径和重命名）"""
        if self.image_canvas.left_contour is None:
            QMessageBox.warning(self, "警告", "请先设置左侧轮廓")
            return

        if self.image_canvas.right_contour is None:
            QMessageBox.warning(self, "警告", "请先设置右侧轮廓")
            return

        if self.image_canvas.left_contour_roi is None or self.image_canvas.right_contour_roi is None:
            QMessageBox.warning(self, "警告", "请先设置左右监测区域")
            return

        self.shape_template.left_contour = self.image_canvas.left_contour
        self.shape_template.right_contour = self.image_canvas.right_contour
        self.shape_template.left_roi = self.image_canvas.left_contour_roi
        self.shape_template.right_roi = self.image_canvas.right_contour_roi
        self.shape_template.valid = True

        # 保存参数
        params = {
            'canny_low': self.canny_low_slider.value(),
            'canny_high': self.canny_high_slider.value(),
            'match_threshold': self.match_threshold_spin.value()
        }

        # 弹出保存对话框
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存异形模板",
            DEFAULT_SHAPE_TEMPLATE_FILE,
            "Pickle文件 (*.pkl);;所有文件 (*.*)"
        )

        if file_path:
            # 确保文件扩展名正确
            if not file_path.lower().endswith('.pkl'):
                file_path += '.pkl'

            # 保存到用户选择的路径
            TemplateManager.save_shape_template(file_path, self.shape_template, params)

            # 同时保存到默认位置（用于下次启动加载）
            TemplateManager.save_shape_template(DEFAULT_SHAPE_TEMPLATE_FILE, self.shape_template, params)

            self.btn_test_shape.setEnabled(True)

            QMessageBox.information(
                self, "保存成功",
                f"异形模板已保存！\n\n"
                f"保存位置: {file_path}\n\n"
                f"同时已备份到默认位置供下次启动使用"
            )

    def _load_shape_template_to_canvas(self):
        """加载异形模板到画布"""
        self.image_canvas.left_contour = self.shape_template.left_contour
        self.image_canvas.right_contour = self.shape_template.right_contour
        self.image_canvas.left_contour_roi = self.shape_template.left_roi
        self.image_canvas.right_contour_roi = self.shape_template.right_roi
        self._update_shape_ui()

    def _load_shape_template(self):
        """加载异形模板"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "加载异形模板", "", "Pickle (*.pkl)"
        )
        if filename:
            try:
                self.shape_template, params = TemplateManager.load_shape_template(filename)
                self._load_shape_template_to_canvas()
                if params:
                    self.canny_low_slider.setValue(params.get('canny_low', 50))
                    self.canny_high_slider.setValue(params.get('canny_high', 150))
                    self.match_threshold_spin.setValue(params.get('match_threshold', 0.7))
                self._update_shape_template_status()
                self.image_canvas.update()
                QMessageBox.information(self, "成功", f"异形模板已加载:\n{filename}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载失败: {str(e)}")

    def _clear_all_shape_template(self):
        """清除所有异形模板"""
        self.image_canvas.clear_all()
        self.shape_template = ShapeTemplate()
        self.left_contour_info.setText("轮廓: 未设置")
        self.right_contour_info.setText("轮廓: 未设置")
        self.left_contour_roi_info.setText("区域: 未设置")
        self.right_contour_roi_info.setText("区域: 未设置")
        self._update_shape_template_status()

    def _test_shape_detection(self):
        """测试异形检测（使用多算法融合提高精度）"""
        if not self.shape_template.valid:
            QMessageBox.warning(self, "警告", "请先保存模板")
            return

        if self.image_canvas.current_frame is None:
            QMessageBox.warning(self, "警告", "请先打开摄像头")
            return

        self.current_mode = "shape"
        # 进入测试模式
        self.image_canvas.enter_test_mode()

        # 启动动态检测
        self.detection_timer.start()

        # 更新按钮状态
        self.btn_test_shape.setText("⏹ 停止检测")
        self.btn_test_shape.clicked.disconnect()
        self.btn_test_shape.clicked.connect(self._exit_shape_test_mode)

    def _perform_shape_dynamic_detection(self):
        """动态实时检测异形 - 根据目标位置实时更新数据，区域随纸张变换"""
        try:
            # 检查当前帧是否有效
            if self.image_canvas.current_frame is None or self.image_canvas.current_frame.size == 0:
                print("[外形动态检测] 错误: 没有当前帧或帧为空")
                return

            frame = self.image_canvas.current_frame
            h, w = frame.shape[:2]

            print("\n" + "-"*40)
            print("[外形动态检测] 开始检测")
            print(f"[外形动态检测] 帧大小: {frame.shape}")

            # 获取模板数据
            template_left_contour = self.shape_template.left_contour
            template_right_contour = self.shape_template.right_contour
            template_left_roi = self.shape_template.left_roi
            template_right_roi = self.shape_template.right_roi

            # 检查是否有完整的设置
            if template_left_contour is None or template_right_contour is None:
                self.status_label.setText("模板轮廓未设置，无法检测")
                print("[外形动态检测] 错误: 模板轮廓未设置")
                return

            if template_left_roi is None or template_right_roi is None:
                self.status_label.setText("模板区域未设置，无法检测")
                print("[外形动态检测] 错误: 模板区域未设置")
                return

            print(f"[外形动态检测] 左ROI: {template_left_roi}")
            print(f"[外形动态检测] 右ROI: {template_right_roi}")

            # 获取当前选择的算法（与双直线模板一致）
            selected_algorithm = self._get_selected_algorithm()
            print(f"[外形动态检测] 使用算法: {selected_algorithm}")

            # 使用多算法检测轮廓
            left_detected = VisionAlgorithm.detect_contours_multialgorithm(
                frame, template_left_roi,
                template_contour=template_left_contour
            )

            right_detected = VisionAlgorithm.detect_contours_multialgorithm(
                frame, template_right_roi,
                template_contour=template_right_contour
            )

            # 提取轮廓列表
            left_contour_list = [c[0] for c in left_detected]
            right_contour_list = [c[0] for c in right_detected]

            print(f"[外形动态检测] 左轮廓候选数: {len(left_contour_list)}, 右轮廓候选数: {len(right_contour_list)}")

            # 使用改进的匹配算法（考虑位置和尺寸）
            best_left_match, left_score, left_offset = VisionAlgorithm.match_contours_by_position_size(
                template_left_contour, left_contour_list
            )

            best_right_match, right_score, right_offset = VisionAlgorithm.match_contours_by_position_size(
                template_right_contour, right_contour_list
            )

            print(f"[外形动态检测] 左匹配分数: {left_score:.3f}, 右匹配分数: {right_score:.3f}")
            print(f"[外形动态检测] 左偏移: {left_offset}, 右偏移: {right_offset}")

            # 更新检测结果
            self.image_canvas.detected_contours = []
            self.image_canvas.detected_rois = []

            # 设置匹配阈值（可调）
            match_threshold = 0.1  # 降低阈值提高检测灵敏度

            # 计算动态ROI（随检测到的轮廓位置变换）
            transformed_left_roi = None
            transformed_right_roi = None

            if best_left_match is not None and left_score > match_threshold:
                self.image_canvas.detected_contours.append((best_left_match, NeonStyle.NEON_GREEN))
                # 根据检测到的轮廓动态调整ROI区域
                x, y, w_contour, h_contour = cv2.boundingRect(best_left_match)
                padding = 20
                transformed_left_roi = (
                    max(0, x - padding),
                    max(0, y - padding),
                    min(w, x + w_contour + padding),
                    min(h, y + h_contour + padding)
                )
                self.image_canvas.detected_rois.append((transformed_left_roi, NeonStyle.NEON_GREEN))
                print(f"[外形动态检测] 左ROI变换: {transformed_left_roi}")

            if best_right_match is not None and right_score > match_threshold:
                self.image_canvas.detected_contours.append((best_right_match, NeonStyle.NEON_ORANGE))
                # 根据检测到的轮廓动态调整ROI区域
                x, y, w_contour, h_contour = cv2.boundingRect(best_right_match)
                padding = 20
                transformed_right_roi = (
                    max(0, x - padding),
                    max(0, y - padding),
                    min(w, x + w_contour + padding),
                    min(h, y + h_contour + padding)
                )
                self.image_canvas.detected_rois.append((transformed_right_roi, NeonStyle.NEON_ORANGE))
                print(f"[外形动态检测] 右ROI变换: {transformed_right_roi}")

            # 计算中心位置和角度
            if best_left_match is not None and best_right_match is not None:
                left_center = VisionAlgorithm._get_contour_center(best_left_match)
                right_center = VisionAlgorithm._get_contour_center(best_right_match)

                cx = (left_center[0] + right_center[0]) / 2
                cy = (left_center[1] + right_center[1]) / 2
                self.image_canvas.detected_center = (cx, cy)

                angle = math.degrees(math.atan2(right_center[1] - left_center[1],
                                              right_center[0] - left_center[0]))
                self.image_canvas.detected_angle = angle

                print(f"[外形动态检测] 中心位置: ({cx:.2f}, {cy:.2f}), 角度: {angle:.2f}°")

                # 计算并显示偏移
                self._calculate_and_display_offsets(cx, cy, angle)

                # 计算并显示偏差
                self._display_contour_deviation(best_left_match, best_right_match)

            self.image_canvas.update()
        except Exception as e:
            print(f"异形检测错误: {str(e)}")
            import traceback
            traceback.print_exc()
            # 不停止定时器，继续运行

    def _display_contour_deviation(self, left_contour, right_contour):
        """显示轮廓偏差信息"""
        if left_contour is None or right_contour is None:
            return

        # 计算左侧偏差
        left_dev = VisionAlgorithm.calculate_contour_deviation(
            self.shape_template.left_contour, left_contour
        )

        # 计算右侧偏差
        right_dev = VisionAlgorithm.calculate_contour_deviation(
            self.shape_template.right_contour, right_contour
        )

        # 在画面上显示偏差信息（通过状态标签）
        avg_offset_x = (left_dev['offset_x'] + right_dev['offset_x']) / 2
        avg_offset_y = (left_dev['offset_y'] + right_dev['offset_y']) / 2
        avg_angle_diff = (left_dev['angle_diff'] + right_dev['angle_diff']) / 2
        avg_scale_diff = (left_dev['scale_diff'] + right_dev['scale_diff']) / 2

        self.data_x_mm.setText(f"X偏: {avg_offset_x:.1f}px")
        self.data_y_mm.setText(f"Y偏: {avg_offset_y:.1f}px")
        self.data_z.setText(f"角度差: {avg_angle_diff:.2f}°")
        self.data_angle.setText(f"尺度差: {avg_scale_diff:.1f}%")

    def _exit_shape_test_mode(self):
        """退出异形测试模式"""
        # 停止动态检测
        self.detection_timer.stop()

        self.image_canvas.exit_test_mode()
        self.image_canvas.update()

        self.btn_test_shape.setText("🔍 手动测试")
        self.btn_test_shape.clicked.disconnect()
        self.btn_test_shape.clicked.connect(self._test_shape_detection)

        self.data_status.setText("已退出测试")
        self.data_status.setStyleSheet(self._get_label_style(12, "#FF9600", True))

    # ==================== 结果显示 ====================

    def _update_result_display(self, x_pixel, y_pixel, z_mm, angle, score):
        """更新结果显示"""
        self.data_x_pixel.setText(f"{x_pixel:.2f}")
        self.data_y_pixel.setText(f"{y_pixel:.2f}")

        if self.calibration_data.is_calibrated:
            x_mm = x_pixel / self.calibration_data.pixel_per_mm_x
            y_mm = y_pixel / self.calibration_data.pixel_per_mm_y
            self.data_x_mm.setText(f"{x_mm:.4f}")
            self.data_y_mm.setText(f"{y_mm:.4f}")
        else:
            self.data_x_mm.setText("--")
            self.data_y_mm.setText("--")

        self.data_z.setText(f"{z_mm:.2f}")
        self.data_angle.setText(f"{angle:.2f}°")
        self.data_score.setText(f"{score:.3f}")
        self.data_status.setText("识别成功")
        self.data_status.setStyleSheet(self._get_label_style(12, "#00FF96", True))

    # ==================== 贴料参数 ====================

    def _save_paste_params(self):
        """保存贴料参数"""
        params = {
            'offset_x': self.offset_x_spin.value(),
            'offset_y': self.offset_y_spin.value(),
            'offset_z': self.offset_z_spin.value(),
            'offset_a': self.offset_a_spin.value(),
            'score_threshold': self.score_threshold_spin.value(),
            'alarm_x': self.alarm_x_spin.value(),
            'alarm_y': self.alarm_y_spin.value(),
            'alarm_a': self.alarm_a_spin.value(),
            'timeout': self.timeout_spin.value(),
            'photo_delay': self.photo_delay_spin.value(),
            'vacuum_delay': self.vacuum_delay_spin.value(),
            'retry_count': self.retry_count_spin.value(),
            'fail_mode': self.fail_mode_combo.currentText()
        }

        filename, _ = QFileDialog.getSaveFileName(
            self, "保存贴料参数", "", "JSON (*.json)"
        )
        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(params, f, indent=2, ensure_ascii=False)
            QMessageBox.information(self, "成功", f"贴料参数已保存:\n{filename}")

    def _load_paste_params(self):
        """加载贴料参数"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "加载贴料参数", "", "JSON (*.json)"
        )
        if filename:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self._apply_paste_params(data)
            QMessageBox.information(self, "成功", "贴料参数已加载")

    # ==================== 交互测试功能 ====================

    def _setup_interactive_test_tab(self):
        """设置交互测试标签页"""
        layout = QVBoxLayout(self.test_tab)
        layout.setSpacing(15)
        
        # 测试步骤显示
        self.test_step_label = QLabel("点击「开始交互测试」按钮开始")
        self.test_step_label.setStyleSheet(self._get_label_style(size=14, color="#00FFFF", bold=True))
        self.test_step_label.setWordWrap(True)
        layout.addWidget(self.test_step_label)
        
        # 测试状态显示
        self.test_status_label = QLabel("准备开始测试...")
        self.test_status_label.setStyleSheet(self._get_label_style(size=12, color="#00FF96"))
        layout.addWidget(self.test_status_label)
        
        # 按钮区域
        btn_layout = QHBoxLayout()
        
        self.btn_start_test = QPushButton("▶ 开始交互测试")
        self.btn_start_test.setStyleSheet(self._get_button_style("#00AA66"))
        self.btn_start_test.clicked.connect(self._start_interactive_test)
        btn_layout.addWidget(self.btn_start_test)
        
        self.btn_next_step = QPushButton("下一步 →")
        self.btn_next_step.setStyleSheet(self._get_button_style("#0088FF"))
        self.btn_next_step.clicked.connect(self._next_test_step)
        self.btn_next_step.setEnabled(False)
        btn_layout.addWidget(self.btn_next_step)
        
        self.btn_feedback = QPushButton("❓ 反馈问题")
        self.btn_feedback.setStyleSheet(self._get_button_style("#FF6666"))
        self.btn_feedback.clicked.connect(self._show_feedback_dialog)
        self.btn_feedback.setEnabled(False)
        btn_layout.addWidget(self.btn_feedback)
        
        layout.addLayout(btn_layout)
        
        # 问题反馈输入
        feedback_group = QGroupBox("问题反馈")
        feedback_group.setStyleSheet(self._get_group_style())
        feedback_layout = QVBoxLayout(feedback_group)
        
        self.feedback_input = QTextEdit()
        self.feedback_input.setStyleSheet(self._get_text_edit_style())
        self.feedback_input.setPlaceholderText("如果这一步有问题，请在此描述问题...")
        self.feedback_input.setMaximumHeight(80)
        feedback_layout.addWidget(self.feedback_input)
        
        self.btn_submit_feedback = QPushButton("提交问题")
        self.btn_submit_feedback.setStyleSheet(self._get_button_style("#FF9600"))
        self.btn_submit_feedback.clicked.connect(self._submit_feedback)
        self.btn_submit_feedback.setEnabled(False)
        feedback_layout.addWidget(self.btn_submit_feedback)
        
        layout.addWidget(feedback_group)
        
        # 测试结果显示
        self.test_result_text = QTextEdit()
        self.test_result_text.setStyleSheet(self._get_text_edit_style())
        self.test_result_text.setReadOnly(True)
        self.test_result_text.setPlaceholderText("测试日志将在这里显示...")
        layout.addWidget(self.test_result_text)
        
        # 初始化测试状态
        self.test_step_index = 0
        self.test_feedback_list = []
        self.is_testing = False
        
        # 定义测试步骤
        self.test_steps = [
            {
                "name": "摄像头检查",
                "description": "步骤1/10：检查摄像头是否正常工作\n\n请确认：\n• 画面是否正常显示？\n• 画面是否清晰？\n• 是否有明显的延迟？",
                "check": self._check_camera
            },
            {
                "name": "模板加载检查",
                "description": "步骤2/10：检查模板是否已加载\n\n请确认：\n• 系统是否加载了上次使用的模板？\n• 如果需要，可以切换到「双直线模板」或「外形模板」标签页加载模板",
                "check": self._check_template_loaded
            },
            {
                "name": "模板设置测试",
                "description": "步骤3/10：测试模板设置功能\n\n请操作：\n1. 切换到「双直线模板」标签页\n2. 尝试设置左侧和右侧的直线与区域\n3. 点击「撤销上一步」测试撤销功能",
                "check": self._check_template_setup
            },
            {
                "name": "算法选择测试",
                "description": "步骤4/10：测试算法选择功能\n\n请操作：\n1. 在「双直线模板」或「外形模板」标签页中\n2. 切换不同的检测算法\n3. 观察识别结果是否实时更新",
                "check": self._check_algorithm_selection
            },
            {
                "name": "标定功能检查",
                "description": "步骤5/10：检查摄像机标定功能\n\n请确认：\n• 切换到「摄像机标定」标签页\n• 检查是否已进行标定\n• 如果未标定，可以尝试进行九点标定",
                "check": self._check_calibration
            },
            {
                "name": "手动测试启动",
                "description": "步骤6/10：启动手动测试\n\n请操作：\n1. 确保已加载或设置好模板\n2. 点击「手动测试」按钮\n3. 观察是否进入测试模式",
                "check": self._check_manual_test_start
            },
            {
                "name": "动态检测测试",
                "description": "步骤7/10：测试动态检测功能\n\n请观察：\n• 移动目标纸张\n• 观察识别线框是否跟随移动\n• 检查偏移值是否实时更新",
                "check": self._check_dynamic_detection
            },
            {
                "name": "识别精度检查",
                "description": "步骤8/10：检查识别精度\n\n请确认：\n• 将目标纸张放置在与模板相同位置\n• 观察识别线框是否与模板对齐\n• 检查偏移值是否接近零",
                "check": self._check_detection_accuracy
            },
            {
                "name": "异形模板测试",
                "description": "步骤9/10：测试异形模板功能（可选）\n\n请操作：\n1. 切换到「外形模板」标签页\n2. 测试轮廓识别和手动绘制功能\n3. 测试异形模板的检测功能",
                "check": self._check_shape_template
            },
            {
                "name": "测试完成",
                "description": "步骤10/10：测试完成！\n\n感谢您完成测试。\n如果有任何问题，请使用「反馈问题」按钮提交。",
                "check": self._check_test_complete
            }
        ]
        
        self._append_test_log("="*50)
        self._append_test_log("交互测试系统已就绪")
        self._append_test_log("="*50)
    
    def _append_test_log(self, message):
        """添加测试日志"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.test_result_text.append(f"[{timestamp}] {message}")
    
    def _start_interactive_test(self):
        """开始交互测试"""
        self.is_testing = True
        self.test_step_index = 0
        self.test_feedback_list = []
        
        self.btn_start_test.setEnabled(False)
        self.btn_next_step.setEnabled(True)
        self.btn_feedback.setEnabled(True)
        self.btn_submit_feedback.setEnabled(True)
        
        self._append_test_log("开始交互测试...")
        self._show_current_step()
    
    def _show_current_step(self):
        """显示当前测试步骤"""
        if self.test_step_index < len(self.test_steps):
            step = self.test_steps[self.test_step_index]
            self.test_step_label.setText(step["description"])
            self.test_status_label.setText(f"当前步骤：{step['name']}")
            self._append_test_log(f"→ 进入步骤：{step['name']}")
            
            # 自动执行检查
            if step["check"]:
                step["check"]()
        else:
            self._finish_test()
    
    def _next_test_step(self):
        """进入下一步"""
        self.test_step_index += 1
        self.feedback_input.clear()
        
        if self.test_step_index < len(self.test_steps):
            self._show_current_step()
        else:
            self._finish_test()
    
    def _finish_test(self):
        """完成测试"""
        self.is_testing = False
        self.test_step_index = len(self.test_steps)
        
        self.btn_start_test.setEnabled(True)
        self.btn_next_step.setEnabled(False)
        self.btn_feedback.setEnabled(False)
        self.btn_submit_feedback.setEnabled(False)
        
        self.test_step_label.setText("测试完成！感谢您的参与。")
        self.test_status_label.setText("测试已结束")
        
        self._append_test_log("="*50)
        self._append_test_log("测试完成总结：")
        if self.test_feedback_list:
            self._append_test_log(f"收集到 {len(self.test_feedback_list)} 个问题反馈：")
            for i, feedback in enumerate(self.test_feedback_list, 1):
                self._append_test_log(f"  {i}. [{feedback['step']}] {feedback['problem']}")
        else:
            self._append_test_log("未收集到问题反馈，测试通过！")
        self._append_test_log("="*50)
    
    def _show_feedback_dialog(self):
        """显示问题反馈对话框"""
        if self.test_step_index >= len(self.test_steps):
            return
            
        step = self.test_steps[self.test_step_index]
        
        feedback, ok = QMessageBox.question(
            self, 
            "问题反馈",
            f"当前步骤：{step['name']}\n\n这一步是否有问题？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if ok and feedback == QMessageBox.Yes:
            self.feedback_input.setFocus()
            self._append_test_log("用户反馈：当前步骤有问题")
    
    def _submit_feedback(self):
        """提交问题反馈"""
        feedback_text = self.feedback_input.toPlainText().strip()
        
        if not feedback_text:
            QMessageBox.warning(self, "提示", "请先输入问题描述！")
            return
        
        if self.test_step_index >= len(self.test_steps):
            step_name = "测试完成"
        else:
            step_name = self.test_steps[self.test_step_index]["name"]
        
        # 保存反馈
        feedback_data = {
            "step": step_name,
            "step_index": self.test_step_index,
            "problem": feedback_text,
            "timestamp": str(__import__("datetime").datetime.now())
        }
        self.test_feedback_list.append(feedback_data)
        
        self._append_test_log(f"✗ 问题反馈已提交：{feedback_text}")
        
        # 弹出对话框显示反馈已记录
        QMessageBox.information(
            self, 
            "反馈已记录",
            f"问题已记录：\n\n步骤：{step_name}\n问题：{feedback_text}\n\n开发人员将根据您的反馈优化系统。"
        )
        
        self.feedback_input.clear()
        
        # 自动保存反馈到文件
        self._save_feedback_to_file()
    
    def _save_feedback_to_file(self):
        """保存反馈到文件"""
        import json
        try:
            feedback_file = os.path.join(DEFAULT_CONFIG_DIR, "test_feedback.json")
            
            # 读取现有反馈
            existing_feedback = []
            if os.path.exists(feedback_file):
                with open(feedback_file, 'r', encoding='utf-8') as f:
                    existing_feedback = json.load(f)
            
            # 添加新反馈
            existing_feedback.extend(self.test_feedback_list)
            
            # 保存
            with open(feedback_file, 'w', encoding='utf-8') as f:
                json.dump(existing_feedback, f, indent=2, ensure_ascii=False)
                
            self._append_test_log(f"反馈已保存到：{feedback_file}")
        except Exception as e:
            self._append_test_log(f"保存反馈失败：{str(e)}")
    
    # 检查函数
    def _check_camera(self):
        """检查摄像头"""
        if self.camera_thread and self.camera_thread.isRunning():
            self.test_status_label.setText("✓ 摄像头正常运行")
            self.test_status_label.setStyleSheet(self._get_label_style(size=12, color="#00FF96"))
            self._append_test_log("✓ 摄像头检查：正常")
        else:
            self.test_status_label.setText("⚠ 摄像头可能未启动")
            self.test_status_label.setStyleSheet(self._get_label_style(size=12, color="#FF9600"))
            self._append_test_log("⚠ 摄像头检查：请确认摄像头已启动")
    
    def _check_template_loaded(self):
        """检查模板是否加载"""
        line_loaded = self.line_template is not None
        shape_loaded = self.shape_template is not None
        
        if line_loaded or shape_loaded:
            self._append_test_log("✓ 模板检查：已加载模板")
            if line_loaded:
                self._append_test_log("  - 双直线模板已加载")
            if shape_loaded:
                self._append_test_log("  - 外形模板已加载")
        else:
            self._append_test_log("⚠ 模板检查：未加载模板，请手动设置或加载")
    
    def _check_template_setup(self):
        """检查模板设置"""
        self._append_test_log("请操作：设置直线模板并测试撤销功能")
    
    def _check_algorithm_selection(self):
        """检查算法选择"""
        self._append_test_log("请操作：切换不同算法并观察结果")
    
    def _check_calibration(self):
        """检查标定"""
        if self.calibration_data and self.calibration_data.is_calibrated:
            self._append_test_log("✓ 标定检查：已完成标定")
            self._append_test_log(f"  - 像素/毫米 (X): {self.calibration_data.pixel_per_mm_x:.2f}")
            self._append_test_log(f"  - 像素/毫米 (Y): {self.calibration_data.pixel_per_mm_y:.2f}")
        else:
            self._append_test_log("⚠ 标定检查：未完成标定")
    
    def _check_manual_test_start(self):
        """检查手动测试启动"""
        self._append_test_log("请操作：点击「手动测试」按钮")
    
    def _check_dynamic_detection(self):
        """检查动态检测"""
        self._append_test_log("请观察：移动纸张时检测结果是否更新")
    
    def _check_detection_accuracy(self):
        """检查检测精度"""
        self._append_test_log("请确认：纸张在原位置时偏移值是否接近零")
    
    def _check_shape_template(self):
        """检查异形模板"""
        self._append_test_log("请操作：测试异形模板功能（可选）")
    
    def _check_test_complete(self):
        """检查测试完成"""
        self._append_test_log("✓ 所有测试步骤已完成")
    
    # ==================== 模拟测试功能 ====================

    def _setup_simulation_tab(self):
        """设置模拟测试标签页"""
        outer = QVBoxLayout(self.sim_tab)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setAutoFillBackground(False)
        scroll.viewport().setAutoFillBackground(False)
        # 右侧为固定窄面板：禁用横向滚动，内容按纵向单列自适应
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # 透明滚动区，透出与其他标签页一致的深色面板；滚动条也统一霓虹配色
        scroll.setStyleSheet("""
            QScrollArea { border: none; background: transparent; }
            QScrollArea > QWidget > QWidget { background: transparent; }
            QScrollBar:vertical { background: #0A1428; width: 12px; border-radius: 6px; }
            QScrollBar::handle:vertical { background: #0066CC; border-radius: 6px; min-height: 30px; }
            QScrollBar::handle:vertical:hover { background: #0088FF; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
        """)
        outer.addWidget(scroll)

        container = QWidget()
        container.setAutoFillBackground(False)
        # 作用域仅限模拟页：统一未单独设样式的字段标签/复选框/单选框配色与字体
        fs = self._get_scaled_font_size
        container.setStyleSheet(f"""
            QLabel {{ color: #9FD8FF; font: normal {fs(12)}px Consolas; }}
            QCheckBox, QRadioButton {{
                color: #00FFFF; font: normal {fs(12)}px Consolas;
                spacing: 6px; background: transparent;
            }}
            QCheckBox::indicator, QRadioButton::indicator {{ width: 15px; height: 15px; }}
        """)
        scroll.setWidget(container)
        layout = QVBoxLayout(container)
        layout.setSpacing(12)

        # ---------- 1. 模拟摄像头控制 ----------
        cam_group = QGroupBox("① 模拟摄像头（无需真实相机）")
        cam_group.setStyleSheet(self._get_group_style())
        cam_layout = QVBoxLayout(cam_group)

        btn_row = QHBoxLayout()
        self.btn_sim_camera = QPushButton("▶ 启动模拟摄像头")
        self.btn_sim_camera.setStyleSheet(self._get_button_style("#00AA66"))
        self.btn_sim_camera.clicked.connect(self._toggle_sim_camera)
        btn_row.addWidget(self.btn_sim_camera)
        cam_layout.addLayout(btn_row)

        self.sim_cam_status = QLabel("状态：未启动")
        self.sim_cam_status.setStyleSheet(self._get_label_style(12, "#FF6666", True))
        cam_layout.addWidget(self.sim_cam_status)

        tip = QLabel("启动后将在主画面生成模拟纸张，可直接设置模板、手动测试。")
        tip.setWordWrap(True)
        tip.setStyleSheet(self._get_label_style(11, "#7E93B8"))
        cam_layout.addWidget(tip)
        layout.addWidget(cam_group)

        # ---------- 2. 工况预设 ----------
        preset_group = QGroupBox("② 实际工况预设")
        preset_group.setStyleSheet(self._get_group_style())
        preset_layout = QVBoxLayout(preset_group)

        self.sim_preset_combo = QComboBox()
        self.sim_preset_combo.setStyleSheet(self._get_combo_style())
        for name in WorkConditionPresets.all_presets().keys():
            self.sim_preset_combo.addItem(name)
        preset_layout.addWidget(self.sim_preset_combo)

        self.btn_apply_preset = QPushButton("应用该工况")
        self.btn_apply_preset.setStyleSheet(self._get_button_style("#0088FF"))
        self.btn_apply_preset.clicked.connect(self._apply_selected_preset)
        preset_layout.addWidget(self.btn_apply_preset)
        layout.addWidget(preset_group)

        # ---------- 3. 工况参数调节 ----------
        param_group = QGroupBox("③ 工况参数调节")
        param_group.setStyleSheet(self._get_group_style())
        param_grid = QGridLayout(param_group)
        param_grid.setHorizontalSpacing(8)
        param_grid.setVerticalSpacing(8)

        # 单列表单（适配右侧窄面板）：每行 标签 + 输入框
        def _spin(row, text, widget):
            param_grid.addWidget(QLabel(text), row, 0)
            widget.setStyleSheet(self._get_spinbox_style())
            param_grid.addWidget(widget, row, 1)
            return widget

        self.sim_offset_x = _spin(0, "X偏移(px):", QDoubleSpinBox())
        self.sim_offset_x.setRange(-100, 100)
        self.sim_offset_x.setValue(0)

        self.sim_offset_y = _spin(1, "Y偏移(px):", QDoubleSpinBox())
        self.sim_offset_y.setRange(-100, 100)
        self.sim_offset_y.setValue(0)

        self.sim_angle = _spin(2, "旋转角(°):", QDoubleSpinBox())
        self.sim_angle.setRange(-10, 10)
        self.sim_angle.setDecimals(2)
        self.sim_angle.setSingleStep(0.5)
        self.sim_angle.setValue(0)

        self.sim_noise = _spin(3, "高斯噪声:", QSpinBox())
        self.sim_noise.setRange(0, 50)
        self.sim_noise.setValue(0)

        self.sim_brightness = _spin(4, "亮度:", QSpinBox())
        self.sim_brightness.setRange(-100, 100)
        self.sim_brightness.setValue(0)

        self.sim_contrast = _spin(5, "对比度:", QSpinBox())
        self.sim_contrast.setRange(-100, 100)
        self.sim_contrast.setValue(0)

        self.sim_saltpepper = _spin(6, "椒盐噪声:", QDoubleSpinBox())
        self.sim_saltpepper.setRange(0, 0.05)
        self.sim_saltpepper.setDecimals(3)
        self.sim_saltpepper.setSingleStep(0.002)
        self.sim_saltpepper.setValue(0)

        self.sim_blur = _spin(7, "运动模糊:", QSpinBox())
        self.sim_blur.setRange(0, 21)
        self.sim_blur.setSingleStep(2)
        self.sim_blur.setValue(0)

        param_grid.setColumnStretch(1, 1)

        # 开关项
        opt_row = QHBoxLayout()
        self.chk_sim_occlusion = QCheckBox("遮挡")
        self.chk_sim_interference = QCheckBox("边缘干扰")
        self.chk_sim_dynamic = QCheckBox("动态运动")
        for chk in (self.chk_sim_occlusion, self.chk_sim_interference, self.chk_sim_dynamic):
            opt_row.addWidget(chk)
        opt_row.addStretch(1)
        param_grid.addLayout(opt_row, 8, 0, 1, 2)

        param_btn_row = QHBoxLayout()
        self.btn_apply_sim_params = QPushButton("应用参数")
        self.btn_apply_sim_params.setStyleSheet(self._get_button_style("#00AA66"))
        self.btn_apply_sim_params.clicked.connect(self._apply_sim_params)
        param_btn_row.addWidget(self.btn_apply_sim_params)

        self.btn_reset_sim_params = QPushButton("重置")
        self.btn_reset_sim_params.setStyleSheet(self._get_button_style("#FF6666"))
        self.btn_reset_sim_params.clicked.connect(self._reset_sim_params)
        param_btn_row.addWidget(self.btn_reset_sim_params)
        param_grid.addLayout(param_btn_row, 9, 0, 1, 2)

        layout.addWidget(param_group)

        # ---------- 4. 真值 vs 检测 实时对比 ----------
        cmp_group = QGroupBox("④ 真实值 / 检测值 实时对比")
        cmp_group.setStyleSheet(self._get_group_style())
        cmp_grid = QGridLayout(cmp_group)
        cmp_grid.setHorizontalSpacing(6)
        headers = ["项目", "真实值", "检测值", "误差"]
        for j in range(1, 4):
            cmp_grid.setColumnStretch(j, 1)
        for j, htxt in enumerate(headers):
            lbl = QLabel(htxt)
            lbl.setStyleSheet(self._get_label_style(12, "#00FFFF", True))
            cmp_grid.addWidget(lbl, 0, j)

        self.sim_true_x = QLabel("--")
        self.sim_det_x = QLabel("--")
        self.sim_err_x = QLabel("--")
        self.sim_true_y = QLabel("--")
        self.sim_det_y = QLabel("--")
        self.sim_err_y = QLabel("--")
        self.sim_true_a = QLabel("--")
        self.sim_det_a = QLabel("--")
        self.sim_err_a = QLabel("--")

        cmp_rows = [
            ("X(px)", self.sim_true_x, self.sim_det_x, self.sim_err_x, "#00FF96"),
            ("Y(px)", self.sim_true_y, self.sim_det_y, self.sim_err_y, "#00FF96"),
            ("A(°)", self.sim_true_a, self.sim_det_a, self.sim_err_a, "#FF9600"),
        ]
        for i, (name, t_lbl, d_lbl, e_lbl, val_color) in enumerate(cmp_rows, start=1):
            cmp_grid.addWidget(QLabel(name), i, 0)
            t_lbl.setStyleSheet(self._get_label_style(12, val_color, True))
            d_lbl.setStyleSheet(self._get_label_style(12, val_color, True))
            e_lbl.setStyleSheet(self._get_label_style(12, "#FF6666", True))
            cmp_grid.addWidget(t_lbl, i, 1)
            cmp_grid.addWidget(d_lbl, i, 2)
            cmp_grid.addWidget(e_lbl, i, 3)
        layout.addWidget(cmp_group)

        # ---------- 5. 批量自动化测试 ----------
        batch_group = QGroupBox("⑤ 批量自动化测试（需先在基准工况设置好模板）")
        batch_group.setStyleSheet(self._get_group_style())
        batch_layout = QVBoxLayout(batch_group)

        range_row = QHBoxLayout()
        self.radio_current_preset = QRadioButton("仅当前工况")
        self.radio_all_presets = QRadioButton("全部工况")
        # 默认仅当前工况（标准光源基准），配合"仅角度随机"贴近实际考核
        self.radio_current_preset.setChecked(True)
        for r in (self.radio_current_preset, self.radio_all_presets):
            range_row.addWidget(r)
        batch_layout.addLayout(range_row)

        sample_row = QHBoxLayout()
        sample_row.addWidget(QLabel("每工况样本数:"))
        self.sim_samples_spin = QSpinBox()
        self.sim_samples_spin.setRange(1, 200)
        self.sim_samples_spin.setValue(10)
        self.sim_samples_spin.setStyleSheet(self._get_spinbox_style())
        sample_row.addWidget(self.sim_samples_spin)

        # 随机变化模式：默认标准光源下仅角度随机（符合实际考核工况）
        sample_row.addWidget(QLabel("随机模式:"))
        self.sim_random_mode_combo = QComboBox()
        self.sim_random_mode_combo.setStyleSheet(self._get_combo_style())
        self.sim_random_mode_combo.addItem("仅角度随机（标准位）", "angle_only")
        self.sim_random_mode_combo.addItem("位置+角度随机", "all")
        self.sim_random_mode_combo.addItem("仅位置随机", "position_only")
        self.sim_random_mode_combo.addItem("固定不随机", "fixed")
        sample_row.addWidget(self.sim_random_mode_combo)
        batch_layout.addLayout(sample_row)

        batch_btn_row = QHBoxLayout()
        self.btn_start_batch = QPushButton("▶ 开始批量测试")
        self.btn_start_batch.setStyleSheet(self._get_button_style("#00AA66"))
        self.btn_start_batch.clicked.connect(self._start_batch_test)
        batch_btn_row.addWidget(self.btn_start_batch)

        self.btn_stop_batch = QPushButton("■ 停止")
        self.btn_stop_batch.setStyleSheet(self._get_button_style("#FF6666"))
        self.btn_stop_batch.clicked.connect(self._stop_batch_test)
        self.btn_stop_batch.setEnabled(False)
        batch_btn_row.addWidget(self.btn_stop_batch)
        batch_layout.addLayout(batch_btn_row)

        self.sim_progress = QProgressBar()
        self.sim_progress.setStyleSheet(
            "QProgressBar { border: 1px solid #0066CC; border-radius: 4px; text-align: center; color: white; background: #0A1428; }"
            "QProgressBar::chunk { background-color: #00AA66; }"
        )
        self.sim_progress.setValue(0)
        batch_layout.addWidget(self.sim_progress)

        self.sim_progress_label = QLabel("等待开始批量测试")
        self.sim_progress_label.setStyleSheet(self._get_label_style(11, "#7E93B8"))
        batch_layout.addWidget(self.sim_progress_label)

        # —— 批量过程实时可视化：每一帧画面 + 真值/检测位姿叠加 + 判定 ——
        self.sim_preview = QLabel("批量测试时在此逐帧显示：\n青色圆圈=真实位姿   绿色=检测合格   红色=超差/漏检")
        self.sim_preview.setAlignment(Qt.AlignCenter)
        self.sim_preview.setMinimumHeight(220)
        self.sim_preview.setStyleSheet(
            "QLabel { background-color: #0A1428; border: 2px solid #0066CC;"
            " border-radius: 8px; color: #7E93B8; font: normal 11px Consolas; }"
        )
        batch_layout.addWidget(self.sim_preview)

        self.sim_case_info = QLabel("当前用例：—")
        self.sim_case_info.setWordWrap(True)
        self.sim_case_info.setStyleSheet(self._get_label_style(11, "#9FD8FF"))
        batch_layout.addWidget(self.sim_case_info)

        layout.addWidget(batch_group)

        # ---------- 6. 统计结果 ----------
        stats_group = QGroupBox("⑥ 测试统计")
        stats_group.setStyleSheet(self._get_group_style())
        stats_layout = QVBoxLayout(stats_group)

        # 整体统计结果（批量完成后刷新）
        self.sim_overall_stats = QLabel("整体统计：尚未执行批量测试")
        self.sim_overall_stats.setWordWrap(True)
        self.sim_overall_stats.setStyleSheet(
            "QLabel { background-color: #0A1428; border: 1px solid #0066CC;"
            " border-radius: 6px; padding: 8px; color: #E0E6F0; font: normal 12px Consolas; }"
        )
        stats_layout.addWidget(self.sim_overall_stats)

        self.sim_stats_table = QTableWidget(0, 7)
        self.sim_stats_table.setHorizontalHeaderLabels(
            ["工况", "检出率", "合格率(考核)", "平均位置误差(px)", "RMSE(px)", "平均角度误差(°)", "平均耗时(ms)"]
        )
        self.sim_stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.sim_stats_table.setStyleSheet(self._get_table_style())
        self.sim_stats_table.setMaximumHeight(160)
        stats_layout.addWidget(self.sim_stats_table)
        layout.addWidget(stats_group)

        # ---------- 7. 报告导出 ----------
        export_row = QHBoxLayout()
        self.btn_export_json = QPushButton("导出JSON报告")
        self.btn_export_json.setStyleSheet(self._get_button_style("#FF9600"))
        self.btn_export_json.clicked.connect(self._export_sim_json)
        export_row.addWidget(self.btn_export_json)

        self.btn_export_csv = QPushButton("导出CSV明细")
        self.btn_export_csv.setStyleSheet(self._get_button_style("#FF9600"))
        self.btn_export_csv.clicked.connect(self._export_sim_csv)
        export_row.addWidget(self.btn_export_csv)
        layout.addLayout(export_row)

        # ---------- 8. 日志 ----------
        self.sim_log = QTextEdit()
        self.sim_log.setReadOnly(True)
        self.sim_log.setStyleSheet(self._get_text_edit_style())
        self.sim_log.setMaximumHeight(120)
        self.sim_log.setPlaceholderText("模拟测试日志...")
        layout.addWidget(self.sim_log)

        # 模拟实时对比定时器
        self.sim_compare_timer = QTimer()
        self.sim_compare_timer.setInterval(500)
        self.sim_compare_timer.timeout.connect(self._update_sim_comparison)
        self.batch_thread = None

        self._sim_log("模拟测试模块已就绪。先启动模拟摄像头，在基准工况设置模板后即可测试。")

    def _sim_log(self, message: str):
        """向模拟测试日志窗口追加信息"""
        ts = time.strftime("%H:%M:%S")
        self.sim_log.append(f"[{ts}] {message}")

    # ---------- 模拟摄像头 ----------
    def _toggle_sim_camera(self):
        """启停模拟摄像头"""
        if self.sim_camera_thread and self.sim_camera_thread.isRunning():
            self.sim_camera_thread.stop()
            self.sim_camera_thread = None
            self.btn_sim_camera.setText("▶ 启动模拟摄像头")
            self.sim_cam_status.setText("状态：未启动")
            self.sim_cam_status.setStyleSheet(self._get_label_style(12, "#FF6666", True))
            self.sim_compare_timer.stop()
            self._sim_log("模拟摄像头已停止")
            return

        # 若真实摄像头开着，先关闭避免冲突
        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.stop()
            self.btn_open_cam.setText("📷 打开摄像头")

        self._apply_sim_params(need_log=False)
        self.sim_camera_thread = SimulatedCameraThread(self.sim_config)
        self.sim_camera_thread.frame_ready.connect(self._on_sim_frame_ready)
        self.sim_camera_thread.truth_ready.connect(self._on_sim_truth_ready)
        self.sim_camera_thread.error_occurred.connect(self._on_sim_error)
        self.sim_camera_thread.start()
        self.btn_sim_camera.setText("■ 停止模拟摄像头")
        self.sim_cam_status.setText("状态：模拟运行中")
        self.sim_cam_status.setStyleSheet(self._get_label_style(12, "#00FF96", True))
        self.sim_compare_timer.start()
        self._sim_log("模拟摄像头已启动")

    def _on_sim_frame_ready(self, frame):
        """模拟帧直接进入主画布（与真实摄像头同一通道）"""
        if self._batch_display_active:
            # 批量测试可视化期间，主画面由批量用例帧接管
            return
        self.image_canvas.set_frame(frame)

    def _on_sim_truth_ready(self, truth: dict):
        """保存当前帧真实位姿"""
        self.sim_last_truth = truth

    def _on_sim_error(self, err: str):
        self._sim_log("错误：" + err)
        QMessageBox.critical(self, "模拟摄像头错误", err)

    # ---------- 预设与参数 ----------
    def _apply_selected_preset(self):
        """应用下拉框所选工况预设"""
        name = self.sim_preset_combo.currentText()
        factory = WorkConditionPresets.all_presets().get(name)
        if factory is None:
            return
        cfg = factory()
        self.sim_config = cfg
        self._load_sim_config_to_ui(cfg)
        if self.sim_camera_thread and self.sim_camera_thread.isRunning():
            self.sim_camera_thread.set_config(cfg)
        self._sim_log(f"已应用工况预设：{name}")

    def _load_sim_config_to_ui(self, cfg: SimulationConfig):
        """把配置回填到参数控件"""
        self.sim_offset_x.setValue(cfg.offset_x)
        self.sim_offset_y.setValue(cfg.offset_y)
        self.sim_angle.setValue(cfg.angle)
        self.sim_noise.setValue(cfg.noise_level)
        self.sim_brightness.setValue(cfg.brightness)
        self.sim_contrast.setValue(cfg.contrast)
        self.sim_saltpepper.setValue(cfg.salt_pepper)
        self.sim_blur.setValue(cfg.motion_blur)
        self.chk_sim_occlusion.setChecked(cfg.occlusion_enabled)
        self.chk_sim_interference.setChecked(cfg.edge_interference)
        self.chk_sim_dynamic.setChecked(cfg.dynamic_motion)

    def _collect_sim_config(self) -> SimulationConfig:
        """从参数控件收集配置（保留分辨率等基础项）"""
        cfg = self.sim_config
        cfg.offset_x = float(self.sim_offset_x.value())
        cfg.offset_y = float(self.sim_offset_y.value())
        cfg.angle = float(self.sim_angle.value())
        cfg.noise_level = int(self.sim_noise.value())
        cfg.brightness = int(self.sim_brightness.value())
        cfg.contrast = int(self.sim_contrast.value())
        cfg.salt_pepper = float(self.sim_saltpepper.value())
        cfg.motion_blur = int(self.sim_blur.value())
        cfg.occlusion_enabled = self.chk_sim_occlusion.isChecked()
        cfg.edge_interference = self.chk_sim_interference.isChecked()
        cfg.dynamic_motion = self.chk_sim_dynamic.isChecked()
        if cfg.dynamic_motion:
            cfg.motion_speed_x, cfg.motion_speed_y, cfg.motion_rot_speed = 1.2, 0.6, 0.08
        return cfg

    def _apply_sim_params(self, need_log: bool = True):
        """应用手动调节的参数"""
        cfg = self._collect_sim_config()
        if self.sim_camera_thread and self.sim_camera_thread.isRunning():
            self.sim_camera_thread.set_config(cfg)
        if need_log:
            self._sim_log(
                f"参数已应用：dx={cfg.offset_x:.1f}, dy={cfg.offset_y:.1f}, "
                f"angle={cfg.angle:.2f}°, 噪声={cfg.noise_level}, 亮度={cfg.brightness}"
            )

    def _reset_sim_params(self):
        """重置为基准工况"""
        self.sim_config = WorkConditionPresets.baseline()
        self._load_sim_config_to_ui(self.sim_config)
        if self.sim_camera_thread and self.sim_camera_thread.isRunning():
            self.sim_camera_thread.set_config(self.sim_config)
        self.sim_preset_combo.setCurrentIndex(0)
        self._sim_log("已重置为基准工况")

    # ---------- 帧级检测回调（供实时对比 & 批量测试使用） ----------
    def _run_detection_on_frame(self, frame: np.ndarray) -> dict:
        """
        在给定帧上执行当前模式的检测，返回标准化结果：
        {detected, dx, dy, angle, score}
        偏移以"模板基准中心/角度"为参照（与模拟真值同口径）。
        """
        try:
            if frame is None or frame.size == 0:
                return {"detected": False, "note": "空帧"}

            algo = self._get_selected_algorithm()

            if self.current_mode == "line":
                return self._detect_line_on_frame(frame, algo)
            else:
                return self._detect_shape_on_frame(frame, algo)
        except Exception as e:
            return {"detected": False, "note": f"检测异常: {e}"}

    @staticmethod
    def _center_from_line_pairs(pairs):
        """
        由 (检测线, 对应模板线) 配对列表估计纸张中心（跨度无关，对部分匹配鲁棒）。

        原理：刚体旋转下"中心到边缘直线的带符号法向距离"保持不变。
        每条检测线提供一个法向约束 n·c = n·p + s，其中：
          - n 为模板法向按相对角差旋转后的单位法向（方向继承模板，规避
            检测线端点顺序导致的法向翻转）；
          - s 为模板线到模板基准中心的带符号法向距离；
          - p 为检测线上任一点（取线中心）。
        存在两两不平行约束（同时含横线与竖线）时用最小二乘精确求解；
        否则（全平行/单线）回退为线中心均值。
        """
        if not pairs:
            return None
        tpl_cx = float(np.mean([VisionAlgorithm.calculate_line_center(t)[0] for _l, t in pairs]))
        tpl_cy = float(np.mean([VisionAlgorithm.calculate_line_center(t)[1] for _l, t in pairs]))

        rows, rhs = [], []
        for line, tpl in pairs:
            (tx1, ty1), (tx2, ty2) = tpl
            tlen = math.hypot(tx2 - tx1, ty2 - ty1)
            (lx1, ly1), (lx2, ly2) = line
            llen = math.hypot(lx2 - lx1, ly2 - ly1)
            if tlen < 1e-6 or llen < 1e-6:
                continue
            # 模板单位法向 与 模板线到模板基准中心的带符号距离
            ntx, nty = -(ty2 - ty1) / tlen, (tx2 - tx1) / tlen
            s = ntx * (tpl_cx - (tx1 + tx2) / 2.0) + nty * (tpl_cy - (ty1 + ty2) / 2.0)
            # 相对角差（归一化到 -90~90，规避无向直线 ±180° 跳变）
            d = (VisionAlgorithm.calculate_line_angle(line)
                 - VisionAlgorithm.calculate_line_angle(tpl))
            while d > 90:
                d -= 180
            while d < -90:
                d += 180
            rot = math.radians(d)
            nx = ntx * math.cos(rot) - nty * math.sin(rot)
            ny = ntx * math.sin(rot) + nty * math.cos(rot)
            pcx, pcy = (lx1 + lx2) / 2.0, (ly1 + ly2) / 2.0
            rows.append((nx, ny))
            rhs.append(nx * pcx + ny * pcy + s)

        if len(rows) >= 2:
            # 分组交替求解：竖边组（|nx|大）约束 cx，横边组（|ny|大）约束 cy，
            # 交叉项用对方轴线估计迭代两轮收敛。避免两约束近平行时
            # 直接最小二乘把模板毫厘级偏差放大为数千像素的病态解
            v_rows = [(nx, ny, r) for (nx, ny), r in zip(rows, rhs) if abs(nx) >= 0.7]
            h_rows = [(nx, ny, r) for (nx, ny), r in zip(rows, rhs) if abs(nx) < 0.7]
            if v_rows and h_rows:
                cx, cy = tpl_cx, tpl_cy
                for _ in range(2):
                    cx = float(np.mean([(r - ny * cy) / nx for nx, ny, r in v_rows]))
                    cy = float(np.mean([(r - nx * cx) / ny for nx, ny, r in h_rows]))
                return cx, cy

        # 回退（全平行/单线）：线中心均值
        cx = float(np.mean([VisionAlgorithm.calculate_line_center(l)[0] for l, _t in pairs]))
        cy = float(np.mean([VisionAlgorithm.calculate_line_center(l)[1] for l, _t in pairs]))
        return cx, cy

    def _detect_line_on_frame(self, frame, algo) -> dict:
        """双直线模式帧检测（外扩搜索 + 断裂合并 + 边缘点精修）"""
        canvas = self.image_canvas
        left_roi = canvas.left_roi
        right_roi = canvas.right_roi
        left_tpl = [l for l in (canvas.left_line_long, canvas.left_line_short) if l is not None]
        right_tpl = [l for l in (canvas.right_line_long, canvas.right_line_short) if l is not None]
        if not left_roi or not right_roi or not left_tpl or not right_tpl:
            return {"detected": False, "note": "双直线模板未设置完整"}

        h, w = frame.shape[:2]
        expand = 130  # 外扩搜索范围，覆盖 ≤5mm(≈50px)偏移 + ≤5°旋转扩展

        def expand_roi(roi):
            x1, y1, x2, y2 = roi
            return (max(0, x1 - expand), max(0, y1 - expand),
                    min(w, x2 + expand), min(h, y2 + expand))

        search_left, search_right = expand_roi(left_roi), expand_roi(right_roi)

        # 基准模板中心（与检测中心同口径：法向约束最小二乘，退化时回退线中心均值）
        tpl_all = left_tpl + right_tpl
        base_cx, base_cy = self._center_from_line_pairs([(t, t) for t in tpl_all])

        def detect_side(search_roi, tpl_list):
            raw = VisionAlgorithm.detect_lines_single_algorithm(frame, search_roi, algo, tpl_list)
            raw = VisionAlgorithm.merge_collinear_lines(raw)
            out = []
            for tpl in tpl_list:
                # 多候选择优：合并后的准确边缘与精修结果按边缘点支持度投票，
                # 强噪声叠加旋转时可避免精修漂移到噪声结构
                tpl_len = math.hypot(tpl[1][0] - tpl[0][0], tpl[1][1] - tpl[0][1])
                best, _score = VisionAlgorithm.refine_best_candidate(
                    frame, raw, tpl, search_roi, expected_length=tpl_len)
                if best is not None:
                    # 交叉匹配防护：纸张旋转≤5°，检测线与模板线无向夹角应很小；
                    # 横线模板误配到竖直边缘时夹角≈90°，直接拒绝
                    d = VisionAlgorithm.undirected_angle_diff(
                        VisionAlgorithm.calculate_line_angle(best),
                        VisionAlgorithm.calculate_line_angle(tpl))
                    if d <= 25:
                        out.append((best, tpl))
            return out

        # 匹配结果为 (检测线, 对应模板线) 配对
        matched = detect_side(search_left, left_tpl) + detect_side(search_right, right_tpl)

        if not matched:
            return {"detected": False, "note": "未匹配到直线"}

        det_cx, det_cy = self._center_from_line_pairs(matched)

        # 角度：每条检测线相对其各自模板线的角度差（逐线归一化到 -90~90 后取平均）。
        # 不依赖单条线的端点顺序，规避 ±90°/180° 跳变造成的角度误判
        ang_diffs = []
        for line, tpl in matched:
            d = (VisionAlgorithm.calculate_line_angle(line)
                 - VisionAlgorithm.calculate_line_angle(tpl))
            while d > 90:
                d -= 180
            while d < -90:
                d += 180
            ang_diffs.append(d)
        ang_diff = float(np.mean(ang_diffs))

        score = len(matched) / len(tpl_all)
        return {
            "detected": True,
            "dx": det_cx - base_cx,
            "dy": det_cy - base_cy,
            "angle": ang_diff,
            "score": score,
        }

    def _detect_shape_on_frame(self, frame, algo) -> dict:
        """外形模式帧检测（外扩搜索）"""
        canvas = self.image_canvas
        left_roi = canvas.left_contour_roi
        right_roi = canvas.right_contour_roi
        left_tpl = canvas.left_contour
        right_tpl = canvas.right_contour
        if left_roi is None or right_roi is None or left_tpl is None or right_tpl is None:
            return {"detected": False, "note": "外形模板未设置完整"}

        h, w = frame.shape[:2]
        expand = 130

        def expand_roi(roi):
            x1, y1, x2, y2 = roi
            return (max(0, x1 - expand), max(0, y1 - expand),
                    min(w, x2 + expand), min(h, y2 + expand))

        search_left, search_right = expand_roi(left_roi), expand_roi(right_roi)

        base_left_c = VisionAlgorithm._get_contour_center(left_tpl)
        base_right_c = VisionAlgorithm._get_contour_center(right_tpl)
        base_cx = (base_left_c[0] + base_right_c[0]) / 2
        base_cy = (base_left_c[1] + base_right_c[1]) / 2
        base_angle = math.degrees(math.atan2(
            base_right_c[1] - base_left_c[1],
            base_right_c[0] - base_left_c[0]
        ))

        left_cands = [c[0] for c in VisionAlgorithm.detect_contours_multialgorithm(
            frame, search_left, template_contour=left_tpl)]
        right_cands = [c[0] for c in VisionAlgorithm.detect_contours_multialgorithm(
            frame, search_right, template_contour=right_tpl)]

        lm, ls, _ = VisionAlgorithm.match_contours_by_position_size(left_tpl, left_cands)
        rm, rs, _ = VisionAlgorithm.match_contours_by_position_size(right_tpl, right_cands)
        if lm is None or rm is None:
            return {"detected": False, "note": "轮廓匹配失败"}

        lc = VisionAlgorithm._get_contour_center(lm)
        rc = VisionAlgorithm._get_contour_center(rm)
        det_cx = (lc[0] + rc[0]) / 2
        det_cy = (lc[1] + rc[1]) / 2
        det_angle = math.degrees(math.atan2(rc[1] - lc[1], rc[0] - lc[0]))

        return {
            "detected": True,
            "dx": det_cx - base_cx,
            "dy": det_cy - base_cy,
            "angle": det_angle - base_angle,
            "score": (ls + rs) / 2,
        }

    # ---------- 实时真值对比 ----------
    def _update_sim_comparison(self):
        """定时比对真实位姿与检测结果"""
        if self.sim_last_truth is None:
            return
        frame = self.image_canvas.current_frame
        if frame is None:
            return
        truth = self.sim_last_truth
        det = self._run_detection_on_frame(frame)

        self.sim_true_x.setText(f"{truth['offset_x']:.2f}")
        self.sim_true_y.setText(f"{truth['offset_y']:.2f}")
        self.sim_true_a.setText(f"{truth['angle_deg']:.2f}")

        if det.get("detected"):
            ex = det["dx"] - truth["offset_x"]
            ey = det["dy"] - truth["offset_y"]
            ea = det["angle"] - truth["angle_deg"]
            self.sim_det_x.setText(f"{det['dx']:.2f}")
            self.sim_det_y.setText(f"{det['dy']:.2f}")
            self.sim_det_a.setText(f"{det['angle']:.2f}")
            self.sim_err_x.setText(f"{ex:+.2f}")
            self.sim_err_y.setText(f"{ey:+.2f}")
            self.sim_err_a.setText(f"{ea:+.2f}")
        else:
            for lbl in (self.sim_det_x, self.sim_det_y, self.sim_det_a,
                        self.sim_err_x, self.sim_err_y, self.sim_err_a):
                lbl.setText("未检出")

    # ---------- 批量测试 ----------
    def _start_batch_test(self):
        """启动批量自动化测试"""
        # 模板完整性检查
        if self.current_mode == "line":
            ok = bool(self.image_canvas.left_roi and self.image_canvas.right_roi
                      and self.image_canvas.left_line_long and self.image_canvas.right_line_long)
        else:
            ok = bool(self.image_canvas.left_contour_roi and self.image_canvas.right_contour_roi
                      and self.image_canvas.left_contour is not None
                      and self.image_canvas.right_contour is not None)
        if not ok:
            QMessageBox.warning(
                self, "模板不完整",
                "请先启动模拟摄像头，在「基准工况」下设置好当前模式的模板（含左右线条/轮廓与区域）后再进行批量测试。"
            )
            return

        samples = int(self.sim_samples_spin.value())
        if self.radio_all_presets.isChecked():
            preset_names = list(WorkConditionPresets.all_presets().keys())
        else:
            preset_names = [self.sim_preset_combo.currentText()]

        self.sim_test_engine = SimulationTestEngine(self._run_detection_on_frame)
        self.batch_thread = BatchTestThread(
            self.sim_test_engine, preset_names, samples,
            self.sim_random_mode_combo.currentData() or "angle_only"
        )
        self.batch_thread.progress.connect(self._on_batch_progress)
        self.batch_thread.case_done.connect(self._on_batch_case_done)
        self.batch_thread.frame_tested.connect(self._on_batch_frame)
        self.batch_thread.finished_all.connect(self._on_batch_finished)
        self.batch_thread.error.connect(self._on_batch_error)

        total = samples * len(preset_names)
        self.sim_progress.setRange(0, total)
        self.sim_progress.setValue(0)
        self.btn_start_batch.setEnabled(False)
        self.btn_stop_batch.setEnabled(True)
        self._sim_batch_count = 0
        self._sim_batch_total = total
        self.sim_preview.clear()
        self.sim_preview.setText("批量测试进行中，逐帧渲染…")
        self.sim_case_info.setText("当前用例：—")
        # 批量期间：主画面由批量用例帧接管，暂停实时对比定时避免重复检测
        self._batch_display_active = True
        self.sim_compare_timer.stop()
        self._sim_log(f"开始批量测试：{len(preset_names)} 个工况 × {samples} 样本 = {total} 次")
        self.batch_thread.start()

    def _stop_batch_test(self):
        if self.batch_thread:
            self.batch_thread.abort()
            self._sim_log("正在停止批量测试...")

    def _on_batch_progress(self, preset_name, cur, total):
        self.sim_progress_label.setText(f"工况【{preset_name}】进度 {cur}/{total}")

    def _on_batch_case_done(self, result: TestCaseResult):
        self._sim_batch_count += 1
        self.sim_progress.setValue(self._sim_batch_count)

    # 基准画面中心（模拟场景基准真值中心，真值中心=该点+偏移）
    _SIM_BASE_CENTER = (640.0, 360.0)
    _SIM_PASS_DIST_PX = 50.0     # ≤5mm ≈ 50px
    _SIM_PASS_ANGLE_DEG = 5.0    # ≤5°

    def _on_batch_frame(self, payload: dict):
        """批量测试逐帧可视化：叠加真实/检测位姿，显示判定与误差"""
        frame = payload.get("frame")
        truth = payload.get("truth", {}) or {}
        res = payload.get("result", {}) or {}
        preset = payload.get("preset", "")
        if frame is None:
            return

        vis = frame.copy()
        h, w = vis.shape[:2]
        tcx = float(truth.get("center_x", self._SIM_BASE_CENTER[0]))
        tcy = float(truth.get("center_y", self._SIM_BASE_CENTER[1]))

        detected = bool(res.get("detected"))
        err_dist = res.get("error_dist")
        err_ang = res.get("error_angle")
        ok = bool(detected and err_dist is not None and err_ang is not None
                  and err_dist <= self._SIM_PASS_DIST_PX
                  and abs(err_ang) <= self._SIM_PASS_ANGLE_DEG)

        def crosshair(img, cx, cy, color, r):
            c = (int(round(cx)), int(round(cy)))
            cv2.circle(img, c, r, color, 2, cv2.LINE_AA)
            cv2.line(img, (c[0] - r - 6, c[1]), (c[0] + r + 6, c[1]), color, 2, cv2.LINE_AA)
            cv2.line(img, (c[0], c[1] - r - 6), (c[0], c[1] + r + 6), color, 2, cv2.LINE_AA)

        # 真实位姿：青色十字
        crosshair(vis, tcx, tcy, (255, 255, 0), 18)
        # 检测位姿：合格绿色 / 超差红色，并连线显示偏差
        if detected and res.get("det_dx") is not None and res.get("det_dy") is not None:
            dcx = self._SIM_BASE_CENTER[0] + float(res["det_dx"])
            dcy = self._SIM_BASE_CENTER[1] + float(res["det_dy"])
            col = (0, 230, 120) if ok else (60, 60, 255)
            crosshair(vis, dcx, dcy, col, 14)
            cv2.line(vis, (int(round(tcx)), int(round(tcy))),
                     (int(round(dcx)), int(round(dcy))), col, 2, cv2.LINE_AA)

        # 顶部状态条 + 判定文字
        cv2.rectangle(vis, (0, 0), (w, 32), (10, 20, 40), -1)
        cv2.putText(vis, "TRUTH", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (255, 255, 0), 2, cv2.LINE_AA)
        verdict = "PASS" if ok else ("MISS" if not detected else "FAIL")
        vcol = (0, 230, 120) if ok else (60, 60, 255)
        cv2.putText(vis, verdict, (w - 100, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, vcol, 2, cv2.LINE_AA)

        # 主画面同步显示批量用例帧（纸张位姿随用例参数变化 + 识别效果叠加）
        self.image_canvas.set_frame(vis)

        rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        lbl_w = max(160, self.sim_preview.width() - 12)
        self.sim_preview.setPixmap(
            pix.scaledToWidth(lbl_w, Qt.SmoothTransformation))

        # 用例文字信息（富文本上色）
        verdict_cn = "合格" if ok else ("漏检" if not detected else "超差")
        verdict_color = "#00FF96" if ok else "#FF5566"
        tx, ty, ta = (truth.get("offset_x", 0), truth.get("offset_y", 0),
                      truth.get("angle_deg", 0))
        if detected and res.get("det_dx") is not None:
            info = (
                f"工况：{preset}　用例#{res.get('case_id', '')}　"
                f"判定：<span style='color:{verdict_color};font-weight:bold'>{verdict_cn}</span><br>"
                f"真值 dx/dy/角度：{tx:.1f} / {ty:.1f} / {ta:.2f}°<br>"
                f"检测 dx/dy/角度：{float(res['det_dx']):.1f} / {float(res['det_dy']):.1f}"
                f" / {float(res.get('det_angle') or 0):.2f}°<br>"
                f"位置误差：<b>{float(err_dist):.2f}px</b>　"
                f"角度误差：{abs(float(err_ang)):.2f}°　"
                f"耗时：{float(res.get('elapsed_ms') or 0):.1f}ms"
            )
        else:
            info = (
                f"工况：{preset}　用例#{res.get('case_id', '')}　"
                f"判定：<span style='color:{verdict_color};font-weight:bold'>漏检</span><br>"
                f"真值 dx/dy/角度：{tx:.1f} / {ty:.1f} / {ta:.2f}°"
            )
        self.sim_case_info.setText(info)

    def _on_batch_finished(self, engine: SimulationTestEngine):
        self._batch_display_active = False
        if self.sim_camera_thread and self.sim_camera_thread.isRunning():
            self.sim_compare_timer.start()
        self.btn_start_batch.setEnabled(True)
        self.btn_stop_batch.setEnabled(False)
        stats = engine.get_statistics()
        self._refresh_sim_stats_table(engine)
        self._refresh_sim_overall_stats(stats)
        self.sim_progress_label.setText(
            f"测试完成：共 {stats['total']} 次，合格 {stats.get('pass_count', 0)} 次，"
            f"合格率 {stats.get('pass_rate', 0) * 100:.2f}%"
        )
        self._sim_log(
            f"批量测试完成：合格率 {stats.get('pass_rate', 0) * 100:.2f}%"
            f"（检出率 {stats.get('detection_rate', 0) * 100:.1f}%），"
            f"平均位置误差 {stats.get('mean_error_dist_px', 0):.3f}px，"
            f"RMSE {stats.get('rmse_dist_px', 0):.3f}px，"
            f"平均耗时 {stats.get('mean_time_ms', 0):.2f}ms"
        )

    def _refresh_sim_overall_stats(self, stats: dict):
        """刷新整体统计结果面板（口径：位置≤50px且角度≤5°为合格）"""
        if not stats or stats.get("total", 0) == 0:
            self.sim_overall_stats.setText("整体统计：暂无数据")
            return
        total = stats.get("total", 0)
        det = stats.get("detected", 0)
        miss = stats.get("miss_count", total - det)
        oot = stats.get("out_of_tol_count", 0)
        passed = stats.get("pass_count", 0)
        det_rate = stats.get("detection_rate", 0.0) * 100
        pass_rate = stats.get("pass_rate", 0.0) * 100
        rate_color = "#00FF96" if pass_rate >= 99.5 else ("#FFD24D" if pass_rate >= 95 else "#FF5566")

        html = (
            f"<b style='color:#00FFFF'>整体统计结果</b>"
            f"　总用例 <b>{total}</b>"
            f"　检出 <b>{det}</b>（{det_rate:.1f}%）"
            f"　漏检 <b style='color:#FF5566'>{miss}</b>"
            f"　检出超差 <b style='color:#FF9600'>{oot}</b>"
            f"　合格 <b style='color:{rate_color}'>{passed}</b>"
            f"　<b style='color:{rate_color}'>考核合格率 {pass_rate:.2f}%</b><br>"
            f"平均位置误差 <b>{stats.get('mean_error_dist_px', 0):.3f}px</b>"
            f"　最大 <b>{stats.get('max_error_dist_px', 0):.3f}px</b>"
            f"　RMSE <b>{stats.get('rmse_dist_px', 0):.3f}px</b>"
            f"　平均角度误差 <b>{stats.get('mean_error_angle_deg', 0):.3f}°</b>"
            f"　最大 <b>{stats.get('max_error_angle_deg', 0):.3f}°</b>"
            f"　平均耗时 <b>{stats.get('mean_time_ms', 0):.2f}ms</b>"
        )
        self.sim_overall_stats.setText(html)

    def _on_batch_error(self, err: str):
        self._batch_display_active = False
        if self.sim_camera_thread and self.sim_camera_thread.isRunning():
            self.sim_compare_timer.start()
        self.btn_start_batch.setEnabled(True)
        self.btn_stop_batch.setEnabled(False)
        self._sim_log("批量测试错误：" + err)
        QMessageBox.critical(self, "批量测试错误", err)

    def _refresh_sim_stats_table(self, engine: SimulationTestEngine):
        """刷新按工况统计的表格（口径与整体统计一致）"""
        by_preset = engine.get_statistics_by_preset()
        self.sim_stats_table.setRowCount(len(by_preset))
        for i, (name, st) in enumerate(by_preset.items()):
            det_rate = st.get("detection_rate", st.get("success_rate", 0)) * 100
            pass_rate = st.get("pass_rate", 0) * 100
            items = [
                name,
                f"{det_rate:.1f}% ({st.get('detected', 0)}/{st.get('total', 0)})",
                f"{pass_rate:.2f}% ({st.get('pass_count', 0)}/{st.get('total', 0)})",
                f"{st.get('mean_error_dist_px', 0):.3f}",
                f"{st.get('rmse_dist_px', 0):.3f}",
                f"{st.get('mean_error_angle_deg', 0):.3f}",
                f"{st.get('mean_time_ms', 0):.2f}",
            ]
            for j, txt in enumerate(items):
                item = QTableWidgetItem(txt)
                if j == 2:
                    item.setForeground(QColor("#00FF96" if pass_rate >= 99.5 else
                                              ("#FFD24D" if pass_rate >= 95 else "#FF5566")))
                else:
                    item.setForeground(QColor("#00FF96" if j == 1 and det_rate >= 95 else "#E0E6F0"))
                self.sim_stats_table.setItem(i, j, item)

    # ---------- 报告导出 ----------
    def _export_sim_json(self):
        if not self.sim_test_engine or not self.sim_test_engine.results:
            QMessageBox.information(self, "提示", "暂无测试数据，请先执行批量测试。")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出JSON测试报告", "simulation_report.json", "JSON文件 (*.json)"
        )
        if path:
            self.sim_test_engine.export_json(path, extra={"mode": self.current_mode})
            self._sim_log(f"JSON报告已导出：{path}")
            QMessageBox.information(self, "导出成功", f"报告已保存：\n{path}")

    def _export_sim_csv(self):
        if not self.sim_test_engine or not self.sim_test_engine.results:
            QMessageBox.information(self, "提示", "暂无测试数据，请先执行批量测试。")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出CSV明细", "simulation_cases.csv", "CSV文件 (*.csv)"
        )
        if path:
            self.sim_test_engine.export_csv(path)
            self._sim_log(f"CSV明细已导出：{path}")
            QMessageBox.information(self, "导出成功", f"明细已保存：\n{path}")

    # ==================== 关闭事件 ====================

    def closeEvent(self, event):
        """关闭事件"""
        # 保存当前会话
        self._save_current_session()

        if self.camera_thread:
            self.camera_thread.stop()

        if self.sim_camera_thread:
            self.sim_camera_thread.stop()

        if getattr(self, "batch_thread", None):
            self.batch_thread.abort()
            self.batch_thread.wait(1000)

        event.accept()


def main():
    """主函数"""
    import sys
    import traceback
    
    # 设置全局异常处理
    def global_exception_handler(exc_type, exc_value, exc_traceback):
        """全局异常处理"""
        error_msg = f"程序发生错误:\n\n类型: {exc_type.__name__}\n信息: {exc_value}\n\n堆栈跟踪:\n{''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))}"
        
        # 尝试显示错误对话框
        try:
            from PyQt5.QtWidgets import QApplication, QMessageBox
            
            app = QApplication.instance()
            if app:
                QMessageBox.critical(None, "程序错误", error_msg)
            else:
                print(error_msg)
        except:
            print(error_msg)
        
        # 退出程序
        sys.exit(1)
    
    # 设置全局异常处理
    sys.excepthook = global_exception_handler
    
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    try:
        window = MainWindow()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        error_msg = f"启动程序时发生错误:\n\n{str(e)}\n\n{traceback.format_exc()}"
        try:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(None, "启动错误", error_msg)
        except:
            print(error_msg)
        sys.exit(1)


if __name__ == "__main__":
    main()
