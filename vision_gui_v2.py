#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
贴内衬机视觉定位系统 - 改进版GUI主程序
工业级、科技风、HUD风格界面

改进内容：
1. 双直线模板：每侧两条垂直线（长和宽），然后区域设置
2. 异形模板：手动设置轮廓，然后区域设置
3. 验证逻辑：区域设置成功后才能保存模板
4. 验证逻辑：获取模板后才能进行手动测试
5. 完善相机标定：完整的标定流程和原理说明
"""

import sys
import math
import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QFrame, QGridLayout, QSlider,
    QComboBox, QGroupBox, QTabWidget, QTableWidget, 
    QTableWidgetItem, QHeaderView, QSpinBox, QDoubleSpinBox,
    QFileDialog, QMessageBox, QStatusBar, QTextEdit, QCheckBox
)
from PyQt5.QtCore import (
    Qt, pyqtSignal, QThread, QRect, QTimer, QPointF
)
from PyQt5.QtGui import (
    QPainter, QColor, QPen, QBrush, QFont, QLinearGradient,
    QRadialGradient, QPolygonF, QImage, QCursor
)

from vision_utils_v2 import (
    CalibrationData, DualLineTemplate, ShapeTemplate, DetectionResult,
    CameraCalibration, VisionAlgorithm, TemplateManager, CameraUtils
)


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
    支持绘制直线、矩形、轮廓等
    """
    roi_selected = pyqtSignal(tuple)
    line_drawn = pyqtSignal(tuple)
    contour_selected = pyqtSignal(tuple)
    
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
        self.drawing_mode = None  # 'line', 'roi', 'contour'
        self.drawing_start = None
        self.drawing_end = None
        
        # 双直线模板数据（每侧两条垂直线）
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
        self.detected_center = None
        self.detected_angle = 0.0
        
        # 动画效果
        self.scan_phase = 0
        self.pulse_phase = 0
        
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
        
        # 绘制ROI区域
        self._draw_rois(painter)
        
        # 绘制模板直线
        self._draw_template_lines(painter)
        
        # 绘制模板轮廓
        self._draw_template_contours(painter)
        
        # 绘制检测结果
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
        painter.setFont(QFont("Consolas", 16, QFont.Bold))
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
        
        # 左ROI
        if self.left_roi:
            x1, y1, x2, y2 = self._scale_rect(self.left_roi)
            pen = QPen(NeonStyle.NEON_GREEN, 2, Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(QBrush(QColor(0, 255, 150, 30)))
            painter.drawRect(x1, y1, x2-x1, y2-y1)
            painter.setFont(QFont("Consolas", 9))
            painter.drawText(x1+5, y1+15, "左ROI")
        
        # 右ROI
        if self.right_roi:
            x1, y1, x2, y2 = self._scale_rect(self.right_roi)
            pen = QPen(NeonStyle.NEON_ORANGE, 2, Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(QBrush(QColor(255, 150, 0, 30)))
            painter.drawRect(x1, y1, x2-x1, y2-y1)
            painter.drawText(x1+5, y1+15, "右ROI")
        
        # 异形左ROI
        if self.left_contour_roi:
            x1, y1, x2, y2 = self._scale_rect(self.left_contour_roi)
            pen = QPen(NeonStyle.NEON_GREEN, 2, Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(QBrush(QColor(0, 255, 150, 30)))
            painter.drawRect(x1, y1, x2-x1, y2-y1)
            painter.drawText(x1+5, y1+15, "左轮廓ROI")
        
        # 异形右ROI
        if self.right_contour_roi:
            x1, y1, x2, y2 = self._scale_rect(self.right_contour_roi)
            pen = QPen(NeonStyle.NEON_ORANGE, 2, Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(QBrush(QColor(255, 150, 0, 30)))
            painter.drawRect(x1, y1, x2-x1, y2-y1)
            painter.drawText(x1+5, y1+15, "右轮廓ROI")
        
        painter.setOpacity(1.0)
        
    def _draw_template_lines(self, painter):
        """绘制模板直线"""
        # 左侧长边
        if self.left_line_long:
            x1, y1 = self._scale_point(self.left_line_long[0])
            x2, y2 = self._scale_point(self.left_line_long[1])
            pen = QPen(NeonStyle.NEON_GREEN, 3)
            painter.setPen(pen)
            painter.drawLine(x1, y1, x2, y2)
            # 标注
            mid_x, mid_y = (x1+x2)//2, (y1+y2)//2
            painter.setFont(QFont("Consolas", 8))
            painter.drawText(mid_x+5, mid_y-5, "左长")
        
        # 左侧短边
        if self.left_line_short:
            x1, y1 = self._scale_point(self.left_line_short[0])
            x2, y2 = self._scale_point(self.left_line_short[1])
            pen = QPen(NeonStyle.NEON_GREEN, 3)
            painter.setPen(pen)
            painter.drawLine(x1, y1, x2, y2)
            mid_x, mid_y = (x1+x2)//2, (y1+y2)//2
            painter.drawText(mid_x+5, mid_y-5, "左短")
        
        # 右侧长边
        if self.right_line_long:
            x1, y1 = self._scale_point(self.right_line_long[0])
            x2, y2 = self._scale_point(self.right_line_long[1])
            pen = QPen(NeonStyle.NEON_ORANGE, 3)
            painter.setPen(pen)
            painter.drawLine(x1, y1, x2, y2)
            mid_x, mid_y = (x1+x2)//2, (y1+y2)//2
            painter.drawText(mid_x+5, mid_y-5, "右长")
        
        # 右侧短边
        if self.right_line_short:
            x1, y1 = self._scale_point(self.right_line_short[0])
            x2, y2 = self._scale_point(self.right_line_short[1])
            pen = QPen(NeonStyle.NEON_ORANGE, 3)
            painter.setPen(pen)
            painter.drawLine(x1, y1, x2, y2)
            mid_x, mid_y = (x1+x2)//2, (y1+y2)//2
            painter.drawText(mid_x+5, mid_y-5, "右短")
        
    def _draw_template_contours(self, painter):
        """绘制模板轮廓"""
        # 左轮廓
        if self.left_contour is not None and len(self.left_contour) >= 3:
            self._draw_single_contour(painter, self.left_contour, NeonStyle.NEON_GREEN, "左轮廓")
        
        # 右轮廓
        if self.right_contour is not None and len(self.right_contour) >= 3:
            self._draw_single_contour(painter, self.right_contour, NeonStyle.NEON_ORANGE, "右轮廓")
        
    def _draw_single_contour(self, painter, contour, color, label):
        """绘制单个轮廓"""
        points = [self._scale_point((int(p[0][0]), int(p[0][1]))) for p in contour]
        polygon = QPolygonF([QPointF(x, y) for x, y in points])
        pen = QPen(color, 2, Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 30)))
        painter.drawPolygon(polygon)
        
        # 标注
        if len(points) > 0:
            cx = sum(p[0] for p in points) / len(points)
            cy = sum(p[1] for p in points) / len(points)
            painter.setFont(QFont("Consolas", 8))
            painter.drawText(int(cx)+5, int(cy)-5, label)
        
    def _draw_detection_results(self, painter):
        """绘制检测结果"""
        # 检测到的直线
        for line, color in self.detected_lines:
            x1, y1 = self._scale_point(line[0])
            x2, y2 = self._scale_point(line[1])
            pen = QPen(color, 4)
            painter.setPen(pen)
            painter.drawLine(x1, y1, x2, y2)
        
        # 检测到的轮廓
        for contour, color in self.detected_contours:
            if len(contour) >= 3:
                points = [self._scale_point((int(p[0][0]), int(p[0][1]))) for p in contour]
                polygon = QPolygonF([QPointF(x, y) for x, y in points])
                pen = QPen(color, 2)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                painter.drawPolygon(polygon)
        
        # 检测中心
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
            
            # 角度线
            if abs(self.detected_angle) > 0.1:
                angle_rad = math.radians(self.detected_angle)
                length = 60
                dx = length * math.cos(angle_rad)
                dy = length * math.sin(angle_rad)
                pen = QPen(NeonStyle.NEON_YELLOW, 2)
                painter.setPen(pen)
                painter.drawLine(int(cx-dx), int(cy-dy), int(cx+dx), int(cy+dy))
        
    def _draw_crosshair(self, painter):
        """绘制十字准心"""
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
        """绘制绘制预览"""
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
            elif self.drawing_mode == "contour":
                pen = QPen(NeonStyle.NEON_PURPLE, 2, Qt.DashLine)
                painter.setPen(pen)
                painter.setBrush(QBrush(QColor(150, 0, 255, 50)))
                painter.drawRect(sx, sy, ex-sx, ey-sy)
                
    def _draw_hud_overlay(self, painter):
        """绘制HUD信息"""
        pen = QPen(NeonStyle.NEON_CYAN, 1)
        painter.setPen(pen)
        painter.setFont(QFont("Consolas", 10))
        painter.drawText(15, 25, f"模式: {self.drawing_mode or '观察'}")
        if self.current_frame is not None:
            h, w = self.current_frame.shape[:2]
            painter.drawText(15, 45, f"分辨率: {w}x{h}")
            
    def _draw_corner_decorations(self, painter):
        """绘制角落装饰"""
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
            
    def mouseMoveEvent(self, event):
        """鼠标移动事件"""
        if self.drawing_start:
            self.drawing_end = (event.x(), event.y())
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
                rect = (min(sx, ex), min(sy, ey), max(sx, ex), max(sy, ey))
                self.contour_selected.emit(rect)
            
            self.drawing_start = None
            self.drawing_end = None
            self.drawing_mode = None
            self.update()
            
    def start_drawing(self, mode):
        """开始绘制"""
        self.drawing_mode = mode
        self.setCursor(QCursor(Qt.CrossCursor))
            
    def stop_drawing(self):
        """停止绘制"""
        self.drawing_mode = None
        self.setCursor(QCursor(Qt.ArrowCursor))
    
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
        self.update()


class MainWindow(QMainWindow):
    """
    主窗口类
    
    改进说明：
    1. 双直线模板设置：每侧两条垂直线（长和宽），然后区域设置
    2. 异形模板设置：手动设置轮廓，然后区域设置
    3. 验证逻辑：区域设置成功后才能保存模板
    4. 验证逻辑：获取模板后才能进行手动测试
    5. 完善相机标定：完整的标定流程和原理说明
    """
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("贴内衬机视觉定位系统 - 改进版")
        self.setGeometry(50, 50, 1800, 1000)
        self.setStyleSheet("QMainWindow { background-color: #0A0F1E; }")
        
        # 初始化变量
        self.camera_thread = None
        self.current_mode = "line"  # "line" 或 "shape"
        self.calibration_data = CalibrationData()
        self.dual_line_template = DualLineTemplate()
        self.shape_template = ShapeTemplate()
        
        # 绘制状态跟踪
        self.line_drawing_side = None  # "left_long", "left_short", "right_long", "right_short"
        self.roi_drawing_side = None  # "left", "right"
        self.contour_drawing_side = None  # "left", "right"
        self.contour_roi_drawing_side = None  # "left", "right"
        
        # 标定图像列表
        self.calib_images = []
        
        # 创建UI
        self.image_canvas = ImageCanvas()
        self._setup_ui()
        self._setup_connections()
        
    def _setup_ui(self):
        """设置UI"""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # 左侧面板（摄像头控制 + 标定）
        main_layout.addWidget(self._create_left_panel(), stretch=1)
        
        # 中间面板（图像显示 + 数据）
        main_layout.addWidget(self._create_center_panel(), stretch=3)
        
        # 右侧面板（模板设置）
        main_layout.addWidget(self._create_right_panel(), stretch=1)
        
        # 状态栏
        self._setup_statusbar()
        
    def _create_left_panel(self):
        """创建左侧面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        
        # 摄像头控制组
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
        
        # 相机标定组
        calib_group = QGroupBox("相机标定")
        calib_group.setStyleSheet(self._group_style())
        calib_layout = QVBoxLayout(calib_group)
        
        calib_layout.addWidget(QLabel("标定方法:"))
        self.calib_method = QComboBox()
        self.calib_method.addItems(["棋盘格标定", "点对应标定", "手动输入"])
        self.calib_method.setStyleSheet(self._combo_style())
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
        self.cb_cols_spin.setStyleSheet(self._spinbox_style())
        cb_size_layout.addWidget(self.cb_cols_spin)
        cb_size_layout.addWidget(QLabel("x"))
        self.cb_rows_spin = QSpinBox()
        self.cb_rows_spin.setRange(3, 20)
        self.cb_rows_spin.setValue(6)
        self.cb_rows_spin.setStyleSheet(self._spinbox_style())
        cb_size_layout.addWidget(self.cb_rows_spin)
        cb_layout.addLayout(cb_size_layout)
        
        cb_layout.addWidget(QLabel("方格尺寸(mm):"))
        self.square_size_spin = QDoubleSpinBox()
        self.square_size_spin.setRange(1, 100)
        self.square_size_spin.setValue(25.0)
        self.square_size_spin.setStyleSheet(self._spinbox_style())
        cb_layout.addWidget(self.square_size_spin)
        
        self.btn_capture_calib = QPushButton("采集标定图像")
        self.btn_capture_calib.setStyleSheet(self._button_style())
        self.btn_capture_calib.clicked.connect(self._capture_calib_image)
        cb_layout.addWidget(self.btn_capture_calib)
        
        self.calib_image_count = QLabel("已采集: 0 张")
        self.calib_image_count.setStyleSheet("color: #00FFFF; font: 10px Consolas;")
        cb_layout.addWidget(self.calib_image_count)
        
        self.btn_calibrate = QPushButton("执行标定")
        self.btn_calibrate.setStyleSheet(self._button_style("#00AA66"))
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
        self.calib_table.setStyleSheet(self._table_style())
        self.calib_table.setMaximumHeight(120)
        points_layout.addWidget(self.calib_table)
        
        calib_btn_layout = QHBoxLayout()
        self.btn_add_point = QPushButton("添加点")
        self.btn_add_point.setStyleSheet(self._button_style())
        self.btn_add_point.clicked.connect(self._add_calib_point)
        calib_btn_layout.addWidget(self.btn_add_point)
        
        self.btn_clear_points = QPushButton("清除")
        self.btn_clear_points.setStyleSheet(self._button_style("#FF6666"))
        self.btn_clear_points.clicked.connect(self._clear_calib_points)
        calib_btn_layout.addWidget(self.btn_clear_points)
        
        points_layout.addLayout(calib_btn_layout)
        
        self.btn_calc_calib = QPushButton("计算标定")
        self.btn_calc_calib.setStyleSheet(self._button_style("#00AA66"))
        self.btn_calc_calib.clicked.connect(self._calculate_calibration_from_points)
        points_layout.addWidget(self.btn_calc_calib)
        
        calib_layout.addWidget(self.points_widget)
        
        # 标定结果显示
        self.calib_info = QLabel("状态: 未标定")
        self.calib_info.setStyleSheet("color: #FF6666; font: 10px Consolas;")
        calib_layout.addWidget(self.calib_info)
        
        # 标定原理说明按钮
        self.btn_show_calib_info = QPushButton("查看标定原理")
        self.btn_show_calib_info.setStyleSheet(self._button_style())
        self.btn_show_calib_info.clicked.connect(self._show_calibration_info)
        calib_layout.addWidget(self.btn_show_calib_info)
        
        # 保存/加载标定
        calib_io_layout = QHBoxLayout()
        self.btn_save_calib = QPushButton("保存标定")
        self.btn_save_calib.setStyleSheet(self._button_style())
        self.btn_save_calib.clicked.connect(self._save_calibration)
        calib_io_layout.addWidget(self.btn_save_calib)
        
        self.btn_load_calib = QPushButton("加载标定")
        self.btn_load_calib.setStyleSheet(self._button_style())
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
        layout.setSpacing(10)
        
        # 标题栏
        title_bar = QWidget()
        title_layout = QHBoxLayout(title_bar)
        
        title_label = QLabel("贴内衬机视觉定位系统")
        title_label.setFont(QFont("Consolas", 18, QFont.Bold))
        title_label.setStyleSheet("color: #00FFFF;")
        title_layout.addWidget(title_label)
        
        # 模式切换
        self.mode_switch = QComboBox()
        self.mode_switch.addItems(["双直线模式(矩形)", "双外形模式(异形)"])
        self.mode_switch.setStyleSheet(self._combo_style())
        self.mode_switch.currentIndexChanged.connect(self._change_mode)
        title_layout.addWidget(self.mode_switch)
        
        layout.addWidget(title_bar)
        
        # 图像画布
        self.image_canvas.setStyleSheet("border: 2px solid #0066CC; border-radius: 10px;")
        layout.addWidget(self.image_canvas, stretch=1)
        
        # 实时检测数据面板
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
        """创建右侧面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        
        # 选项卡
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet(self._tab_style())
        
        # 双直线模板选项卡
        self.line_tab = QWidget()
        self._setup_line_template_tab()
        self.tab_widget.addTab(self.line_tab, "双直线模板")
        
        # 异形模板选项卡
        self.shape_tab = QWidget()
        self._setup_shape_template_tab()
        self.tab_widget.addTab(self.shape_tab, "外形模板")
        
        # 贴料参数选项卡
        self.paste_tab = QWidget()
        self._setup_paste_params_tab()
        self.tab_widget.addTab(self.paste_tab, "贴料参数")
        
        layout.addWidget(self.tab_widget)
        layout.addStretch()
        
        return panel
    
    def _setup_line_template_tab(self):
        """设置双直线模板选项卡
        
        流程：
        1. 分别设置左右两侧的长边和短边（两条垂直线）
        2. 设置左右监测区域
        3. 所有设置完成后才能保存模板
        4. 保存模板后才能进行测试
        """
        layout = QVBoxLayout(self.line_tab)
        layout.setSpacing(10)
        
        # 左侧模板组
        left_group = QGroupBox("左侧模板（长边+短边）")
        left_group.setStyleSheet(self._group_style())
        left_layout = QVBoxLayout(left_group)
        
        # 左侧长边
        left_long_layout = QHBoxLayout()
        self.btn_draw_left_long = QPushButton("画长边")
        self.btn_draw_left_long.setStyleSheet(self._button_style("#00AA66"))
        self.btn_draw_left_long.clicked.connect(lambda: self._start_draw_line("left_long"))
        left_long_layout.addWidget(self.btn_draw_left_long)
        
        self.btn_clear_left_long = QPushButton("清除")
        self.btn_clear_left_long.setStyleSheet(self._button_style("#FF6666"))
        self.btn_clear_left_long.clicked.connect(lambda: self._clear_line("left_long"))
        left_long_layout.addWidget(self.btn_clear_left_long)
        
        left_layout.addLayout(left_long_layout)
        
        self.left_long_info = QLabel("长边: 未设置")
        self.left_long_info.setStyleSheet("color: #00FF96; font: 10px Consolas;")
        left_layout.addWidget(self.left_long_info)
        
        # 左侧短边
        left_short_layout = QHBoxLayout()
        self.btn_draw_left_short = QPushButton("画短边")
        self.btn_draw_left_short.setStyleSheet(self._button_style("#00AA66"))
        self.btn_draw_left_short.clicked.connect(lambda: self._start_draw_line("left_short"))
        left_short_layout.addWidget(self.btn_draw_left_short)
        
        self.btn_clear_left_short = QPushButton("清除")
        self.btn_clear_left_short.setStyleSheet(self._button_style("#FF6666"))
        self.btn_clear_left_short.clicked.connect(lambda: self._clear_line("left_short"))
        left_short_layout.addWidget(self.btn_clear_left_short)
        
        left_layout.addLayout(left_short_layout)
        
        self.left_short_info = QLabel("短边: 未设置")
        self.left_short_info.setStyleSheet("color: #00FF96; font: 10px Consolas;")
        left_layout.addWidget(self.left_short_info)
        
        # 左侧监测区域
        self.btn_draw_left_roi = QPushButton("设置左监测区域")
        self.btn_draw_left_roi.setStyleSheet(self._button_style())
        self.btn_draw_left_roi.clicked.connect(lambda: self._start_draw_roi("left"))
        left_layout.addWidget(self.btn_draw_left_roi)
        
        self.left_roi_info = QLabel("区域: 未设置")
        self.left_roi_info.setStyleSheet("color: #00FF96; font: 10px Consolas;")
        left_layout.addWidget(self.left_roi_info)
        
        layout.addWidget(left_group)
        
        # 右侧模板组
        right_group = QGroupBox("右侧模板（长边+短边）")
        right_group.setStyleSheet(self._group_style())
        right_layout = QVBoxLayout(right_group)
        
        # 右侧长边
        right_long_layout = QHBoxLayout()
        self.btn_draw_right_long = QPushButton("画长边")
        self.btn_draw_right_long.setStyleSheet(self._button_style("#00AA66"))
        self.btn_draw_right_long.clicked.connect(lambda: self._start_draw_line("right_long"))
        right_long_layout.addWidget(self.btn_draw_right_long)
        
        self.btn_clear_right_long = QPushButton("清除")
        self.btn_clear_right_long.setStyleSheet(self._button_style("#FF6666"))
        self.btn_clear_right_long.clicked.connect(lambda: self._clear_line("right_long"))
        right_long_layout.addWidget(self.btn_clear_right_long)
        
        right_layout.addLayout(right_long_layout)
        
        self.right_long_info = QLabel("长边: 未设置")
        self.right_long_info.setStyleSheet("color: #FF9600; font: 10px Consolas;")
        right_layout.addWidget(self.right_long_info)
        
        # 右侧短边
        right_short_layout = QHBoxLayout()
        self.btn_draw_right_short = QPushButton("画短边")
        self.btn_draw_right_short.setStyleSheet(self._button_style("#00AA66"))
        self.btn_draw_right_short.clicked.connect(lambda: self._start_draw_line("right_short"))
        right_short_layout.addWidget(self.btn_draw_right_short)
        
        self.btn_clear_right_short = QPushButton("清除")
        self.btn_clear_right_short.setStyleSheet(self._button_style("#FF6666"))
        self.btn_clear_right_short.clicked.connect(lambda: self._clear_line("right_short"))
        right_short_layout.addWidget(self.btn_clear_right_short)
        
        right_layout.addLayout(right_short_layout)
        
        self.right_short_info = QLabel("短边: 未设置")
        self.right_short_info.setStyleSheet("color: #FF9600; font: 10px Consolas;")
        right_layout.addWidget(self.right_short_info)
        
        # 右侧监测区域
        self.btn_draw_right_roi = QPushButton("设置右监测区域")
        self.btn_draw_right_roi.setStyleSheet(self._button_style())
        self.btn_draw_right_roi.clicked.connect(lambda: self._start_draw_roi("right"))
        right_layout.addWidget(self.btn_draw_right_roi)
        
        self.right_roi_info = QLabel("区域: 未设置")
        self.right_roi_info.setStyleSheet("color: #FF9600; font: 10px Consolas;")
        right_layout.addWidget(self.right_roi_info)
        
        layout.addWidget(right_group)
        
        # 操作组
        action_group = QGroupBox("操作")
        action_group.setStyleSheet(self._group_style())
        action_layout = QVBoxLayout(action_group)
        
        self.line_template_status = QLabel("模板状态: 未完成")
        self.line_template_status.setStyleSheet("color: #FF6666; font: bold 10px Consolas;")
        action_layout.addWidget(self.line_template_status)
        
        self.btn_save_line_template = QPushButton("保存当前模板")
        self.btn_save_line_template.setStyleSheet(self._button_style("#0088FF"))
        self.btn_save_line_template.clicked.connect(self._save_line_template)
        self.btn_save_line_template.setEnabled(False)  # 初始禁用
        action_layout.addWidget(self.btn_save_line_template)
        
        self.btn_test_line = QPushButton("手动测试")
        self.btn_test_line.setStyleSheet(self._button_style("#AA00AA"))
        self.btn_test_line.clicked.connect(self._test_line_detection)
        self.btn_test_line.setEnabled(False)  # 初始禁用
        action_layout.addWidget(self.btn_test_line)
        
        # 加载模板
        self.btn_load_line_template = QPushButton("加载模板")
        self.btn_load_line_template.setStyleSheet(self._button_style())
        self.btn_load_line_template.clicked.connect(self._load_line_template)
        action_layout.addWidget(self.btn_load_line_template)
        
        # 清除所有
        self.btn_clear_all_lines = QPushButton("清除所有")
        self.btn_clear_all_lines.setStyleSheet(self._button_style("#FF6666"))
        self.btn_clear_all_lines.clicked.connect(self._clear_all_line_template)
        action_layout.addWidget(self.btn_clear_all_lines)
        
        layout.addWidget(action_group)
        layout.addStretch()
        
    def _setup_shape_template_tab(self):
        """设置异形模板选项卡
        
        流程：
        1. 分别设置左右两侧的轮廓
        2. 设置左右监测区域
        3. 所有设置完成后才能保存模板
        4. 保存模板后才能进行测试
        """
        layout = QVBoxLayout(self.shape_tab)
        layout.setSpacing(10)
        
        # 参数设置组
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
        self.match_threshold_spin.setSingleStep(0.05)
        self.match_threshold_spin.setStyleSheet(self._spinbox_style())
        param_layout.addWidget(self.match_threshold_spin)
        
        layout.addWidget(param_group)
        
        # 左侧轮廓组
        left_group = QGroupBox("左侧轮廓")
        left_group.setStyleSheet(self._group_style())
        left_layout = QVBoxLayout(left_group)
        
        self.btn_draw_left_contour = QPushButton("框选左轮廓")
        self.btn_draw_left_contour.setStyleSheet(self._button_style("#00AA66"))
        self.btn_draw_left_contour.clicked.connect(lambda: self._start_draw_contour("left"))
        left_layout.addWidget(self.btn_draw_left_contour)
        
        self.left_contour_info = QLabel("轮廓: 未设置")
        self.left_contour_info.setStyleSheet("color: #00FF96; font: 10px Consolas;")
        left_layout.addWidget(self.left_contour_info)
        
        self.btn_draw_left_contour_roi = QPushButton("设置左监测区域")
        self.btn_draw_left_contour_roi.setStyleSheet(self._button_style())
        self.btn_draw_left_contour_roi.clicked.connect(lambda: self._start_draw_contour_roi("left"))
        left_layout.addWidget(self.btn_draw_left_contour_roi)
        
        self.left_contour_roi_info = QLabel("区域: 未设置")
        self.left_contour_roi_info.setStyleSheet("color: #00FF96; font: 10px Consolas;")
        left_layout.addWidget(self.left_contour_roi_info)
        
        layout.addWidget(left_group)
        
        # 右侧轮廓组
        right_group = QGroupBox("右侧轮廓")
        right_group.setStyleSheet(self._group_style())
        right_layout = QVBoxLayout(right_group)
        
        self.btn_draw_right_contour = QPushButton("框选右轮廓")
        self.btn_draw_right_contour.setStyleSheet(self._button_style("#00AA66"))
        self.btn_draw_right_contour.clicked.connect(lambda: self._start_draw_contour("right"))
        right_layout.addWidget(self.btn_draw_right_contour)
        
        self.right_contour_info = QLabel("轮廓: 未设置")
        self.right_contour_info.setStyleSheet("color: #FF9600; font: 10px Consolas;")
        right_layout.addWidget(self.right_contour_info)
        
        self.btn_draw_right_contour_roi = QPushButton("设置右监测区域")
        self.btn_draw_right_contour_roi.setStyleSheet(self._button_style())
        self.btn_draw_right_contour_roi.clicked.connect(lambda: self._start_draw_contour_roi("right"))
        right_layout.addWidget(self.btn_draw_right_contour_roi)
        
        self.right_contour_roi_info = QLabel("区域: 未设置")
        self.right_contour_roi_info.setStyleSheet("color: #FF9600; font: 10px Consolas;")
        right_layout.addWidget(self.right_contour_roi_info)
        
        layout.addWidget(right_group)
        
        # 操作组
        action_group = QGroupBox("操作")
        action_group.setStyleSheet(self._group_style())
        action_layout = QVBoxLayout(action_group)
        
        self.shape_template_status = QLabel("模板状态: 未完成")
        self.shape_template_status.setStyleSheet("color: #FF6666; font: bold 10px Consolas;")
        action_layout.addWidget(self.shape_template_status)
        
        self.btn_save_shape_template = QPushButton("保存当前模板")
        self.btn_save_shape_template.setStyleSheet(self._button_style("#0088FF"))
        self.btn_save_shape_template.clicked.connect(self._save_shape_template)
        self.btn_save_shape_template.setEnabled(False)  # 初始禁用
        action_layout.addWidget(self.btn_save_shape_template)
        
        self.btn_test_shape = QPushButton("手动测试")
        self.btn_test_shape.setStyleSheet(self._button_style("#AA00AA"))
        self.btn_test_shape.clicked.connect(self._test_shape_detection)
        self.btn_test_shape.setEnabled(False)  # 初始禁用
        action_layout.addWidget(self.btn_test_shape)
        
        self.btn_load_shape_template = QPushButton("加载模板")
        self.btn_load_shape_template.setStyleSheet(self._button_style())
        self.btn_load_shape_template.clicked.connect(self._load_shape_template)
        action_layout.addWidget(self.btn_load_shape_template)
        
        self.btn_clear_all_shapes = QPushButton("清除所有")
        self.btn_clear_all_shapes.setStyleSheet(self._button_style("#FF6666"))
        self.btn_clear_all_shapes.clicked.connect(self._clear_all_shape_template)
        action_layout.addWidget(self.btn_clear_all_shapes)
        
        layout.addWidget(action_group)
        layout.addStretch()


    def _setup_paste_params_tab(self):
        """设置贴料参数选项卡"""
        layout = QVBoxLayout(self.paste_tab)
        layout.setSpacing(10)
        
        # 偏移补偿组
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
        
        # 阈值设置组
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
        
        # 时序设置组
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
        
        # 失败处理组
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
        
        # 保存/加载按钮
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
        """设置状态栏"""
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
        """设置信号连接"""
        self.image_canvas.roi_selected.connect(self._on_roi_selected)
        self.image_canvas.line_drawn.connect(self._on_line_drawn)
        self.image_canvas.contour_selected.connect(self._on_contour_selected)
        
    # ==================== 样式定义 ====================
    
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
            QPushButton:disabled {{
                background-color: rgba(50, 50, 50, 150);
                color: #666666;
                border-color: #333333;
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
        
        # 检测棋盘格
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
            
            self.calib_info.setText(
                f"状态: 已标定 ✓\n"
                f"重投影误差: {rms_error:.4f} px\n"
                f"焦距: fx={self.calibration_data.camera_matrix[0,0]:.1f}, "
                f"fy={self.calibration_data.camera_matrix[1,1]:.1f}\n"
                f"像素/mm: X={self.calibration_data.pixel_per_mm_x:.2f}, "
                f"Y={self.calibration_data.pixel_per_mm_y:.2f}"
            )
            self.calib_info.setStyleSheet("color: #00FF96; font: 10px Consolas;")
            
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
                physical_points.append((phys, 0))  # 假设Y方向为0
            
            self.calibration_data = CameraCalibration.calibrate_from_points(
                pixel_points, physical_points
            )
            
            self.calib_info.setText(
                f"状态: 已标定 ✓\n"
                f"像素/mm: X={self.calibration_data.pixel_per_mm_x:.4f}, "
                f"Y={self.calibration_data.pixel_per_mm_y:.4f}"
            )
            self.calib_info.setStyleSheet("color: #00FF96; font: 10px Consolas;")
            
            QMessageBox.information(self, "成功", "标定计算完成！")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"计算失败: {str(e)}")
            
    def _show_calibration_info(self):
        """显示标定原理说明"""
        info_text = """
        <h2>相机标定原理与流程</h2>
        
        <h3>一、针孔相机模型</h3>
        <p>相机成像遵循针孔模型，3D空间点通过投影变换映射到2D图像平面。</p>
        
        <h3>二、内参矩阵 (Intrinsic Matrix)</h3>
        <p>K = | fx  0  cx |</p>
        <p>    |  0 fy  cy |</p>
        <p>    |  0  0   1 |</p>
        <ul>
            <li>fx, fy: 焦距（像素单位）</li>
            <li>cx, cy: 主点坐标（图像中心）</li>
        </ul>
        
        <h3>三、畸变模型</h3>
        <p>径向畸变: k1, k2, k3, k4, k5, k6</p>
        <p>切向畸变: p1, p2</p>
        
        <h3>四、标定流程</h3>
        <ol>
            <li>准备棋盘格标定板（推荐9x6内角点）</li>
            <li>从不同角度拍摄至少15-20张图像</li>
            <li>检测每张图像中的角点</li>
            <li>求解相机内参和畸变系数</li>
            <li>计算重投影误差评估精度</li>
        </ol>
        
        <h3>五、精度评估</h3>
        <ul>
            <li>重投影误差 < 0.5px: 优秀</li>
            <li>重投影误差 < 1.0px: 良好</li>
            <li>重投影误差 > 1.0px: 需重新标定</li>
        </ul>
        
        <h3>六、应用场景</h3>
        <ul>
            <li>图像去畸变</li>
            <li>像素坐标转物理坐标</li>
            <li>3D重建</li>
            <li>视觉测量</li>
        </ul>
        """
        
        msg = QMessageBox(self)
        msg.setWindowTitle("相机标定原理说明")
        msg.setTextFormat(Qt.RichText)
        msg.setText(info_text)
        msg.setStyleSheet("""
            QMessageBox {
                background-color: #0A0F1E;
            }
            QMessageBox QLabel {
                color: #00FFFF;
                font: 11px Consolas;
            }
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
            QMessageBox.information(self, "成功", "标定数据已保存")
            
    def _load_calibration(self):
        """加载标定数据"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "加载标定数据", "", "JSON (*.json)"
        )
        if filename:
            self.calibration_data = TemplateManager.load_calibration(filename)
            self.calib_info.setText(
                f"状态: 已加载 ✓\n"
                f"重投影误差: {self.calibration_data.reprojection_error:.4f} px\n"
                f"像素/mm: X={self.calibration_data.pixel_per_mm_x:.2f}, "
                f"Y={self.calibration_data.pixel_per_mm_y:.2f}"
            )
            self.calib_info.setStyleSheet("color: #00FF96; font: 10px Consolas;")
            QMessageBox.information(self, "成功", "标定数据已加载")


    def _change_mode(self, index):
        """切换模式"""
        self.current_mode = "shape" if index == 1 else "line"
        if self.current_mode == "line":
            self.tab_widget.setCurrentIndex(0)
        else:
            self.tab_widget.setCurrentIndex(1)
            
    # ==================== 双直线模板设置 ====================
    
    def _start_draw_line(self, side):
        """开始绘制直线
        
        参数:
            side: "left_long", "left_short", "right_long", "right_short"
        """
        self.line_drawing_side = side
        self.image_canvas.start_drawing("line")
        self.status_label.setText(f"绘制直线: {side}")
        
    def _clear_line(self, side):
        """清除直线"""
        if side == "left_long":
            self.image_canvas.left_line_long = None
            self.left_long_info.setText("长边: 未设置")
        elif side == "left_short":
            self.image_canvas.left_line_short = None
            self.left_short_info.setText("短边: 未设置")
        elif side == "right_long":
            self.image_canvas.right_line_long = None
            self.right_long_info.setText("长边: 未设置")
        elif side == "right_short":
            self.image_canvas.right_line_short = None
            self.right_short_info.setText("短边: 未设置")
        
        self._update_line_template_status()
        self.image_canvas.update()
        
    def _start_draw_roi(self, side):
        """开始绘制ROI区域"""
        self.roi_drawing_side = side
        self.image_canvas.start_drawing("roi")
        self.status_label.setText(f"绘制ROI: {side}")
        
    def _on_line_drawn(self, line_data):
        """直线绘制完成回调"""
        sx, sy, ex, ey = line_data
        line = ((sx, sy), (ex, ey))
        
        if self.line_drawing_side == "left_long":
            self.image_canvas.left_line_long = line
            angle = VisionAlgorithm.calculate_line_angle(line)
            length = math.sqrt((ex-sx)**2 + (ey-sy)**2)
            self.left_long_info.setText(f"长边: θ={angle:.1f}°, L={length:.1f}px")
            
        elif self.line_drawing_side == "left_short":
            self.image_canvas.left_line_short = line
            angle = VisionAlgorithm.calculate_line_angle(line)
            length = math.sqrt((ex-sx)**2 + (ey-sy)**2)
            self.left_short_info.setText(f"短边: θ={angle:.1f}°, L={length:.1f}px")
            
        elif self.line_drawing_side == "right_long":
            self.image_canvas.right_line_long = line
            angle = VisionAlgorithm.calculate_line_angle(line)
            length = math.sqrt((ex-sx)**2 + (ey-sy)**2)
            self.right_long_info.setText(f"长边: θ={angle:.1f}°, L={length:.1f}px")
            
        elif self.line_drawing_side == "right_short":
            self.image_canvas.right_line_short = line
            angle = VisionAlgorithm.calculate_line_angle(line)
            length = math.sqrt((ex-sx)**2 + (ey-sy)**2)
            self.right_short_info.setText(f"短边: θ={angle:.1f}°, L={length:.1f}px")
        
        self.line_drawing_side = None
        self._update_line_template_status()
        self.image_canvas.update()
        
    def _on_roi_selected(self, rect):
        """ROI区域选择完成回调"""
        if self.roi_drawing_side == "left":
            self.image_canvas.left_roi = rect
            self.left_roi_info.setText(f"区域: ({rect[0]:.0f},{rect[1]:.0f})-({rect[2]:.0f},{rect[3]:.0f})")
        elif self.roi_drawing_side == "right":
            self.image_canvas.right_roi = rect
            self.right_roi_info.setText(f"区域: ({rect[0]:.0f},{rect[1]:.0f})-({rect[2]:.0f},{rect[3]:.0f})")
        
        self.roi_drawing_side = None
        self._update_line_template_status()
        self.image_canvas.update()
        
    def _update_line_template_status(self):
        """更新双直线模板状态"""
        # 检查是否所有直线都已设置
        has_all_lines = (
            self.image_canvas.left_line_long is not None and
            self.image_canvas.left_line_short is not None and
            self.image_canvas.right_line_long is not None and
            self.image_canvas.right_line_short is not None
        )
        
        # 检查是否所有区域都已设置
        has_all_rois = (
            self.image_canvas.left_roi is not None and
            self.image_canvas.right_roi is not None
        )
        
        # 更新状态显示
        if has_all_lines and has_all_rois:
            self.line_template_status.setText("模板状态: 已完成 ✓")
            self.line_template_status.setStyleSheet("color: #00FF96; font: bold 10px Consolas;")
            self.btn_save_line_template.setEnabled(True)
        elif has_all_lines:
            self.line_template_status.setText("模板状态: 需设置区域")
            self.line_template_status.setStyleSheet("color: #FF9600; font: bold 10px Consolas;")
            self.btn_save_line_template.setEnabled(False)
        else:
            self.line_template_status.setText("模板状态: 未完成")
            self.line_template_status.setStyleSheet("color: #FF6666; font: bold 10px Consolas;")
            self.btn_save_line_template.setEnabled(False)
        
        # 检查是否可以测试
        self.btn_test_line.setEnabled(self.dual_line_template.valid)
        
    def _save_line_template(self):
        """保存双直线模板"""
        # 验证完整性
        if not self.image_canvas.left_line_long or not self.image_canvas.left_line_short:
            QMessageBox.warning(self, "警告", "请先设置左侧的长边和短边")
            return
        
        if not self.image_canvas.right_line_long or not self.image_canvas.right_line_short:
            QMessageBox.warning(self, "警告", "请先设置右侧的长边和短边")
            return
        
        if not self.image_canvas.left_roi or not self.image_canvas.right_roi:
            QMessageBox.warning(self, "警告", "请先设置左右监测区域")
            return
        
        # 验证垂直性
        left_long_angle = VisionAlgorithm.calculate_line_angle(self.image_canvas.left_line_long)
        left_short_angle = VisionAlgorithm.calculate_line_angle(self.image_canvas.left_line_short)
        angle_diff = abs(left_long_angle - left_short_angle)
        if abs(angle_diff - 90) > 10 and abs(angle_diff - 270) > 10:
            reply = QMessageBox.question(
                self, "警告", 
                f"左侧长边和短边夹角为{angle_diff:.1f}°，不是垂直关系。\n是否继续保存？",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
                return
        
        right_long_angle = VisionAlgorithm.calculate_line_angle(self.image_canvas.right_line_long)
        right_short_angle = VisionAlgorithm.calculate_line_angle(self.image_canvas.right_line_short)
        angle_diff = abs(right_long_angle - right_short_angle)
        if abs(angle_diff - 90) > 10 and abs(angle_diff - 270) > 10:
            reply = QMessageBox.question(
                self, "警告", 
                f"右侧长边和短边夹角为{angle_diff:.1f}°，不是垂直关系。\n是否继续保存？",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
                return
        
        # 保存模板
        self.dual_line_template.left_line_long = self.image_canvas.left_line_long
        self.dual_line_template.left_line_short = self.image_canvas.left_line_short
        self.dual_line_template.right_line_long = self.image_canvas.right_line_long
        self.dual_line_template.right_line_short = self.image_canvas.right_line_short
        self.dual_line_template.left_roi = self.image_canvas.left_roi
        self.dual_line_template.right_roi = self.image_canvas.right_roi
        self.dual_line_template.valid = True
        
        # 启用测试按钮
        self.btn_test_line.setEnabled(True)
        
        # 计算并显示基准数据
        left_intersection = VisionAlgorithm.calculate_line_intersection(
            self.image_canvas.left_line_long, self.image_canvas.left_line_short
        )
        right_intersection = VisionAlgorithm.calculate_line_intersection(
            self.image_canvas.right_line_long, self.image_canvas.right_line_short
        )
        
        if left_intersection and right_intersection:
            cx = (left_intersection[0] + right_intersection[0]) / 2
            cy = (left_intersection[1] + right_intersection[1]) / 2
            avg_angle = (left_long_angle + right_long_angle) / 2
            
            QMessageBox.information(
                self, "保存成功", 
                f"双直线模板已保存！\n\n"
                f"基准数据：\n"
                f"中心: ({cx:.2f}, {cy:.2f}) px\n"
                f"角度: {avg_angle:.2f}°\n\n"
                f"现在可以进行手动测试"
            )
        
    def _load_line_template(self):
        """加载双直线模板"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "加载双直线模板", "", "JSON (*.json)"
        )
        if filename:
            try:
                self.dual_line_template = TemplateManager.load_dual_line_template(filename)
                
                # 更新画布
                self.image_canvas.left_line_long = self.dual_line_template.left_line_long
                self.image_canvas.left_line_short = self.dual_line_template.left_line_short
                self.image_canvas.right_line_long = self.dual_line_template.right_line_long
                self.image_canvas.right_line_short = self.dual_line_template.right_line_short
                self.image_canvas.left_roi = self.dual_line_template.left_roi
                self.image_canvas.right_roi = self.dual_line_template.right_roi
                
                # 更新显示
                if self.dual_line_template.left_line_long:
                    angle = VisionAlgorithm.calculate_line_angle(self.dual_line_template.left_line_long)
                    self.left_long_info.setText(f"长边: θ={angle:.1f}°")
                if self.dual_line_template.left_line_short:
                    angle = VisionAlgorithm.calculate_line_angle(self.dual_line_template.left_line_short)
                    self.left_short_info.setText(f"短边: θ={angle:.1f}°")
                if self.dual_line_template.right_line_long:
                    angle = VisionAlgorithm.calculate_line_angle(self.dual_line_template.right_line_long)
                    self.right_long_info.setText(f"长边: θ={angle:.1f}°")
                if self.dual_line_template.right_line_short:
                    angle = VisionAlgorithm.calculate_line_angle(self.dual_line_template.right_line_short)
                    self.right_short_info.setText(f"短边: θ={angle:.1f}°")
                if self.dual_line_template.left_roi:
                    rect = self.dual_line_template.left_roi
                    self.left_roi_info.setText(f"区域: ({rect[0]:.0f},{rect[1]:.0f})-({rect[2]:.0f},{rect[3]:.0f})")
                if self.dual_line_template.right_roi:
                    rect = self.dual_line_template.right_roi
                    self.right_roi_info.setText(f"区域: ({rect[0]:.0f},{rect[1]:.0f})-({rect[2]:.0f},{rect[3]:.0f})")
                
                self._update_line_template_status()
                self.image_canvas.update()
                
                QMessageBox.information(self, "成功", "双直线模板已加载")
                
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载失败: {str(e)}")
        
    def _clear_all_line_template(self):
        """清除所有双直线模板"""
        self.image_canvas.left_line_long = None
        self.image_canvas.left_line_short = None
        self.image_canvas.right_line_long = None
        self.image_canvas.right_line_short = None
        self.image_canvas.left_roi = None
        self.image_canvas.right_roi = None
        
        self.left_long_info.setText("长边: 未设置")
        self.left_short_info.setText("短边: 未设置")
        self.right_long_info.setText("长边: 未设置")
        self.right_short_info.setText("短边: 未设置")
        self.left_roi_info.setText("区域: 未设置")
        self.right_roi_info.setText("区域: 未设置")
        
        self.dual_line_template = DualLineTemplate()
        self._update_line_template_status()
        self.image_canvas.update()
        
    def _test_line_detection(self):
        """测试双直线检测"""
        if not self.dual_line_template.valid:
            QMessageBox.warning(self, "警告", "请先保存模板")
            return
        
        if self.image_canvas.current_frame is None:
            QMessageBox.warning(self, "警告", "请先打开摄像头")
            return
        
        frame = self.image_canvas.current_frame
        
        # 在左ROI中检测直线
        left_lines = VisionAlgorithm.detect_lines_in_roi(
            frame, self.dual_line_template.left_roi,
            canny_low=50, canny_high=150, hough_threshold=50
        )
        
        # 在右ROI中检测直线
        right_lines = VisionAlgorithm.detect_lines_in_roi(
            frame, self.dual_line_template.right_roi,
            canny_low=50, canny_high=150, hough_threshold=50
        )
        
        # 清空之前的检测结果
        self.image_canvas.detected_lines = []
        
        # 匹配检测到的直线
        if len(left_lines) >= 2:
            self.image_canvas.detected_lines.append((left_lines[0], NeonStyle.NEON_GREEN))
            self.image_canvas.detected_lines.append((left_lines[1], NeonStyle.NEON_GREEN))
        
        if len(right_lines) >= 2:
            self.image_canvas.detected_lines.append((right_lines[0], NeonStyle.NEON_ORANGE))
            self.image_canvas.detected_lines.append((right_lines[1], NeonStyle.NEON_ORANGE))
        
        # 计算中心位置和角度
        if len(left_lines) >= 2 and len(right_lines) >= 2:
            # 计算左侧交点
            left_intersection = VisionAlgorithm.calculate_line_intersection(
                left_lines[0], left_lines[1]
            )
            # 计算右侧交点
            right_intersection = VisionAlgorithm.calculate_line_intersection(
                right_lines[0], right_lines[1]
            )
            
            if left_intersection and right_intersection:
                cx = (left_intersection[0] + right_intersection[0]) / 2
                cy = (left_intersection[1] + right_intersection[1]) / 2
                self.image_canvas.detected_center = (cx, cy)
                
                # 计算角度
                left_angle = VisionAlgorithm.calculate_line_angle(left_lines[0])
                right_angle = VisionAlgorithm.calculate_line_angle(right_lines[0])
                avg_angle = (left_angle + right_angle) / 2
                self.image_canvas.detected_angle = avg_angle
                
                # 更新显示
                self._update_result_display(cx, cy, 0, avg_angle, 0.95)
        
        self.image_canvas.update()
        
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
        
    def _on_contour_selected(self, rect):
        """轮廓选择完成回调"""
        if self.image_canvas.current_frame is None:
            return
        
        x1, y1, x2, y2 = rect
        frame = self.image_canvas.current_frame
        roi = frame[int(y1):int(y2), int(x1):int(x2)]
        
        if roi.size == 0:
            return
        
        # 检测轮廓
        contours = VisionAlgorithm.detect_contours_in_roi(
            frame, rect,
            canny_low=self.canny_low_slider.value(),
            canny_high=self.canny_high_slider.value()
        )
        
        if contours:
            # 选择最大的轮廓
            largest = max(contours, key=cv2.contourArea)
            
            if self.contour_drawing_side == "left":
                self.image_canvas.left_contour = largest
                area = cv2.contourArea(largest)
                self.left_contour_info.setText(f"轮廓: 已设置 (面积={area:.0f})")
            elif self.contour_drawing_side == "right":
                self.image_canvas.right_contour = largest
                area = cv2.contourArea(largest)
                self.right_contour_info.setText(f"轮廓: 已设置 (面积={area:.0f})")
        
        self.contour_drawing_side = None
        self._update_shape_template_status()
        self.image_canvas.update()
        
        # 如果是轮廓ROI绘制
        if self.contour_roi_drawing_side:
            if self.contour_roi_drawing_side == "left":
                self.image_canvas.left_contour_roi = rect
                self.left_contour_roi_info.setText(
                    f"区域: ({rect[0]:.0f},{rect[1]:.0f})-({rect[2]:.0f},{rect[3]:.0f})"
                )
            elif self.contour_roi_drawing_side == "right":
                self.image_canvas.right_contour_roi = rect
                self.right_contour_roi_info.setText(
                    f"区域: ({rect[0]:.0f},{rect[1]:.0f})-({rect[2]:.0f},{rect[3]:.0f})"
                )
            self.contour_roi_drawing_side = None
            self._update_shape_template_status()
            self.image_canvas.update()
        
    def _update_shape_template_status(self):
        """更新异形模板状态"""
        # 检查是否所有轮廓都已设置
        has_all_contours = (
            self.image_canvas.left_contour is not None and
            self.image_canvas.right_contour is not None
        )
        
        # 检查是否所有区域都已设置
        has_all_rois = (
            self.image_canvas.left_contour_roi is not None and
            self.image_canvas.right_contour_roi is not None
        )
        
        # 更新状态显示
        if has_all_contours and has_all_rois:
            self.shape_template_status.setText("模板状态: 已完成 ✓")
            self.shape_template_status.setStyleSheet("color: #00FF96; font: bold 10px Consolas;")
            self.btn_save_shape_template.setEnabled(True)
        elif has_all_contours:
            self.shape_template_status.setText("模板状态: 需设置区域")
            self.shape_template_status.setStyleSheet("color: #FF9600; font: bold 10px Consolas;")
            self.btn_save_shape_template.setEnabled(False)
        else:
            self.shape_template_status.setText("模板状态: 未完成")
            self.shape_template_status.setStyleSheet("color: #FF6666; font: bold 10px Consolas;")
            self.btn_save_shape_template.setEnabled(False)
        
        # 检查是否可以测试
        self.btn_test_shape.setEnabled(self.shape_template.valid)
        
    def _save_shape_template(self):
        """保存异形模板"""
        # 验证完整性
        if self.image_canvas.left_contour is None:
            QMessageBox.warning(self, "警告", "请先设置左侧轮廓")
            return
        
        if self.image_canvas.right_contour is None:
            QMessageBox.warning(self, "警告", "请先设置右侧轮廓")
            return
        
        if self.image_canvas.left_contour_roi is None or self.image_canvas.right_contour_roi is None:
            QMessageBox.warning(self, "警告", "请先设置左右监测区域")
            return
        
        # 保存模板
        self.shape_template.left_contour = self.image_canvas.left_contour
        self.shape_template.right_contour = self.image_canvas.right_contour
        self.shape_template.left_roi = self.image_canvas.left_contour_roi
        self.shape_template.right_roi = self.image_canvas.right_contour_roi
        self.shape_template.valid = True
        
        # 启用测试按钮
        self.btn_test_shape.setEnabled(True)
        
        QMessageBox.information(
            self, "保存成功", 
            "异形模板已保存！\n\n现在可以进行手动测试"
        )
        
    def _load_shape_template(self):
        """加载异形模板"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "加载异形模板", "", "Pickle (*.pkl)"
        )
        if filename:
            try:
                self.shape_template, params = TemplateManager.load_shape_template(filename)
                
                # 更新画布
                self.image_canvas.left_contour = self.shape_template.left_contour
                self.image_canvas.right_contour = self.shape_template.right_contour
                self.image_canvas.left_contour_roi = self.shape_template.left_roi
                self.image_canvas.right_contour_roi = self.shape_template.right_roi
                
                # 更新参数
                if params:
                    self.canny_low_slider.setValue(params.get('canny_low', 50))
                    self.canny_high_slider.setValue(params.get('canny_high', 150))
                    self.match_threshold_spin.setValue(params.get('match_threshold', 0.7))
                
                # 更新显示
                if self.shape_template.left_contour is not None:
                    area = cv2.contourArea(self.shape_template.left_contour)
                    self.left_contour_info.setText(f"轮廓: 已设置 (面积={area:.0f})")
                if self.shape_template.right_contour is not None:
                    area = cv2.contourArea(self.shape_template.right_contour)
                    self.right_contour_info.setText(f"轮廓: 已设置 (面积={area:.0f})")
                if self.shape_template.left_roi:
                    rect = self.shape_template.left_roi
                    self.left_contour_roi_info.setText(
                        f"区域: ({rect[0]:.0f},{rect[1]:.0f})-({rect[2]:.0f},{rect[3]:.0f})"
                    )
                if self.shape_template.right_roi:
                    rect = self.shape_template.right_roi
                    self.right_contour_roi_info.setText(
                        f"区域: ({rect[0]:.0f},{rect[1]:.0f})-({rect[2]:.0f},{rect[3]:.0f})"
                    )
                
                self._update_shape_template_status()
                self.image_canvas.update()
                
                QMessageBox.information(self, "成功", "异形模板已加载")
                
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载失败: {str(e)}")
        
    def _clear_all_shape_template(self):
        """清除所有异形模板"""
        self.image_canvas.left_contour = None
        self.image_canvas.right_contour = None
        self.image_canvas.left_contour_roi = None
        self.image_canvas.right_contour_roi = None
        
        self.left_contour_info.setText("轮廓: 未设置")
        self.right_contour_info.setText("轮廓: 未设置")
        self.left_contour_roi_info.setText("区域: 未设置")
        self.right_contour_roi_info.setText("区域: 未设置")
        
        self.shape_template = ShapeTemplate()
        self._update_shape_template_status()
        self.image_canvas.update()
        
    def _test_shape_detection(self):
        """测试异形检测"""
        if not self.shape_template.valid:
            QMessageBox.warning(self, "警告", "请先保存模板")
            return
        
        if self.image_canvas.current_frame is None:
            QMessageBox.warning(self, "警告", "请先打开摄像头")
            return
        
        frame = self.image_canvas.current_frame
        
        # 在左ROI中检测轮廓
        left_contours = VisionAlgorithm.detect_contours_in_roi(
            frame, self.shape_template.left_roi,
            canny_low=self.canny_low_slider.value(),
            canny_high=self.canny_high_slider.value()
        )
        
        # 在右ROI中检测轮廓
        right_contours = VisionAlgorithm.detect_contours_in_roi(
            frame, self.shape_template.right_roi,
            canny_low=self.canny_low_slider.value(),
            canny_high=self.canny_high_slider.value()
        )
        
        # 清空之前的检测结果
        self.image_canvas.detected_contours = []
        
        # 匹配轮廓
        match_threshold = self.match_threshold_spin.value()
        
        best_left_match = None
        best_left_score = 0
        for contour in left_contours:
            score = VisionAlgorithm.match_shape(
                self.shape_template.left_contour, contour
            )
            if score > match_threshold and score > best_left_score:
                best_left_score = score
                best_left_match = contour
        
        best_right_match = None
        best_right_score = 0
        for contour in right_contours:
            score = VisionAlgorithm.match_shape(
                self.shape_template.right_contour, contour
            )
            if score > match_threshold and score > best_right_score:
                best_right_score = score
                best_right_match = contour
        
        # 添加检测结果
        if best_left_match is not None:
            self.image_canvas.detected_contours.append(
                (best_left_match, NeonStyle.NEON_GREEN)
            )
        
        if best_right_match is not None:
            self.image_canvas.detected_contours.append(
                (best_right_match, NeonStyle.NEON_ORANGE)
            )
        
        # 计算中心位置和角度
        if best_left_match is not None and best_right_match is not None:
            # 计算左轮廓中心
            M_left = cv2.moments(best_left_match)
            if M_left['m00'] > 0:
                cx_left = M_left['m10'] / M_left['m00']
                cy_left = M_left['m01'] / M_left['m00']
            else:
                cx_left, cy_left = 0, 0
            
            # 计算右轮廓中心
            M_right = cv2.moments(best_right_match)
            if M_right['m00'] > 0:
                cx_right = M_right['m10'] / M_right['m00']
                cy_right = M_right['m01'] / M_right['m00']
            else:
                cx_right, cy_right = 0, 0
            
            # 计算整体中心
            cx = (cx_left + cx_right) / 2
            cy = (cy_left + cy_right) / 2
            self.image_canvas.detected_center = (cx, cy)
            
            # 计算角度
            angle = math.degrees(math.atan2(cy_right - cy_left, cx_right - cx_left))
            self.image_canvas.detected_angle = angle
            
            # 更新显示
            avg_score = (best_left_score + best_right_score) / 2
            self._update_result_display(cx, cy, 0, angle, avg_score)
        
        self.image_canvas.update()
        
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
        self.data_status.setStyleSheet("color: #00FF96; font: bold 12px Consolas;")
        
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
            QMessageBox.information(self, "成功", "贴料参数已保存")
            
    def _load_paste_params(self):
        """加载贴料参数"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "加载贴料参数", "", "JSON (*.json)"
        )
        if filename:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.offset_x_spin.setValue(data.get('offset_x', 0))
            self.offset_y_spin.setValue(data.get('offset_y', 0))
            self.offset_z_spin.setValue(data.get('offset_z', 0))
            self.offset_a_spin.setValue(data.get('offset_a', 0))
            self.score_threshold_spin.setValue(data.get('score_threshold', 0.7))
            self.alarm_x_spin.setValue(data.get('alarm_x', 5))
            self.alarm_y_spin.setValue(data.get('alarm_y', 5))
            self.alarm_a_spin.setValue(data.get('alarm_a', 3))
            self.timeout_spin.setValue(data.get('timeout', 3000))
            self.photo_delay_spin.setValue(data.get('photo_delay', 100))
            self.vacuum_delay_spin.setValue(data.get('vacuum_delay', 200))
            self.retry_count_spin.setValue(data.get('retry_count', 3))
            
            QMessageBox.information(self, "成功", "贴料参数已加载")
        
    # ==================== 关闭事件 ====================
    
    def closeEvent(self, event):
        """关闭事件"""
        if self.camera_thread:
            self.camera_thread.stop()
        event.accept()


def main():
    """主函数"""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
