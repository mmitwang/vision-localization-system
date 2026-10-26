#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
贴内衬机视觉定位系统 - GUI主程序
工业级、科技风、HUD风格界面
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
    QFileDialog, QMessageBox, QStatusBar
)
from PyQt5.QtCore import (
    Qt, pyqtSignal, QThread, QRect
)
from PyQt5.QtGui import (
    QPainter, QColor, QPen, QBrush, QFont, QLinearGradient,
    QRadialGradient, QPolygonF, QImage, QCursor
)

from vision_utils import (
    CalibrationData, VisionAlgorithm, TemplateManager, CameraUtils
)


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


class CameraThread(QThread):
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
        self.left_template_contour = None
        self.right_template_contour = None
        self.scan_phase = 0
        self.pulse_phase = 0
        
    def set_frame(self, frame):
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
        self._draw_rois(painter)
        self._draw_lines(painter)
        self._draw_contours(painter)
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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("贴内衬机视觉定位系统")
        self.setGeometry(50, 50, 1600, 1000)
        self.setStyleSheet("QMainWindow { background-color: #0A0F1E; }")
        
        self.camera_thread = None
        self.current_mode = "line"
        self.calibration_data = CalibrationData()
        self.image_canvas = ImageCanvas()
        self._setup_ui()
        self._setup_connections()
        
    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        main_layout.addWidget(self._create_left_panel(), stretch=1)
        main_layout.addWidget(self._create_center_panel(), stretch=3)
        main_layout.addWidget(self._create_right_panel(), stretch=1)
        
        self._setup_statusbar()
        
    def _create_left_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        
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
        
        self.image_canvas.setStyleSheet("border: 2px solid #0066CC; border-radius: 10px;")
        layout.addWidget(self.image_canvas, stretch=1)
        
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
        
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet(self._tab_style())
        
        self.line_tab = QWidget()
        self._setup_line_template_tab()
        self.tab_widget.addTab(self.line_tab, "直线模板")
        
        self.shape_tab = QWidget()
        self._setup_shape_template_tab()
        self.tab_widget.addTab(self.shape_tab, "外形模板")
        
        self.paste_tab = QWidget()
        self._setup_paste_params_tab()
        self.tab_widget.addTab(self.paste_tab, "贴料参数")
        
        layout.addWidget(self.tab_widget)
        layout.addStretch()
        
        return panel
        
    def _setup_line_template_tab(self):
        layout = QVBoxLayout(self.line_tab)
        layout.setSpacing(10)
        
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
        
    def _detect_cameras(self):
        cameras = CameraUtils.detect_cameras()
        self.camera_combo.clear()
        self.camera_combo.addItem("请选择摄像头")
        for idx, w, h in cameras:
            self.camera_combo.addItem(f"摄像头 {idx} ({w}x{h})")
                
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
        
    def _on_camera_error(self, error):
        QMessageBox.critical(self, "摄像头错误", error)
        
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
            self.calib_info.setText(f"状态: 已标定 ✓\n像素/mm: X={self.calibration_data.pixel_per_mm_x:.4f}, Y={self.calibration_data.pixel_per_mm_y:.4f}")
            self.calib_info.setStyleSheet("color: #00FF96; font: 11px Consolas;")
            QMessageBox.information(self, "成功", "标定计算完成！")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"计算失败: {str(e)}")
            
    def _change_mode(self, index):
        self.current_mode = "shape" if index == 1 else "line"
        if self.current_mode == "line":
            self.tab_widget.setCurrentIndex(0)
        else:
            self.tab_widget.setCurrentIndex(1)
            
    def _clear_left_line(self):
        self.image_canvas.left_line = None
        self.left_line_info.setText("状态: 未设置")
        
    def _clear_right_line(self):
        self.image_canvas.right_line = None
        self.right_line_info.setText("状态: 未设置")
        
    def _on_roi_selected(self, rect):
        if self.image_canvas.left_roi is None:
            self.image_canvas.left_roi = rect
            self.left_line_info.setText(f"左监测区: ({rect[0]:.0f}, {rect[1]:.0f})-({rect[2]:.0f}, {rect[3]:.0f})")
        else:
            self.image_canvas.right_roi = rect
            self.right_line_info.setText(f"右监测区: ({rect[0]:.0f}, {rect[1]:.0f})-({rect[2]:.0f}, {rect[3]:.0f})")

    def _on_line_drawn(self, line_data):
        sx, sy, ex, ey = line_data
        if self.image_canvas.left_line is None:
            self.image_canvas.left_line = ((sx, sy), (ex, ey))
            theta = math.degrees(math.atan2(ey-sy, ex-sx))
            self.left_line_info.setText(f"左模板: θ={theta:.1f}°")
        else:
            self.image_canvas.right_line = ((sx, sy), (ex, ey))
            theta = math.degrees(math.atan2(ey-sy, ex-sx))
            self.right_line_info.setText(f"右模板: θ={theta:.1f}°")

    def _save_base_template(self):
        if not self.image_canvas.left_line or not self.image_canvas.right_line:
            QMessageBox.warning(self, "警告", "请先设置左右模板")
            return
        l1_start, l1_end = self.image_canvas.left_line
        l2_start, l2_end = self.image_canvas.right_line
        cx = (l1_start[0] + l1_end[0] + l2_start[0] + l2_end[0]) / 4
        cy = (l1_start[1] + l1_end[1] + l2_start[1] + l2_end[1]) / 4
        theta1 = math.degrees(math.atan2(l1_end[1]-l1_start[1], l1_end[0]-l1_start[0]))
        theta2 = math.degrees(math.atan2(l2_end[1]-l2_start[1], l2_end[0]-l2_start[0]))
        angle = (theta1 + theta2) / 2
        QMessageBox.information(self, "成功", "A面值录入成功！\n" + f"X: {cx:.2f}, Y: {cy:.2f}, Z: 0.00, A: {angle:.2f}°")

    def _test_line_detection(self):
        if self.image_canvas.current_frame is None:
            QMessageBox.warning(self, "警告", "请先打开摄像头")
            return
        lines = VisionAlgorithm.detect_lines(self.image_canvas.current_frame)
        if len(lines) >= 2:
            self.image_canvas.detected_left_line = lines[0]
            self.image_canvas.detected_right_line = lines[1]
            cx, cy = VisionAlgorithm.calculate_center(lines[:2])
            angle = VisionAlgorithm.calculate_angle(lines[:2])
            self._update_result_display(cx, cy, 0, angle, 0.95)
            self.image_canvas.update()

    def _save_line_template(self):
        filename, _ = QFileDialog.getSaveFileName(self, "保存直线模板", "", "JSON (*.json)")
        if filename:
            TemplateManager.save_line_template(
                filename,
                self.image_canvas.left_line,
                self.image_canvas.right_line,
                self.image_canvas.left_roi,
                self.image_canvas.right_roi
            )
            QMessageBox.information(self, "成功", "直线模板已保存")

    def _load_line_template(self):
        filename, _ = QFileDialog.getOpenFileName(self, "加载直线模板", "", "JSON (*.json)")
        if filename:
            data = TemplateManager.load_line_template(filename)
            if data.get('left_line'):
                self.image_canvas.left_line = (tuple(data['left_line']['start']), tuple(data['left_line']['end']))
            if data.get('right_line'):
                self.image_canvas.right_line = (tuple(data['right_line']['start']), tuple(data['right_line']['end']))
            self.image_canvas.left_roi = tuple(data.get('left_roi', ())) if data.get('left_roi') else None
            self.image_canvas.right_roi = tuple(data.get('right_roi', ())) if data.get('right_roi') else None
            self.image_canvas.update()
            QMessageBox.information(self, "成功", "直线模板已加载")

    def _clear_left_shape(self):
        self.image_canvas.left_template_contour = None
        self.left_shape_info.setText("状态: 未设置")

    def _clear_right_shape(self):
        self.image_canvas.right_template_contour = None
        self.right_shape_info.setText("状态: 未设置")

    def _on_template_selected(self, rect):
        if self.image_canvas.current_frame is None:
            return
        x1, y1, x2, y2 = rect
        frame = self.image_canvas.current_frame
        roi = frame[int(y1):int(y2), int(x1):int(x2)]
        if roi.size == 0:
            return
        contours = VisionAlgorithm.detect_contours(
            roi,
            self.canny_low_slider.value(),
            self.canny_high_slider.value()
        )
        if contours:
            largest = max(contours, key=cv2.contourArea)
            adjusted = largest.copy()
            adjusted[:, 0, 0] += int(x1)
            adjusted[:, 0, 1] += int(y1)
            if self.image_canvas.left_template_contour is None:
                self.image_canvas.left_template_contour = adjusted
                self.left_shape_info.setText(f"状态: 已设置\n面积: {cv2.contourArea(largest):.0f}")
            else:
                self.image_canvas.right_template_contour = adjusted
                self.right_shape_info.setText(f"状态: 已设置\n面积: {cv2.contourArea(largest):.0f}")
            self.image_canvas.update()

    def _test_shape_detection(self):
        if self.image_canvas.current_frame is None:
            QMessageBox.warning(self, "警告", "请先打开摄像头")
            return
        contours = VisionAlgorithm.detect_contours(
            self.image_canvas.current_frame,
            self.canny_low_slider.value(),
            self.canny_high_slider.value()
        )
        detected = []
        if self.image_canvas.left_template_contour is not None:
            for contour in contours:
                if len(contour) < 3:
                    continue
                match_score = VisionAlgorithm.match_shape(self.image_canvas.left_template_contour, contour)
                if match_score > self.match_threshold_spin.value():
                    detected.append((contour, NeonStyle.NEON_GREEN))
        if self.image_canvas.right_template_contour is not None:
            for contour in contours:
                if len(contour) < 3:
                    continue
                match_score = VisionAlgorithm.match_shape(self.image_canvas.right_template_contour, contour)
                if match_score > self.match_threshold_spin.value():
                    detected.append((contour, NeonStyle.NEON_ORANGE))
        self.image_canvas.detected_contours = detected
        if len(detected) >= 2:
            all_points = []
            for contour, _ in detected[:2]:
                for p in contour:
                    all_points.append(p[0])
            if all_points:
                all_points = np.array(all_points)
                cx = np.mean(all_points[:, 0])
                cy = np.mean(all_points[:, 1])
                self.image_canvas.detected_center = (cx, cy)
                dx = all_points[-1][0] - all_points[0][0]
                dy = all_points[-1][1] - all_points[0][1]
                angle = math.degrees(math.atan2(dy, dx))
                self.image_canvas.detected_angle = angle
                self._update_result_display(cx, cy, 0, angle, 0.9)
        self.image_canvas.update()

    def _save_shape_template(self):
        filename, _ = QFileDialog.getSaveFileName(self, "保存异形模板", "", "Pickle (*.pkl)")
        if filename:
            TemplateManager.save_shape_template(
                filename,
                self.image_canvas.left_template_contour,
                self.image_canvas.right_template_contour,
                self.canny_low_slider.value(),
                self.canny_high_slider.value(),
                self.match_threshold_spin.value()
            )
            QMessageBox.information(self, "成功", "异形模板已保存")

    def _save_paste_params(self):
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
        filename, _ = QFileDialog.getSaveFileName(self, "保存贴料参数", "", "JSON (*.json)")
        if filename:
            TemplateManager.save_paste_params(filename, params)
            QMessageBox.information(self, "成功", "贴料参数已保存")

    def _load_paste_params(self):
        filename, _ = QFileDialog.getOpenFileName(self, "加载贴料参数", "", "JSON (*.json)")
        if filename:
            data = TemplateManager.load_paste_params(filename)
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

    def _update_result_display(self, x_pixel, y_pixel, z_mm, angle, score):
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

    def closeEvent(self, event):
        if self.camera_thread:
            self.camera_thread.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
