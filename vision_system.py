#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
内衬视觉定位系统 - Inner Liner Vision Positioning System
基于AUTO20.BAS流程开发的未来科技风HUD界面
支持双直线边缘识别与双外形模板识别
"""

import sys
import math
import time
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QFrame, QGridLayout,
                             QProgressBar, QSlider, QComboBox, QGroupBox, QCheckBox,
                             QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QPropertyAnimation, QRectF
from PyQt5.QtGui import (QPainter, QColor, QPen, QBrush, QFont, QLinearGradient,
                         QRadialGradient, QPainterPath, QPolygonF, QImage, QPixmap,
                         QConicalGradient)


@dataclass
class VisionResult:
    x: float = 0.0
    y: float = 0.0
    angle: float = 0.0
    score: float = 0.0
    valid: bool = False


@dataclass
class EdgeLine:
    rho: float = 0.0
    theta: float = 0.0
    start: Tuple[float, float] = (0, 0)
    end: Tuple[float, float] = (0, 0)
    score: float = 0.0


class NeonStyle:
    NEON_BLUE = QColor(0, 150, 255)
    NEON_CYAN = QColor(0, 255, 255)
    NEON_PURPLE = QColor(150, 0, 255)
    NEON_GREEN = QColor(0, 255, 150)
    NEON_RED = QColor(255, 50, 100)
    NEON_ORANGE = QColor(255, 150, 0)
    DARK_BG = QColor(5, 10, 25)
    PANEL_BG = QColor(10, 20, 40)
    GRID_COLOR = QColor(20, 40, 80)
    TEXT_COLOR = QColor(0, 200, 255)


class GlowingWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.glow_intensity = 0.5
        self.glow_phase = 0
        self.glow_timer = QTimer(self)
        self.glow_timer.timeout.connect(self._update_glow)
        self.glow_timer.start(50)
        
    def _update_glow(self):
        self.glow_phase += 0.1
        self.glow_intensity = 0.5 + 0.3 * math.sin(self.glow_phase)
        self.update()


class HUDButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setFont(QFont("Consolas", 10, QFont.Bold))
        self.setStyleSheet("""
            QPushButton {
                background-color: rgba(0, 50, 100, 150);
                color: #00FFFF;
                border: 2px solid #0088FF;
                border-radius: 5px;
                padding: 8px 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(0, 100, 200, 200);
                border-color: #00FFFF;
                box-shadow: 0 0 20px #00FFFF;
            }
            QPushButton:pressed {
                background-color: rgba(0, 150, 255, 255);
            }
        """)


class HUDLabel(QLabel):
    def __init__(self, text="", parent=None, color=NeonStyle.NEON_CYAN):
        super().__init__(text, parent)
        self.neon_color = color
        self.setFont(QFont("Consolas", 11))
        self.setStyleSheet(f"""
            QLabel {{
                color: rgb({color.red()}, {color.green()}, {color.blue()});
                background: transparent;
            }}
        """)


class HUDPanel(QFrame):
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.title = title
        self.setFrameStyle(QFrame.Box | QFrame.Raised)
        self.setStyleSheet("""
            QFrame {
                background-color: rgba(10, 20, 40, 200);
                border: 2px solid #0066CC;
                border-radius: 10px;
            }
        """)
        
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        pen = QPen(NeonStyle.NEON_BLUE, 2)
        painter.setPen(pen)
        
        gradient = QLinearGradient(0, 0, self.width(), 0)
        gradient.setColorAt(0, QColor(0, 100, 255, 100))
        gradient.setColorAt(0.5, QColor(0, 200, 255, 150))
        gradient.setColorAt(1, QColor(0, 100, 255, 100))
        painter.setBrush(QBrush(gradient))
        
        painter.drawRoundedRect(5, 5, self.width()-10, 25, 5, 5)
        
        if self.title:
            painter.setPen(QPen(NeonStyle.NEON_CYAN, 1))
            painter.setFont(QFont("Consolas", 10, QFont.Bold))
            painter.drawText(15, 22, self.title)


class EdgeDetector:
    def __init__(self):
        self.canny_low = 50
        self.canny_high = 150
        self.hough_threshold = 50
        self.min_line_length = 30
        self.max_line_gap = 10
        
    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        return edges
    
    def detect_lines(self, edges: np.ndarray, image_shape: Tuple[int, int]) -> List[EdgeLine]:
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, self.hough_threshold,
                                self.min_line_length, self.max_line_gap)
        
        edge_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
                if length > self.min_line_length:
                    theta = math.atan2(y2-y1, x2-x1)
                    rho = x1 * math.cos(theta) + y1 * math.sin(theta)
                    edge_lines.append(EdgeLine(
                        rho=rho, theta=theta,
                        start=(x1, y1), end=(x2, y2),
                        score=length
                    ))
        
        edge_lines.sort(key=lambda x: x.score, reverse=True)
        return edge_lines
    
    def find_parallel_edges(self, lines: List[EdgeLine], angle_tolerance: float = 0.1) -> List[Tuple[EdgeLine, EdgeLine]]:
        parallel_pairs = []
        used = set()
        
        for i, line1 in enumerate(lines):
            if i in used:
                continue
            for j, line2 in enumerate(lines[i+1:], i+1):
                if j in used:
                    continue
                
                angle_diff = abs(line1.theta - line2.theta)
                if angle_diff < angle_tolerance or abs(angle_diff - np.pi) < angle_tolerance:
                    parallel_pairs.append((line1, line2))
                    used.add(i)
                    used.add(j)
                    break
        
        return parallel_pairs


class TemplateMatcher:
    def __init__(self):
        self.templates = []
        self.match_threshold = 0.7
        self.scale_range = (0.8, 1.2)
        self.angle_range = (-15, 15)
        
    def add_template(self, template: np.ndarray, name: str = ""):
        if len(template.shape) == 3:
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        self.templates.append({'image': template, 'name': name})
        
    def create_default_template(self, width: int = 100, height: int = 80) -> np.ndarray:
        template = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(template, (10, 10), (width-10, height-10), 255, 2)
        cv2.line(template, (10, 10), (width-10, height-10), 255, 1)
        cv2.line(template, (width-10, 10), (10, height-10), 255, 1)
        return template
    
    def match_template(self, image: np.ndarray, template_idx: int = 0) -> VisionResult:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        if template_idx >= len(self.templates):
            return VisionResult(valid=False)
            
        template = self.templates[template_idx]['image']
        th, tw = template.shape[:2]
        
        best_result = VisionResult(valid=False)
        
        for scale in np.linspace(self.scale_range[0], self.scale_range[1], 5):
            scaled_template = cv2.resize(template, None, fx=scale, fy=scale)
            sth, stw = scaled_template.shape[:2]
            
            if sth >= gray.shape[0] or stw >= gray.shape[1]:
                continue
                
            result = cv2.matchTemplate(gray, scaled_template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            if max_val > best_result.score and max_val > self.match_threshold:
                best_result = VisionResult(
                    x=max_loc[0] + stw/2,
                    y=max_loc[1] + sth/2,
                    angle=0,
                    score=max_val,
                    valid=True
                )
                
        return best_result
    
    def multi_scale_match(self, image: np.ndarray) -> List[VisionResult]:
        results = []
        for i in range(len(self.templates)):
            result = self.match_template(image, i)
            if result.valid:
                results.append(result)
        return results


class VisionProcessor(QThread):
    result_ready = pyqtSignal(VisionResult, list, list)
    frame_ready = pyqtSignal(np.ndarray)
    
    def __init__(self):
        super().__init__()
        self.edge_detector = EdgeDetector()
        self.template_matcher = TemplateMatcher()
        self.running = False
        self.paused = False
        self.current_frame = None
        self.mode = "edge"
        
        self.template_matcher.add_template(
            self.template_matcher.create_default_template(80, 60), "Template1"
        )
        self.template_matcher.add_template(
            self.template_matcher.create_default_template(100, 70), "Template2"
        )
        
    def run(self):
        self.running = True
        while self.running:
            if not self.paused and self.current_frame is not None:
                self._process_frame()
            time.sleep(0.033)
            
    def _process_frame(self):
        frame = self.current_frame.copy()
        
        edges = self.edge_detector.detect_edges(frame)
        lines = self.edge_detector.detect_lines(edges, frame.shape[:2])
        parallel_pairs = self.edge_detector.find_parallel_edges(lines)
        
        if self.mode == "template":
            template_results = self.template_matcher.multi_scale_match(frame)
            if template_results:
                result = max(template_results, key=lambda x: x.score)
            else:
                result = VisionResult(valid=False)
        else:
            if parallel_pairs:
                line1, line2 = parallel_pairs[0]
                cx = (line1.start[0] + line1.end[0] + line2.start[0] + line2.end[0]) / 4
                cy = (line1.start[1] + line1.end[1] + line2.start[1] + line2.end[1]) / 4
                angle = math.degrees((line1.theta + line2.theta) / 2)
                result = VisionResult(x=cx, y=cy, angle=angle, score=1.0, valid=True)
            else:
                result = VisionResult(valid=False)
        
        self.result_ready.emit(result, lines, parallel_pairs)
        self.frame_ready.emit(frame)
        
    def set_frame(self, frame: np.ndarray):
        self.current_frame = frame
        
    def stop(self):
        self.running = False
        self.wait()


class VisionCanvas(GlowingWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.current_frame = None
        self.vision_result = VisionResult()
        self.detected_lines: List[EdgeLine] = []
        self.parallel_pairs: List[Tuple[EdgeLine, EdgeLine]] = []
        self.show_crosshair = True
        self.show_target_box = True
        self.show_vectors = True
        self.show_contours = True
        self.rotation_angle = 0
        self.scan_line_pos = 0
        self.scan_direction = 1
        self.pulse_phase = 0
        self.grid_enabled = True
        
    def set_frame(self, frame: np.ndarray):
        self.current_frame = frame
        self.update()
        
    def set_result(self, result: VisionResult, lines: List[EdgeLine], 
                   pairs: List[Tuple[EdgeLine, EdgeLine]]):
        self.vision_result = result
        self.detected_lines = lines
        self.parallel_pairs = pairs
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        
        self._draw_background(painter)
        
        if self.current_frame is not None:
            self._draw_image(painter)
        else:
            self._draw_placeholder(painter)
            
        self._draw_scan_effect(painter)
        
        if self.detected_lines:
            self._draw_detected_lines(painter)
            
        if self.parallel_pairs:
            self._draw_parallel_edges(painter)
            
        if self.vision_result.valid:
            if self.show_target_box:
                self._draw_target_box(painter)
            if self.show_vectors:
                self._draw_rotation_vectors(painter)
                
        if self.show_crosshair:
            self._draw_crosshair(painter)
            
        self._draw_hud_overlay(painter)
        self._draw_corner_decorations(painter)
        
        self.scan_line_pos += self.scan_direction * 3
        if self.scan_line_pos > self.width() or self.scan_line_pos < 0:
            self.scan_direction *= -1
            
        self.pulse_phase += 0.15
        
    def _draw_background(self, painter: QPainter):
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0, QColor(5, 10, 30))
        gradient.setColorAt(0.5, QColor(10, 20, 50))
        gradient.setColorAt(1, QColor(5, 15, 35))
        painter.fillRect(self.rect(), gradient)
        
        if self.grid_enabled:
            pen = QPen(QColor(20, 40, 80, 100), 1, Qt.DotLine)
            painter.setPen(pen)
            grid_size = 40
            for x in range(0, self.width(), grid_size):
                painter.drawLine(x, 0, x, self.height())
            for y in range(0, self.height(), grid_size):
                painter.drawLine(0, y, self.width(), y)
                
    def _draw_image(self, painter: QPainter):
        h, w = self.current_frame.shape[:2]
        if len(self.current_frame.shape) == 2:
            q_image = QImage(self.current_frame.data, w, h, w, QImage.Format_Grayscale8)
        else:
            q_image = QImage(self.current_frame.data, w, h, 3*w, QImage.Format_BGR888)
        
        pixmap = QPixmap.fromImage(q_image)
        scaled = pixmap.scaled(self.width(), self.height(), Qt.KeepAspectRatio)
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        painter.drawPixmap(x, y, scaled)
        
    def _draw_placeholder(self, painter: QPainter):
        cx, cy = self.width()//2, self.height()//2
        
        gradient = QRadialGradient(cx, cy, 200)
        gradient.setColorAt(0, QColor(0, 100, 200, 50))
        gradient.setColorAt(0.5, QColor(0, 50, 150, 30))
        gradient.setColorAt(1, QColor(0, 20, 80, 10))
        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(cx-200, cy-200, 400, 400)
        
        pen = QPen(NeonStyle.NEON_BLUE, 2)
        painter.setPen(pen)
        painter.setFont(QFont("Consolas", 14))
        painter.drawText(self.rect(), Qt.AlignCenter, "等待图像输入...")
        
        pen.setStyle(Qt.DashLine)
        pen.setColor(QColor(0, 150, 255, 100))
        painter.setPen(pen)
        painter.drawRect(50, 50, self.width()-100, self.height()-100)
        
    def _draw_scan_effect(self, painter: QPainter):
        gradient = QLinearGradient(self.scan_line_pos-50, 0, self.scan_line_pos+50, 0)
        gradient.setColorAt(0, QColor(0, 255, 255, 0))
        gradient.setColorAt(0.5, QColor(0, 255, 255, 60))
        gradient.setColorAt(1, QColor(0, 255, 255, 0))
        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.NoPen)
        painter.drawRect(self.scan_line_pos-50, 0, 100, self.height())
        
    def _draw_crosshair(self, painter: QPainter):
        cx, cy = self.width()//2, self.height()//2
        size = 30
        gap = 10
        
        pulse = int(5 * math.sin(self.pulse_phase))
        
        pen = QPen(NeonStyle.NEON_CYAN, 2)
        painter.setPen(pen)
        
        painter.drawLine(cx - size - pulse, cy, cx - gap, cy)
        painter.drawLine(cx + gap, cy, cx + size + pulse, cy)
        painter.drawLine(cx, cy - size - pulse, cx, cy - gap)
        painter.drawLine(cx, cy + gap, cx, cy + size + pulse)
        
        painter.drawEllipse(cx - gap, cy - gap, gap*2, gap*2)
        
        pen.setWidth(1)
        pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        painter.drawLine(0, cy, cx - size - 20, cy)
        painter.drawLine(cx + size + 20, cy, self.width(), cy)
        painter.drawLine(cx, 0, cx, cy - size - 20)
        painter.drawLine(cx, cy + size + 20, cx, self.height())
        
    def _draw_target_box(self, painter: QPainter):
        x = int(self.vision_result.x)
        y = int(self.vision_result.y)
        angle = self.vision_result.angle
        
        box_w, box_h = 120, 80
        
        painter.save()
        painter.translate(x, y)
        painter.rotate(angle)
        
        pen = QPen(NeonStyle.NEON_GREEN, 3)
        painter.setPen(pen)
        
        corners = [
            (-box_w//2, -box_h//2),
            (box_w//2, -box_h//2),
            (box_w//2, box_h//2),
            (-box_w//2, box_h//2)
        ]
        
        corner_len = 20
        for i, (cx, cy) in enumerate(corners):
            next_c = corners[(i+1) % 4]
            dx = 1 if next_c[0] > cx else (-1 if next_c[0] < cx else 0)
            dy = 1 if next_c[1] > cy else (-1 if next_c[1] < cy else 0)
            
            painter.drawLine(cx, cy, cx + dx*corner_len, cy)
            painter.drawLine(cx, cy, cx, cy + dy*corner_len)
        
        pen.setColor(NeonStyle.NEON_GREEN)
        pen.setWidth(1)
        pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        painter.drawRect(-box_w//2, -box_h//2, box_w, box_h)
        
        painter.restore()
        
        gradient = QRadialGradient(x, y, 60)
        gradient.setColorAt(0, QColor(0, 255, 150, 40))
        gradient.setColorAt(0.5, QColor(0, 255, 150, 20))
        gradient.setColorAt(1, QColor(0, 255, 150, 0))
        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(x-60, y-60, 120, 120)
        
    def _draw_rotation_vectors(self, painter: QPainter):
        x = int(self.vision_result.x)
        y = int(self.vision_result.y)
        angle = math.radians(self.vision_result.angle)
        
        length = 80
        
        pen = QPen(NeonStyle.NEON_ORANGE, 3)
        painter.setPen(pen)
        
        dx = length * math.cos(angle)
        dy = length * math.sin(angle)
        painter.drawLine(x, y, int(x + dx), int(y + dy))
        
        arrow_size = 15
        arrow_angle = math.pi / 6
        ax = x + dx
        ay = y + dy
        painter.drawLine(int(ax), int(ay), 
                        int(ax - arrow_size*math.cos(angle - arrow_angle)),
                        int(ay - arrow_size*math.sin(angle - arrow_angle)))
        painter.drawLine(int(ax), int(ay),
                        int(ax - arrow_size*math.cos(angle + arrow_angle)),
                        int(ay - arrow_size*math.sin(angle + arrow_angle)))
        
        pen.setColor(NeonStyle.NEON_PURPLE)
        painter.setPen(pen)
        perp_angle = angle + math.pi/2
        pdx = 50 * math.cos(perp_angle)
        pdy = 50 * math.sin(perp_angle)
        painter.drawLine(int(x - pdx), int(y - pdy), int(x + pdx), int(y + pdy))
        
    def _draw_detected_lines(self, painter: QPainter):
        pen = QPen(NeonStyle.NEON_BLUE, 2)
        painter.setPen(pen)
        
        for line in self.detected_lines[:20]:
            painter.drawLine(int(line.start[0]), int(line.start[1]),
                           int(line.end[0]), int(line.end[1]))
            
    def _draw_parallel_edges(self, painter: QPainter):
        colors = [NeonStyle.NEON_GREEN, NeonStyle.NEON_ORANGE]
        
        for i, (line1, line2) in enumerate(self.parallel_pairs[:2]):
            color = colors[i % 2]
            
            pen = QPen(color, 4)
            painter.setPen(pen)
            
            painter.drawLine(int(line1.start[0]), int(line1.start[1]),
                           int(line1.end[0]), int(line1.end[1]))
            painter.drawLine(int(line2.start[0]), int(line2.start[1]),
                           int(line2.end[0]), int(line2.end[1]))
            
            mid_x = (line1.start[0] + line1.end[0] + line2.start[0] + line2.end[0]) / 4
            mid_y = (line1.start[1] + line1.end[1] + line2.start[1] + line2.end[1]) / 4
            
            gradient = QRadialGradient(mid_x, mid_y, 30)
            gradient.setColorAt(0, QColor(color.red(), color.green(), color.blue(), 100))
            gradient.setColorAt(1, QColor(color.red(), color.green(), color.blue(), 0))
            painter.setBrush(QBrush(gradient))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(int(mid_x-30), int(mid_y-30), 60, 60)
            
    def _draw_hud_overlay(self, painter: QPainter):
        pen = QPen(NeonStyle.NEON_CYAN, 1)
        painter.setPen(pen)
        painter.setFont(QFont("Consolas", 9))
        
        info_lines = [
            f"VISION SYSTEM v2.0",
            f"MODE: {'EDGE' if not self.vision_result.valid else 'LOCK'}",
            f"X: {self.vision_result.x:.2f}",
            f"Y: {self.vision_result.y:.2f}",
            f"θ: {self.vision_result.angle:.2f}°",
            f"SCORE: {self.vision_result.score:.3f}",
        ]
        
        y_offset = 30
        for line in info_lines:
            painter.drawText(15, y_offset, line)
            y_offset += 18
            
        painter.drawText(self.width() - 150, 30, f"LINES: {len(self.detected_lines)}")
        painter.drawText(self.width() - 150, 48, f"PAIRS: {len(self.parallel_pairs)}")
        
        pen.setStyle(Qt.DotLine)
        painter.setPen(pen)
        painter.drawLine(10, self.height()-40, self.width()-10, self.height()-40)
        
        progress = (self.scan_line_pos / self.width()) if self.width() > 0 else 0
        pen.setStyle(Qt.SolidLine)
        pen.setColor(NeonStyle.NEON_GREEN)
        painter.setPen(pen)
        bar_width = int((self.width()-20) * progress)
        painter.drawLine(10, self.height()-30, 10 + bar_width, self.height()-30)
        
    def _draw_corner_decorations(self, painter: QPainter):
        pen = QPen(NeonStyle.NEON_BLUE, 3)
        painter.setPen(pen)
        
        corner_size = 40
        
        painter.drawLine(0, corner_size, 0, 0)
        painter.drawLine(0, 0, corner_size, 0)
        
        painter.drawLine(self.width()-corner_size, 0, self.width(), 0)
        painter.drawLine(self.width(), 0, self.width(), corner_size)
        
        painter.drawLine(0, self.height()-corner_size, 0, self.height())
        painter.drawLine(0, self.height(), corner_size, self.height())
        
        painter.drawLine(self.width()-corner_size, self.height(), self.width(), self.height())
        painter.drawLine(self.width(), self.height()-corner_size, self.width(), self.height())


class StatusIndicator(QWidget):
    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self.label = label
        self.active = False
        self.color = NeonStyle.NEON_RED
        self.setMinimumSize(120, 40)
        
    def set_active(self, active: bool, color: QColor = None):
        self.active = active
        if color:
            self.color = color
        else:
            self.color = NeonStyle.NEON_GREEN if active else NeonStyle.NEON_RED
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        painter.setPen(QPen(NeonStyle.NEON_BLUE, 1))
        painter.setBrush(QBrush(QColor(10, 20, 40, 200)))
        painter.drawRoundedRect(0, 0, self.width()-1, self.height()-1, 5, 5)
        
        indicator_color = self.color if self.active else NeonStyle.NEON_RED
        gradient = QRadialGradient(20, self.height()//2, 10)
        gradient.setColorAt(0, indicator_color)
        gradient.setColorAt(1, QColor(indicator_color.red()//3, 
                                      indicator_color.green()//3,
                                      indicator_color.blue()//3))
        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(10, self.height()//2-8, 16, 16)
        
        painter.setPen(QPen(NeonStyle.NEON_CYAN, 1))
        painter.setFont(QFont("Consolas", 9))
        painter.drawText(35, self.height()//2 + 5, self.label)


class DataPanel(HUDPanel):
    def __init__(self, title="", parent=None):
        super().__init__(title, parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 35, 10, 10)
        self.layout.setSpacing(5)
        
    def add_data_row(self, label: str, value_widget=None):
        row_layout = QHBoxLayout()
        label_widget = HUDLabel(label)
        row_layout.addWidget(label_widget)
        if value_widget:
            row_layout.addWidget(value_widget)
            row_layout.addStretch()
        self.layout.addLayout(row_layout)
        return label_widget


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("内衬视觉定位系统 - Inner Liner Vision Positioning System")
        self.setGeometry(100, 100, 1400, 900)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0A0F1E;
            }
        """)
        
        self.camera_manager = CameraManager()
        self.use_camera = False
        self.test_mode = True
        
        self.vision_processor = VisionProcessor()
        self.vision_processor.result_ready.connect(self._on_vision_result)
        self.vision_processor.frame_ready.connect(self._on_frame_ready)
        
        self.detection_history = deque(maxlen=100)
        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()
        
        self._setup_ui()
        self._setup_timers()
        
        self.vision_processor.start()
        
    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        left_panel = self._create_left_panel()
        main_layout.addWidget(left_panel, stretch=1)
        
        center_panel = self._create_center_panel()
        main_layout.addWidget(center_panel, stretch=3)
        
        right_panel = self._create_right_panel()
        main_layout.addWidget(right_panel, stretch=1)
        
    def _create_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        
        camera_panel = HUDPanel("摄像头设置")
        camera_layout = QVBoxLayout(camera_panel)
        camera_layout.setContentsMargins(10, 40, 10, 10)
        
        self.btn_detect_cam = HUDButton("🔍 检测摄像头")
        self.btn_detect_cam.clicked.connect(self._detect_cameras)
        camera_layout.addWidget(self.btn_detect_cam)
        
        camera_layout.addWidget(HUDLabel("选择摄像头:"))
        self.camera_combo = QComboBox()
        self.camera_combo.addItem("测试模式 (模拟图像)")
        self.camera_combo.setStyleSheet(self._combo_style())
        self.camera_combo.currentIndexChanged.connect(self._select_camera)
        camera_layout.addWidget(self.camera_combo)
        
        camera_layout.addWidget(HUDLabel("分辨率:"))
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["640x480", "800x600", "1280x720", "1920x1080"])
        self.resolution_combo.setStyleSheet(self._combo_style())
        self.resolution_combo.currentIndexChanged.connect(self._change_resolution)
        camera_layout.addWidget(self.resolution_combo)
        
        layout.addWidget(camera_panel)
        
        control_panel = HUDPanel("控制面板")
        control_layout = QVBoxLayout(control_panel)
        control_layout.setContentsMargins(10, 40, 10, 10)
        
        self.btn_start = HUDButton("▶ 启动检测")
        self.btn_start.clicked.connect(self._toggle_detection)
        control_layout.addWidget(self.btn_start)
        
        self.btn_pause = HUDButton("⏸ 暂停")
        self.btn_pause.clicked.connect(self._toggle_pause)
        control_layout.addWidget(self.btn_pause)
        
        self.btn_reset = HUDButton("↺ 重置")
        self.btn_reset.clicked.connect(self._reset_system)
        control_layout.addWidget(self.btn_reset)
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["边缘检测模式", "模板匹配模式"])
        self.mode_combo.setStyleSheet(self._combo_style())
        self.mode_combo.currentIndexChanged.connect(self._change_mode)
        control_layout.addWidget(self.mode_combo)
        
        layout.addWidget(control_panel)
        
        param_panel = HUDPanel("参数设置")
        param_layout = QVBoxLayout(param_panel)
        param_layout.setContentsMargins(10, 40, 10, 10)
        
        param_layout.addWidget(HUDLabel("Canny低阈值:"))
        self.canny_low_slider = QSlider(Qt.Horizontal)
        self.canny_low_slider.setRange(10, 200)
        self.canny_low_slider.setValue(50)
        self.canny_low_slider.setStyleSheet(self._slider_style())
        self.canny_low_slider.valueChanged.connect(self._update_canny_low)
        param_layout.addWidget(self.canny_low_slider)
        
        param_layout.addWidget(HUDLabel("Canny高阈值:"))
        self.canny_high_slider = QSlider(Qt.Horizontal)
        self.canny_high_slider.setRange(50, 300)
        self.canny_high_slider.setValue(150)
        self.canny_high_slider.setStyleSheet(self._slider_style())
        self.canny_high_slider.valueChanged.connect(self._update_canny_high)
        param_layout.addWidget(self.canny_high_slider)
        
        param_layout.addWidget(HUDLabel("霍夫阈值:"))
        self.hough_slider = QSlider(Qt.Horizontal)
        self.hough_slider.setRange(20, 150)
        self.hough_slider.setValue(50)
        self.hough_slider.setStyleSheet(self._slider_style())
        self.hough_slider.valueChanged.connect(self._update_hough)
        param_layout.addWidget(self.hough_slider)
        
        layout.addWidget(param_panel)
        
        display_panel = HUDPanel("显示选项")
        display_layout = QVBoxLayout(display_panel)
        display_layout.setContentsMargins(10, 40, 10, 10)
        
        self.chk_crosshair = QCheckBox("十字准线")
        self.chk_crosshair.setChecked(True)
        self.chk_crosshair.setStyleSheet(self._checkbox_style())
        self.chk_crosshair.toggled.connect(self._toggle_crosshair)
        display_layout.addWidget(self.chk_crosshair)
        
        self.chk_target = QCheckBox("瞄准框")
        self.chk_target.setChecked(True)
        self.chk_target.setStyleSheet(self._checkbox_style())
        self.chk_target.toggled.connect(self._toggle_target)
        display_layout.addWidget(self.chk_target)
        
        self.chk_vectors = QCheckBox("旋转矢量")
        self.chk_vectors.setChecked(True)
        self.chk_vectors.setStyleSheet(self._checkbox_style())
        self.chk_vectors.toggled.connect(self._toggle_vectors)
        display_layout.addWidget(self.chk_vectors)
        
        self.chk_grid = QCheckBox("网格显示")
        self.chk_grid.setChecked(True)
        self.chk_grid.setStyleSheet(self._checkbox_style())
        self.chk_grid.toggled.connect(self._toggle_grid)
        display_layout.addWidget(self.chk_grid)
        
        layout.addWidget(display_panel)
        layout.addStretch()
        
        return panel
        
    def _create_center_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(5)
        
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        
        title_label = QLabel("内衬视觉定位系统")
        title_label.setFont(QFont("Consolas", 16, QFont.Bold))
        title_label.setStyleSheet("color: #00FFFF; background: transparent;")
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        self.fps_label = HUDLabel("FPS: 0", color=NeonStyle.NEON_GREEN)
        header_layout.addWidget(self.fps_label)
        
        layout.addWidget(header)
        
        self.vision_canvas = VisionCanvas()
        layout.addWidget(self.vision_canvas, stretch=1)
        
        status_bar = QWidget()
        status_layout = QHBoxLayout(status_bar)
        status_layout.setSpacing(20)
        
        self.indicator_vision = StatusIndicator("视觉系统")
        self.indicator_edge = StatusIndicator("边缘检测")
        self.indicator_template = StatusIndicator("模板匹配")
        self.indicator_lock = StatusIndicator("目标锁定")
        
        status_layout.addWidget(self.indicator_vision)
        status_layout.addWidget(self.indicator_edge)
        status_layout.addWidget(self.indicator_template)
        status_layout.addWidget(self.indicator_lock)
        status_layout.addStretch()
        
        layout.addWidget(status_bar)
        
        return panel
        
    def _create_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        
        result_panel = HUDPanel("检测结果")
        result_layout = QVBoxLayout(result_panel)
        result_layout.setContentsMargins(10, 40, 10, 10)
        
        self.result_labels = {}
        for name in ["X坐标:", "Y坐标:", "角度:", "匹配度:", "状态:"]:
            row = QHBoxLayout()
            label = HUDLabel(name)
            value = HUDLabel("0.00", color=NeonStyle.NEON_GREEN)
            self.result_labels[name] = value
            row.addWidget(label)
            row.addStretch()
            row.addWidget(value)
            result_layout.addLayout(row)
            
        layout.addWidget(result_panel)
        
        history_panel = HUDPanel("历史记录")
        history_layout = QVBoxLayout(history_panel)
        history_layout.setContentsMargins(10, 40, 10, 10)
        
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(4)
        self.history_table.setHorizontalHeaderLabels(["X", "Y", "θ", "Score"])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.history_table.setStyleSheet("""
            QTableWidget {
                background-color: rgba(10, 20, 40, 200);
                color: #00FFFF;
                border: 1px solid #0066CC;
                gridline-color: #003366;
            }
            QHeaderView::section {
                background-color: rgba(0, 50, 100, 200);
                color: #00FFFF;
                border: 1px solid #0066CC;
                padding: 5px;
            }
        """)
        history_layout.addWidget(self.history_table)
        
        layout.addWidget(history_panel, stretch=1)
        
        stats_panel = HUDPanel("统计数据")
        stats_layout = QVBoxLayout(stats_panel)
        stats_layout.setContentsMargins(10, 40, 10, 10)
        
        self.stats_labels = {}
        for name in ["总检测次数:", "成功次数:", "成功率:", "平均误差:"]:
            row = QHBoxLayout()
            label = HUDLabel(name)
            value = HUDLabel("0", color=NeonStyle.NEON_ORANGE)
            self.stats_labels[name] = value
            row.addWidget(label)
            row.addStretch()
            row.addWidget(value)
            stats_layout.addLayout(row)
            
        layout.addWidget(stats_panel)
        
        return panel
        
    def _slider_style(self) -> str:
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
            QSlider::sub-page:horizontal {
                background: #0088FF;
                border-radius: 4px;
            }
        """
        
    def _checkbox_style(self) -> str:
        return """
            QCheckBox {
                color: #00FFFF;
                spacing: 10px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 2px solid #0088FF;
                border-radius: 4px;
            }
            QCheckBox::indicator:checked {
                background: #00FFFF;
                border-color: #00FFFF;
            }
        """
        
    def _combo_style(self) -> str:
        return """
            QComboBox {
                background-color: rgba(0, 50, 100, 150);
                color: #00FFFF;
                border: 2px solid #0088FF;
                border-radius: 5px;
                padding: 5px;
                min-height: 25px;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 8px solid #00FFFF;
                margin-right: 5px;
            }
            QComboBox QAbstractItemView {
                background-color: rgba(10, 20, 40, 230);
                color: #00FFFF;
                border: 2px solid #0088FF;
                selection-background-color: rgba(0, 100, 200, 200);
            }
        """
        
    def _setup_timers(self):
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self._get_frame)
        self.update_timer.start(33)
        
        self.stats_timer = QTimer(self)
        self.stats_timer.timeout.connect(self._update_stats)
        self.stats_timer.start(1000)
        
    def _generate_test_frame(self):
        h, w = 480, 640
        frame = np.zeros((h, w), dtype=np.uint8)
        
        noise = np.random.randint(0, 30, (h, w), dtype=np.uint8)
        frame = cv2.add(frame, noise)
        
        t = time.time()
        
        cx = int(w/2 + 100 * math.sin(t * 0.5))
        cy = int(h/2 + 50 * math.cos(t * 0.7))
        angle = 30 * math.sin(t * 0.3)
        
        rect_w, rect_h = 120, 80
        pts = np.array([
            [-rect_w/2, -rect_h/2],
            [rect_w/2, -rect_h/2],
            [rect_w/2, rect_h/2],
            [-rect_w/2, rect_h/2]
        ])
        
        rad = math.radians(angle)
        rot = np.array([
            [math.cos(rad), -math.sin(rad)],
            [math.sin(rad), math.cos(rad)]
        ])
        pts = pts @ rot.T
        pts += [cx, cy]
        pts = pts.astype(np.int32)
        
        cv2.polylines(frame, [pts], True, 200, 3)
        
        cv2.rectangle(frame, (cx-70, cy-50), (cx+70, cy+50), 150, 2)
        
        cv2.line(frame, (50, 100), (50, 380), 180, 2)
        cv2.line(frame, (590, 100), (590, 380), 180, 2)
        
        cv2.line(frame, (150, 50), (490, 50), 160, 2)
        cv2.line(frame, (150, 430), (490, 430), 160, 2)
        
        self.vision_processor.set_frame(frame)
        
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_time)
            self.frame_count = 0
            self.last_time = current_time
            self.fps_label.setText(f"FPS: {self.fps:.1f}")
            
    def _on_vision_result(self, result: VisionResult, lines: list, pairs: list):
        self.vision_canvas.set_result(result, lines, pairs)
        
        self.result_labels["X坐标:"].setText(f"{result.x:.2f}")
        self.result_labels["Y坐标:"].setText(f"{result.y:.2f}")
        self.result_labels["角度:"].setText(f"{result.angle:.2f}°")
        self.result_labels["匹配度:"].setText(f"{result.score:.3f}")
        self.result_labels["状态:"].setText("锁定" if result.valid else "搜索")
        
        self.indicator_vision.set_active(True, NeonStyle.NEON_GREEN)
        self.indicator_edge.set_active(len(lines) > 0, NeonStyle.NEON_BLUE)
        self.indicator_template.set_active(result.valid, NeonStyle.NEON_PURPLE)
        self.indicator_lock.set_active(result.valid, NeonStyle.NEON_GREEN)
        
        if result.valid:
            self.detection_history.append(result)
            self._add_history_row(result)
            
    def _on_frame_ready(self, frame: np.ndarray):
        self.vision_canvas.set_frame(frame)
        
    def _add_history_row(self, result: VisionResult):
        row = self.history_table.rowCount()
        self.history_table.insertRow(row)
        self.history_table.setItem(row, 0, QTableWidgetItem(f"{result.x:.1f}"))
        self.history_table.setItem(row, 1, QTableWidgetItem(f"{result.y:.1f}"))
        self.history_table.setItem(row, 2, QTableWidgetItem(f"{result.angle:.1f}"))
        self.history_table.setItem(row, 3, QTableWidgetItem(f"{result.score:.3f}"))
        
        if row > 50:
            self.history_table.removeRow(0)
            
    def _update_stats(self):
        total = len(self.detection_history)
        success = sum(1 for r in self.detection_history if r.valid)
        rate = (success / total * 100) if total > 0 else 0
        
        self.stats_labels["总检测次数:"].setText(str(total))
        self.stats_labels["成功次数:"].setText(str(success))
        self.stats_labels["成功率:"].setText(f"{rate:.1f}%")
        
    def _toggle_detection(self):
        if self.update_timer.isActive():
            self.update_timer.stop()
            self.btn_start.setText("▶ 启动检测")
        else:
            self.update_timer.start(33)
            self.btn_start.setText("⏹ 停止检测")
            
    def _toggle_pause(self):
        self.vision_processor.paused = not self.vision_processor.paused
        self.btn_pause.setText("▶ 继续" if self.vision_processor.paused else "⏸ 暂停")
        
    def _reset_system(self):
        self.detection_history.clear()
        self.history_table.setRowCount(0)
        self.vision_canvas.detected_lines = []
        self.vision_canvas.parallel_pairs = []
        self.vision_canvas.vision_result = VisionResult()
        self.vision_canvas.update()
        
    def _change_mode(self, index):
        self.vision_processor.mode = "template" if index == 1 else "edge"
        
    def _update_canny_low(self, value):
        self.vision_processor.edge_detector.canny_low = value
        
    def _update_canny_high(self, value):
        self.vision_processor.edge_detector.canny_high = value
        
    def _update_hough(self, value):
        self.vision_processor.edge_detector.hough_threshold = value
        
    def _toggle_crosshair(self, checked):
        self.vision_canvas.show_crosshair = checked
        
    def _toggle_target(self, checked):
        self.vision_canvas.show_target_box = checked
        
    def _toggle_vectors(self, checked):
        self.vision_canvas.show_vectors = checked
        
    def _toggle_grid(self, checked):
        self.vision_canvas.grid_enabled = checked
        
    def _detect_cameras(self):
        self.camera_combo.clear()
        self.camera_combo.addItem("测试模式 (模拟图像)")
        
        cameras = self.camera_manager.detect_cameras()
        for idx, name in cameras:
            self.camera_combo.addItem(name)
            
        if cameras:
            self.indicator_vision.set_active(True, NeonStyle.NEON_GREEN)
            
    def _select_camera(self, index):
        if index == 0:
            self.test_mode = True
            self.use_camera = False
            self.camera_manager.is_running = False
        else:
            camera_idx = index - 1
            if self.camera_manager.open_camera(camera_idx):
                self.test_mode = False
                self.use_camera = True
                self.indicator_vision.set_active(True, NeonStyle.NEON_GREEN)
            else:
                self.indicator_vision.set_active(False, NeonStyle.NEON_RED)
                
    def _change_resolution(self, index):
        resolutions = [(640, 480), (800, 600), (1280, 720), (1920, 1080)]
        if index < len(resolutions):
            w, h = resolutions[index]
            self.camera_manager.set_resolution(w, h)
            
    def _get_frame(self):
        if self.use_camera and self.camera_manager.is_running:
            frame = self.camera_manager.read_frame()
            if frame is not None:
                self.vision_processor.set_frame(frame)
        elif self.test_mode:
            self._generate_test_frame()
            
    def closeEvent(self, event):
        self.vision_processor.stop()
        self.update_timer.stop()
        self.stats_timer.stop()
        self.camera_manager.release()
        event.accept()


try:
    import cv2
except ImportError:
    print("正在安装OpenCV...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python"])
    import cv2


class CameraManager:
    def __init__(self):
        self.cameras = {}
        self.current_camera = None
        self.current_index = -1
        self.is_running = False
        self.resolution = (640, 480)
        self.fps = 30
        
    def detect_cameras(self, max_cameras: int = 10) -> List[Tuple[int, str]]:
        available = []
        for i in range(max_cameras):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                name = f"摄像头 {i}"
                backend = cap.getBackendName()
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                available.append((i, f"{name} ({w}x{h}) [{backend}]"))
                self.cameras[i] = cap
            else:
                cap.release()
        return available
    
    def open_camera(self, index: int) -> bool:
        if self.current_camera is not None:
            self.current_camera.release()
            
        self.current_camera = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if self.current_camera.isOpened():
            self.current_index = index
            self.current_camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.current_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.current_camera.set(cv2.CAP_PROP_FPS, self.fps)
            self.is_running = True
            return True
        return False
    
    def read_frame(self) -> Optional[np.ndarray]:
        if self.current_camera is not None and self.is_running:
            ret, frame = self.current_camera.read()
            if ret:
                return frame
        return None
    
    def set_resolution(self, width: int, height: int):
        self.resolution = (width, height)
        if self.current_camera is not None:
            self.current_camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.current_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
    def release(self):
        self.is_running = False
        if self.current_camera is not None:
            self.current_camera.release()
        for cap in self.cameras.values():
            cap.release()
        self.cameras.clear()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
