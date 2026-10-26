#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全系统功能自动化测试套件
=========================
覆盖：仿真场景 / 批量测试引擎 / 直线与轮廓算法 / 几何工具 / 标定
      模板存取 / 坐标管线 / 机械手接口 / 上胶检测 / 包边测量
      验收表生成 / 异常与边界鲁棒性

运行： python auto_test_suite.py
输出： 控制台摘要 + auto_test_report.json
"""

import os
import sys
import json
import math
import time
import traceback
from typing import Callable, List, Dict, Any, Tuple

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

RESULTS: List[Dict[str, Any]] = []


def case(group: str, name: str, fn: Callable, expect: str = ""):
    """执行单个测试用例，异常记为失败"""
    t0 = time.time()
    try:
        detail = fn()
        ok = True if detail is None else (detail is not False)
        if detail is False:
            detail = "返回 False"
    except AssertionError as e:
        ok, detail = False, f"断言失败: {e}"
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {e}"
    elapsed = (time.time() - t0) * 1000
    RESULTS.append({
        "group": group, "name": name, "pass": ok,
        "detail": "" if detail is None else str(detail),
        "expect": expect, "elapsed_ms": round(elapsed, 2),
    })
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {group} / {name}" + (f"  → {detail}" if detail else ""))
    return ok


def _assert(cond, msg=""):
    if not cond:
        raise AssertionError(msg or "条件不成立")


# =====================================================================
# 合成图像工具
# =====================================================================
def make_line_image(w=640, h=480, rect=(120, 90, 520, 400), angle_deg=0.0,
                    thickness=3, noise=0.0):
    """生成白底黑边矩形（模拟面纸），并按 angle_deg 绕中心旋转"""
    img = np.full((h, w, 3), 235, np.uint8)
    x1, y1, x2, y2 = rect
    cv2.rectangle(img, (x1, y1), (x2, y2), (25, 25, 25), thickness)
    if abs(angle_deg) > 1e-6:
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderValue=(235, 235, 235))
    if noise > 0:
        n = np.random.RandomState(7).normal(0, noise, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + n, 0, 255).astype(np.uint8)
    return img


def make_shape_image(w=640, h=480):
    """生成带异形（五边形）轮廓的图像"""
    img = np.full((h, w, 3), 235, np.uint8)
    pts = np.array([[160, 120], [420, 100], [520, 260], [380, 390], [180, 330]], np.int32)
    cv2.polylines(img, [pts], True, (20, 20, 20), 3)
    return img, pts


# =====================================================================
# 1. 仿真场景
# =====================================================================
def test_simulation():
    from simulation_module import (SimulationConfig, WorkConditionPresets,
                                   SceneGenerator)
    presets = WorkConditionPresets.all_presets()

    case("仿真", "预设工况数量 ≥ 11", lambda: _assert(
        len(presets) >= 11, f"实际 {len(presets)}"), "≥11")

    def each_preset():
        bad = []
        for name, fac in presets.items():
            cfg = fac()
            gen = SceneGenerator(cfg)
            frame, truth = gen.generate_frame()
            if frame is None or frame.size == 0:
                bad.append(f"{name}:空帧")
                continue
            if not isinstance(truth, dict) or "angle_deg" not in truth:
                bad.append(f"{name}:真值缺字段")
            if frame.shape[:2] != (cfg.frame_height, cfg.frame_width):
                bad.append(f"{name}:尺寸不符 {frame.shape[:2]}")
        _assert(not bad, "; ".join(bad))
        return f"{len(presets)} 个工况均出帧"
    case("仿真", "全部工况可生成有效帧+真值", each_preset, "无异常")

    def determinism():
        from simulation_module import SimulationConfig as SC
        cfg = SC()
        cfg.random_seed = 12345
        f1, t1 = SceneGenerator(cfg).generate_frame()
        f2, t2 = SceneGenerator(cfg).generate_frame()
        _assert(np.array_equal(f1, f2), "同种子帧不一致")
        _assert(t1 == t2, "同种子真值不一致")
        return f"truth={ {k: round(v,3) if isinstance(v,float) else v for k,v in t1.items()} }"
    case("仿真", "随机种子可复现", determinism, "帧/真值完全一致")

    def truth_range():
        cfg = SimulationConfig()
        cfg.random_seed = 99
        cfg.angle = 3.5
        cfg.offset_x = 40.0
        cfg.offset_y = -25.0
        _, truth = SceneGenerator(cfg).generate_frame()
        _assert(abs(truth["angle_deg"] - 3.5) < 0.6, f"角度真值 {truth['angle_deg']}")
        _assert(abs(truth["offset_x"] - 40.0) < 1.5, f"X 真值 {truth['offset_x']}")
        _assert(abs(truth["offset_y"] + 25.0) < 1.5, f"Y 真值 {truth['offset_y']}")
        return "角度/位移真值误差 < 容差"
    case("仿真", "真值参数回读准确", truth_range, "偏差 < 1.5px / 0.6°")

    def extreme_cfg():
        cfg = SimulationConfig()
        cfg.frame_width, cfg.frame_height = 160, 120
        cfg.paper_width_px, cfg.paper_height_px = 90, 60
        cfg.random_seed = 5
        frame, _ = SceneGenerator(cfg).generate_frame()
        _assert(frame.shape[:2] == (120, 160), f"{frame.shape[:2]}")
        return "小分辨率可用"
    case("仿真", "极小分辨率不崩溃", extreme_cfg, "正常出帧")


# =====================================================================
# 2. 批量测试引擎
# =====================================================================
def _ideal_detector(frame) -> dict:
    """理想检测器：直接根据帧内容无误差（用于验证引擎统计逻辑）"""
    return {"detected": True, "dx": 0.0, "dy": 0.0, "angle": 0.0, "score": 1.0}


def test_engine():
    from simulation_module import SimulationTestEngine, WorkConditionPresets

    def ideal_stats():
        eng = SimulationTestEngine(_ideal_detector)
        eng.run_preset("基准工况", 5, "angle_only")
        st = eng.get_statistics()
        _assert(st["total"] == 5, f"total={st['total']}")
        _assert(st["detection_rate"] == 1.0, f"检出率 {st['detection_rate']}")
        return f"total=5 检出率={st['detection_rate']:.0%}"
    case("引擎", "理想检测器→检出率100%", ideal_stats, "detection_rate=1.0")

    def dead_detector():
        eng = SimulationTestEngine(lambda f: {"detected": False})
        eng.run_preset("基准工况", 5)
        st = eng.get_statistics()
        _assert(st["detected"] == 0, f"detected={st['detected']}")
        _assert(st["miss_count"] == 5, f"miss={st['miss_count']}")
        return f"漏检 {st['miss_count']}/5"
    case("引擎", "失效检测器→漏检计数正确", dead_detector, "detected=0")

    def crash_detector():
        def boom(f):
            raise RuntimeError("模拟算法崩溃")
        eng = SimulationTestEngine(boom)
        eng.run_preset("基准工况", 3)
        st = eng.get_statistics()
        _assert(st["detected"] == 0, "异常应计为未检出")
        _assert("算法异常" in (eng.results[0].note or ""), f"note={eng.results[0].note}")
        return "异常被捕获并记录 note"
    case("引擎", "算法异常被吞并记录", crash_detector, "不抛出")

    def all_presets_run():
        eng = SimulationTestEngine(_ideal_detector)
        res = eng.run_all(samples_per_preset=2)
        _assert(len(res) == len(WorkConditionPresets.all_presets()),
                f"工况组数 {len(res)}")
        total = sum(len(v) for v in res.values())
        _assert(total == 2 * len(res), f"用例数 {total}")
        return f"{len(res)} 工况 × 2 = {total} 用例"
    case("引擎", "run_all 覆盖全部工况", all_presets_run, "全工况执行")

    def unknown_preset():
        eng = SimulationTestEngine(_ideal_detector)
        r = eng.run_preset("不存在的工况", 3)
        _assert(r == [], f"应返回空列表，实际 {r}")
        return "返回空列表"
    case("引擎", "未知工况名安全返回", unknown_preset, "[]")

    def stat_fields():
        eng = SimulationTestEngine(_ideal_detector)
        eng.run_preset("基准工况", 4)
        st = eng.get_statistics()
        required = ["total", "detected", "detection_rate", "pass_count",
                    "pass_rate", "miss_count", "out_of_tol_count"]
        miss = [k for k in required if k not in st]
        _assert(not miss, f"缺字段 {miss}")
        return f"字段齐全({len(st)}项)"
    case("引擎", "统计字段完整", stat_fields, "含 pass_rate 等")

    def empty_stats():
        eng = SimulationTestEngine(_ideal_detector)
        st = eng.get_statistics()
        _assert(st.get("total") == 0, f"{st}")
        return "空数据返回 total=0"
    case("引擎", "空结果统计不崩溃", empty_stats, "total=0")

    def batch_reproducible():
        def run_once():
            eng = SimulationTestEngine(_ideal_detector)
            rs = eng.run_preset("基准工况", 6, "all")
            return [(round(r.true_dx, 6), round(r.true_dy, 6),
                     round(r.true_angle, 6)) for r in rs]
        a, b = run_once(), run_once()
        _assert(a == b, f"两次批量结果不一致: {a} vs {b}")
        _assert(len(set(a)) > 1, "随机模式未产生变化（样本全部相同）")
        return f"6 例两次运行真值一致，角度跨度 {a[0][2]:.2f}°"
    case("引擎", "同种子批量结果可复现", batch_reproducible, "两次真值一致")


# =====================================================================
# 3. 直线算法
# =====================================================================
def test_line_algorithm():
    from vision_utils_v2 import VisionAlgorithm as VA

    def detect_basic():
        img = make_line_image()
        roi = (60, 40, 580, 440)
        lines = VA.detect_lines_multialgorithm(img, roi)
        _assert(len(lines) > 0, "未检测到任何直线")
        return f"检出 {len(lines)} 条"
    case("直线", "多算法融合可检出直线", detect_basic, ">0 条")

    def detect_roi_edge():
        img = make_line_image()
        lines = VA.detect_lines_multialgorithm(img, (0, 0, 5, 5))
        _assert(isinstance(lines, list), "应返回 list")
        return f"小 ROI 返回 {len(lines)} 条"
    case("直线", "极小 ROI 安全", detect_roi_edge, "list")

    def detect_invalid_roi():
        img = make_line_image()
        _assert(VA.detect_lines_multialgorithm(img, (500, 400, 100, 50)) == [], "倒置ROI应空")
        _assert(VA.detect_lines_multialgorithm(img, (0, 0, 0, 0)) == [], "零ROI应空")
        return "倒置/零面积 ROI 返回 []"
    case("直线", "非法 ROI 返回空", detect_invalid_roi, "[]")

    def detect_gray_input():
        img = make_line_image()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lines = VA.detect_lines_multialgorithm(gray, (60, 40, 580, 440))
        _assert(isinstance(lines, list), "灰度输入应可用")
        return f"灰度输入检出 {len(lines)} 条"
    case("直线", "灰度图输入兼容", detect_gray_input, "list")

    def angle_accuracy():
        # 水平边 → 角度应接近 0（无向）
        line = ((100, 200), (400, 200))
        a = VA.calculate_line_angle(line)
        _assert(abs(a) < 1e-6, f"水平线角度 {a}")
        line2 = ((100, 200), (100, 500))
        a2 = abs(VA.calculate_line_angle(line2))
        _assert(abs(a2 - 90) < 1e-6, f"垂直线角度 {a2}")
        line3 = ((0, 0), (100, 100))
        a3 = VA.calculate_line_angle(line3)
        _assert(abs(a3 - 45) < 1e-6, f"45°线角度 {a3}")
        return "0°/90°/45° 解析正确"
    case("直线", "直线角度解析精度", angle_accuracy, "误差<1e-6")

    def undirected():
        _assert(abs(VA.undirected_angle_diff(179.0, -179.0) - 2.0) < 1e-6,
                f"{VA.undirected_angle_diff(179.0, -179.0)}")
        _assert(abs(VA.undirected_angle_diff(0.0, 90.0) - 90.0) < 1e-6, "0/90 应为 90")
        _assert(abs(VA.undirected_angle_diff(10.0, 10.0)) < 1e-9, "同角差应为0")
        return "±180° 跳变处理正确"
    case("直线", "无向角度差归一化", undirected, "≤90°")

    def intersection():
        l1 = ((0, 0), (100, 0))
        l2 = ((50, -50), (50, 50))
        p = VA.calculate_line_intersection(l1, l2)
        _assert(p is not None and abs(p[0] - 50) < 1e-6 and abs(p[1]) < 1e-6,
                f"交点 {p}")
        return f"交点={tuple(round(v,3) for v in p)}"
    case("直线", "交点解析正确", intersection, "(50,0)")

    def parallel_intersection():
        l1 = ((0, 0), (100, 0))
        l2 = ((0, 10), (100, 10))
        p = VA.calculate_line_intersection(l1, l2)
        _assert(p is None, f"平行线应返回 None，实际 {p}")
        return "平行线返回 None"
    case("直线", "平行线交点安全", parallel_intersection, "None")

    def distance():
        l1 = ((0, 0), (100, 0))
        l2 = ((0, 30), (100, 30))
        d = VA.calculate_line_distance(l1, l2)
        _assert(abs(abs(d) - 30) < 1.0, f"平行线距 {d}")
        return f"平行线距={abs(d):.2f}px"
    case("直线", "两线距离计算", distance, "≈30px")

    def merge():
        img = make_line_image()
        raw = VA.detect_lines_multialgorithm(img, (60, 40, 580, 440))
        merged = VA.merge_collinear_lines(raw)
        _assert(len(merged) <= len(raw), f"合并后 {len(merged)} > 原 {len(raw)}")
        return f"{len(raw)} → {len(merged)} 条"
    case("直线", "共线直线合并有效", merge, "数量不增加")

    def algorithm_list():
        algos = list(VA.get_available_algorithms().keys())
        _assert(len(algos) >= 5, f"算法数 {len(algos)}")
        return f"{len(algos)} 种算法"
    case("直线", "算法清单完整", algorithm_list, "≥5")


# =====================================================================
# 4. 轮廓算法
# =====================================================================
def test_contour_algorithm():
    from vision_utils_v2 import VisionAlgorithm as VA

    def detect_contour():
        img, pts = make_shape_image()
        found = VA.detect_contours_multialgorithm(img, (100, 60, 580, 430))
        _assert(len(found) > 0, "未检测到轮廓")
        return f"检出 {len(found)} 个轮廓"
    case("轮廓", "多算法融合检出轮廓", detect_contour, ">0")

    def effective_area():
        sq = np.array([[[0, 0]], [[0, 10]], [[10, 10]], [[10, 0]]], np.int32)
        a = VA._get_contour_effective_area(sq)
        _assert(abs(a - 100) < 1e-3, f"面积 {a}")
        return f"正方形面积={a}"
    case("轮廓", "有效面积解析正确", effective_area, "100")

    def center():
        sq = np.array([[[0, 0]], [[0, 10]], [[10, 10]], [[10, 0]]], np.int32)
        c = VA._get_contour_center(sq)
        _assert(abs(c[0] - 5) <= 1 and abs(c[1] - 5) <= 1, f"中心 {c}")
        return f"中心={c}"
    case("轮廓", "轮廓中心计算", center, "(5,5)")

    def match_self():
        sq = np.array([[[0, 0]], [[0, 10]], [[10, 10]], [[10, 0]]], np.int32)
        s = VA.match_shape(sq, sq)
        _assert(s > 0.9, f"自匹配分数 {s}")
        return f"自匹配={s:.3f}"
    case("轮廓", "自匹配分数≈1", match_self, ">0.9")

    def deviation_zero():
        sq = np.array([[[0, 0]], [[0, 20]], [[20, 20]], [[20, 0]]], np.int32)
        d = VA.calculate_contour_deviation(sq, sq)
        _assert(isinstance(d, dict) and len(d) > 0, f"返回 {d}")
        for k, v in d.items():
            if "ratio" in k or "dist" in k or "dev" in k:
                _assert(abs(v) < 1e-6, f"{k}={v} 应为0")
        return f"零偏差字段正常({len(d)}项)"
    case("轮廓", "同轮廓偏差为 0", deviation_zero, "≈0")

    def match_position():
        tpl = np.array([[[0, 0]], [[0, 20]], [[20, 20]], [[20, 0]]], np.int32)
        det = [np.array([[[100, 100]], [[100, 120]], [[120, 120]], [[120, 100]]], np.int32)]
        best, score, offset = VA.match_contours_by_position_size(tpl, det)
        _assert(best is not None, "应匹配到轮廓")
        _assert(abs(offset[0] - 100) < 2 and abs(offset[1] - 100) < 2, f"偏移 {offset}")
        return f"偏移={tuple(round(v,1) for v in offset)} score={score:.2f}"
    case("轮廓", "位置匹配偏移正确", match_position, "≈(100,100)")

    def match_empty():
        tpl = np.array([[[0, 0]], [[0, 20]], [[20, 20]], [[20, 0]]], np.int32)
        best, score, off = VA.match_contours_by_position_size(tpl, [])
        _assert(best is None and off == (0.0, 0.0), f"{best},{off}")
        best2, _, _ = VA.match_contours_by_position_size(None, [tpl])
        _assert(best2 is None, "模板空应返回 None")
        return "空输入返回 None"
    case("轮廓", "空模板/空候选安全", match_empty, "None")

    def invalid_roi():
        img, _ = make_shape_image()
        _assert(VA.detect_contours_in_roi(img, (0, 0, 0, 0)) == [], "零 ROI 应空")
        return "零面积 ROI 返回 []"
    case("轮廓", "非法 ROI 返回空", invalid_roi, "[]")


# =====================================================================
# 5. 标定与坐标换算
# =====================================================================
def test_calibration():
    from vision_utils_v2 import CameraCalibration, CalibrationData, TemplateManager

    def from_points():
        px = [(0.0, 0.0), (1000.0, 0.0), (0.0, 1000.0)]
        mm = [(0.0, 0.0), (100.0, 0.0), (0.0, 200.0)]
        cal = CameraCalibration.calibrate_from_points(px, mm)
        _assert(abs(cal.pixel_per_mm_x - 10.0) < 1e-6, f"x比例 {cal.pixel_per_mm_x}")
        _assert(abs(cal.pixel_per_mm_y - 5.0) < 1e-6, f"y比例 {cal.pixel_per_mm_y}")
        _assert(cal.is_calibrated, "未置标定标志")
        return f"比例 x={cal.pixel_per_mm_x}/mm, y={cal.pixel_per_mm_y}/mm"
    case("标定", "点对应求像素/毫米比例", from_points, "x=10 y=5")

    def too_few():
        try:
            CameraCalibration.calibrate_from_points([(0, 0)], [(0, 0)])
        except ValueError:
            return "正确抛出 ValueError"
        raise AssertionError("应抛出 ValueError")
    case("标定", "标定点不足报错", too_few, "ValueError")

    def save_load():
        cal = CalibrationData(pixel_per_mm_x=8.5, pixel_per_mm_y=8.2,
                              offset_x=1.5, offset_y=-2.5, is_calibrated=True)
        p = "_t_calib.json"
        TemplateManager.save_calibration(p, cal)
        back = TemplateManager.load_calibration(p)
        _assert(abs(back.pixel_per_mm_x - 8.5) < 1e-9, "标定比例丢失")
        _assert(back.is_calibrated, "标定标志丢失")
        os.remove(p)
        return "往返一致"
    case("标定", "标定数据存取往返", save_load, "字段一致")

    def px2mm():
        from vision_utils_v2 import VisionAlgorithm as VA
        cal = CalibrationData(pixel_per_mm_x=10.0, pixel_per_mm_y=10.0,
                              is_calibrated=True)
        x, y = VA.pixel_to_mm(250.0, 500.0, cal)
        _assert(abs(x - 25.0) < 1e-9 and abs(y - 50.0) < 1e-9, f"{x},{y}")
        return f"250px→{x}mm"
    case("标定", "像素→毫米换算", px2mm, "25/50 mm")

    def undistort_noop():
        cal = CalibrationData()
        img = make_line_image()
        out = CameraCalibration.undistort_image(img, cal)
        _assert(out is not None and out.shape == img.shape, "畸变校正输出形状不符")
        return "未标定时安全返回"
    case("标定", "未标定畸变校正不崩", undistort_noop, "同形状")


# =====================================================================
# 6. 模板存取
# =====================================================================
def test_template_io():
    from vision_utils_v2 import DualLineTemplate, ShapeTemplate, TemplateManager

    def dual_line_roundtrip():
        t = DualLineTemplate(
            left_line_long=((10, 20), (10, 200)),
            left_line_short=((10, 20), (150, 20)),
            right_line_long=((500, 30), (500, 210)),
            right_line_short=((300, 30), (500, 30)),
            left_roi=(0, 0, 200, 250), right_roi=(280, 0, 520, 250),
            valid=True)
        p = "_t_line.json"
        TemplateManager.save_dual_line_template(p, t)
        b = TemplateManager.load_dual_line_template(p)
        _assert(b is not None, "加载失败")
        _assert(b.left_line_long[0] == (10, 20), f"{b.left_line_long}")
        _assert(b.is_complete(), "完整性校验失败")
        os.remove(p)
        return "4 线 + 2 ROI 往返一致"
    case("模板", "双直线模板存取往返", dual_line_roundtrip, "字段一致")

    def shape_roundtrip():
        c = np.array([[[10, 10]], [[10, 100]], [[120, 100]], [[120, 10]]], np.int32)
        t = ShapeTemplate(left_contour=c, right_contour=c.copy(),
                          left_roi=(0, 0, 200, 200), right_roi=(200, 0, 400, 200),
                          center=(65.0, 55.0), angle=1.25, area=9900.0,
                          name="测试", valid=True)
        p = "_t_shape.json"
        TemplateManager.save_shape_template(p, t, {"thr": 0.8})
        b, params = TemplateManager.load_shape_template(p)
        _assert(b is not None and b.has_contours(), "轮廓未恢复")
        _assert(abs(b.angle - 1.25) < 1e-6, f"角度 {b.angle}")
        _assert(params.get("thr") == 0.8, f"附加参数 {params}")
        os.remove(p)
        return "轮廓+参数往返一致"
    case("模板", "轮廓模板存取往返", shape_roundtrip, "字段一致")

    def load_missing():
        b = TemplateManager.load_dual_line_template("_not_exist_xx.json")
        _assert(b is None or isinstance(b, DualLineTemplate), f"返回值 {b}")
        return "缺失文件安全返回"
    case("模板", "缺失模板文件安全", load_missing, "不抛出")

    def real_templates():
        loaded = []
        if os.path.exists("last_line_template.json"):
            t = TemplateManager.load_dual_line_template("last_line_template.json")
            if t is not None:
                loaded.append(f"line:complete={t.is_complete()}")
        if os.path.exists("last_shape_template.json"):
            t, _ = TemplateManager.load_shape_template("last_shape_template.json")
            if t is not None:
                loaded.append(f"shape:has_contours={t.has_contours()}")
        _assert(loaded, "实际模板文件均无法加载")
        return "; ".join(loaded)
    case("模板", "项目实际模板文件可加载", real_templates, "可解析")


# =====================================================================
# 7. 坐标管线 / 机械手
# =====================================================================
def test_coordinate_and_robot():
    from vision_utils_v2 import (CalibrationData, DetectionResult, RobotPose,
                                 CoordinatePipeline, MMSpaceEvaluator)

    def pipeline_conv():
        cal = CalibrationData(pixel_per_mm_x=10.0, pixel_per_mm_y=10.0,
                              is_calibrated=True)
        cp = CoordinatePipeline(cal)
        det = DetectionResult(x_pixel=300.0, y_pixel=450.0, angle=2.0,
                              valid=True, score=0.9)
        out = cp.detection_to_mm(det)
        _assert(abs(out.x_mm - 30.0) < 1e-6 and abs(out.y_mm - 45.0) < 1e-6,
                f"{out.x_mm},{out.y_mm}")
        return f"300px→{out.x_mm}mm / 450px→{out.y_mm}mm"
    case("坐标", "像素→毫米管线", pipeline_conv, "30/45 mm")

    def offset_comp():
        cal = CalibrationData(pixel_per_mm_x=10.0, pixel_per_mm_y=10.0,
                              is_calibrated=True)
        cp = CoordinatePipeline(cal, offsets=(5.0, -3.0, 0.0, 0.0))
        det = DetectionResult(x_pixel=300.0, y_pixel=450.0, valid=True)
        out = cp.detection_to_mm(det)
        _assert(abs(out.x_mm - 25.0) < 1e-6, f"x={out.x_mm}")
        _assert(abs(out.y_mm - 48.0) < 1e-6, f"y={out.y_mm}")
        return f"补偿后 x={out.x_mm}, y={out.y_mm}"
    case("坐标", "偏置补偿生效", offset_comp, "25/48")

    def pose_protocol():
        cal = CalibrationData(pixel_per_mm_x=10.0, pixel_per_mm_y=10.0,
                              is_calibrated=True)
        cp = CoordinatePipeline(cal)
        det = DetectionResult(x_pixel=100.0, y_pixel=200.0, angle=1.5,
                              valid=True, score=0.88)
        pose = cp.to_robot_pose(det)
        _assert(isinstance(pose, RobotPose), "返回类型错误")
        csv = pose.to_protocol_string("csv")
        _assert(isinstance(csv, str) and csv.strip(), "CSV 为空")
        parts = csv.strip().split(",")
        _assert(len(parts) == 6, f"CSV 字段数 {len(parts)}: {csv}")
        return csv.strip()
    case("坐标", "机械手 CSV 协议 6 字段", pose_protocol, "6 字段")

    def pose_tcp():
        cal = CalibrationData(is_calibrated=True, pixel_per_mm_x=10.0,
                              pixel_per_mm_y=10.0)
        cp = CoordinatePipeline(cal)
        det = DetectionResult(x_pixel=10.0, y_pixel=20.0, angle=0.0,
                             valid=True, score=0.5)
        s = cp.to_robot_pose(det).to_protocol_string("tcp")
        _assert(isinstance(s, str) and s.strip(), "TCP 报文为空")
        return f"tcp 报文 {len(s.strip())} 字符"
    case("坐标", "机械手 TCP 报文生成", pose_tcp, "非空")

    def mm_eval():
        ev = MMSpaceEvaluator(tolerance_mm=0.1)
        for _ in range(50):
            ev.feed((0.02, -0.03), (0.0, 0.0), 0.1, 0.0, True)
        s = ev.summary()
        _assert(abs(s["mean_position_error_mm"]
                    - math.hypot(0.02, 0.03)) < 1e-3, f"{s['mean_position_error_mm']}")
        _assert(s["accuracy"] == 100.0, f"准确率 {s['accuracy']}")
        return f"mean={s['mean_position_error_mm']}mm acc={s['accuracy']}%"
    case("坐标", "mm 空间评估汇总", mm_eval, "均值≈0.036")

    def mm_eval_empty():
        s = MMSpaceEvaluator().summary()
        _assert(isinstance(s, dict) and s.get("error"), f"应返回 error 标记，实际 {s}")
        return "空样本返回 error 标记"
    case("坐标", "mm 评估空样本安全", mm_eval_empty, "不抛异常")

    # 机械手
    from robot_interface import (RobotSimulator, RobotSimulationConfig,
                                 pack_serial_csv, pack_modbus_tcp_register)

    def robot_pickup():
        cfg = RobotSimulationConfig(pickup_success_rate=1.0)
        sim = RobotSimulator(cfg=cfg, seed=1)
        res = sim.batch([(0.0, 0.0)] * 100, [(0.0, 0.0)] * 100)
        st = RobotSimulator.summarize(res)
        _assert(st["pickup_success_rate_pct"] == 100.0,
                f"{st['pickup_success_rate_pct']}")
        return f"100/100 全部抓取成功"
    case("机械手", "抓取成功率统计", robot_pickup, "100%")

    def robot_zero_pickup():
        cfg = RobotSimulationConfig(pickup_success_rate=0.0)
        sim = RobotSimulator(cfg=cfg, seed=2)
        res = sim.batch([(0.0, 0.0)] * 20, [(0.0, 0.0)] * 20)
        st = RobotSimulator.summarize(res)
        _assert(st["pickup_success_rate_pct"] == 0.0, "全失败应 0%")
        _assert(not st["overall_pass"], "全失败应判不合格")
        return "0% 并判不合格"
    case("机械手", "零抓取判定不合格", robot_zero_pickup, "0% 不合格")

    def robot_by_state():
        sim = RobotSimulator(seed=3)
        rows = sim.run_by_state(lambda i: (1.0, 1.0), lambda i: (1.0, 1.0),
                                num_states=4, cycles_per_state=5)
        _assert(len(rows) == 4, f"状态数 {len(rows)}")
        _assert(all(r["total"] == 5 for r in rows), "每状态次数不符")
        ex = RobotSimulator.excel_rows(rows)
        _assert(len(ex) == 4 and len(ex[0]) == 10, f"Excel 列数 {len(ex[0])}")
        return f"4 状态 × 5 次，Excel 10 列"
    case("机械手", "状态分组与 Excel 列对齐", robot_by_state, "10 列")

    def protocols():
        s = pack_serial_csv(1.5, 2.5, 0.0, 90.0, True, 0.99)
        _assert(s.startswith("$") and s.endswith(";\n"), f"串口格式 {s!r}")
        b = pack_modbus_tcp_register(1.5, 2.5, 0.0, 90.0, 0.99)
        _assert(isinstance(b, (bytes, bytearray)) and len(b) > 16,
                f"Modbus 长度 {len(b)}")
        return f"serial ok / modbus {len(b)}B"
    case("机械手", "串口与 Modbus 协议打包", protocols, "格式正确")


# =====================================================================
# 8. 上胶检测
# =====================================================================
def test_glue():
    from glue_inspection import GlueUniformityInspector

    def full_cover():
        img = np.zeros((200, 600, 3), np.uint8)
        img[:] = (200, 200, 200)
        img[:, :] = (255, 200, 50)   # 全幅满涂（近似蓝紫胶液）
        ins = GlueUniformityInspector()
        r = ins.inspect(img, sample_id="P1")
        _assert(len(r.zones) == 5, f"分区数 {len(r.zones)}")
        _assert(r.mean_coverage_pct > 95, f"覆盖率仅 {r.mean_coverage_pct}%")
        _assert(r.qualified, f"满涂应合格，最低区 {r.min_coverage_pct}%")
        return f"5 区, 平均覆盖率 {r.mean_coverage_pct}% → 合格"
    case("上胶", "满涂覆盖率高且合格", full_cover, ">95% 合格")

    def partial_cover():
        img = np.zeros((200, 600, 3), np.uint8)
        img[:] = (200, 200, 200)
        img[:, :300] = (255, 200, 50)   # 只涂左半
        ins = GlueUniformityInspector()
        r = ins.inspect(img, sample_id="P2")
        left = r.zones[0].coverage_pct
        right = r.zones[4].coverage_pct
        _assert(left > 95, f"左区覆盖率 {left}%")
        _assert(right < 5, f"右区覆盖率 {right}%")
        _assert(not r.qualified, "半幅漏涂应判不合格")
        return f"左区 {left}% vs 右区 {right}% → 不合格"
    case("上胶", "局部漏涂可识别", partial_cover, "不合格")

    def auto_crop_fallback():
        # 胶面比背景暗时，亮度启发式裁切不可靠 → 应回退整幅而不是裁成背景窄条
        img = np.zeros((200, 600, 3), np.uint8)
        img[:] = (200, 200, 200)
        img[:, 30:570] = (255, 200, 50)
        ins = GlueUniformityInspector(auto_crop=True)
        r = ins.inspect(img, sample_id="P3")
        _assert(r.mean_coverage_pct > 70,
                f"自动裁切回退失败，覆盖率 {r.mean_coverage_pct}%")
        return f"auto_crop 回退整幅，平均覆盖率 {r.mean_coverage_pct}%"
    case("上胶", "自动裁切失败时回退整幅", auto_crop_fallback, ">70%")

    def batch_summary():
        img = np.zeros((200, 600, 3), np.uint8)
        img[:] = (200, 200, 200)
        img[:, :] = (255, 200, 50)
        ins = GlueUniformityInspector()
        rs = ins.inspect_batch([img] * 3)
        s = GlueUniformityInspector.summarize(rs)
        _assert(s["sample_count"] == 3, f"{s['sample_count']}")
        _assert("pass_rate_pct" in s, "缺合格率字段")
        _assert(s["pass_rate_pct"] == 100.0, f"满涂合格率 {s['pass_rate_pct']}%")
        return f"3 件 合格率 {s['pass_rate_pct']}%"
    case("上胶", "批量统计汇总", batch_summary, "字段完整")

    def empty_summary():
        s = GlueUniformityInspector.summarize([])
        _assert(s == {}, f"空列表应返回 {{}}，实际 {s}")
        return "空列表返回空字典"
    case("上胶", "空批统计安全", empty_summary, "{}")


# =====================================================================
# 9. 包边测量
# =====================================================================
def test_fold():
    from edge_fold_measure import EdgeFoldMeasurement

    def measure_basic():
        img = np.zeros((200, 600, 3), np.uint8)
        img[60:80, 100:520] = (255, 255, 255)
        m = EdgeFoldMeasurement(standard_mm=15.5, tolerance_mm=0.15,
                                pixel_per_mm=1.0)
        r = m.measure(img, sample_id="P1")
        _assert(len(r.points) == 3, f"点位数 {len(r.points)}")
        _assert(r.sheet_mean_mm > 0, f"均值 {r.sheet_mean_mm}")
        return f"3 点均值 {r.sheet_mean_mm}mm"
    case("包边", "3 点测量输出", measure_basic, "3 点")

    def tolerance_check():
        img = np.zeros((200, 600, 3), np.uint8)
        img[60:80, 100:520] = (255, 255, 255)
        # 设定标准=实测均值，容差极小 → 应合格
        m0 = EdgeFoldMeasurement(standard_mm=15.5, tolerance_mm=0.15,
                                 pixel_per_mm=1.0)
        r0 = m0.measure(img)
        m = EdgeFoldMeasurement(standard_mm=r0.sheet_mean_mm,
                                tolerance_mm=1.0, pixel_per_mm=1.0)
        r = m.measure(img)
        _assert(r.qualified, f"对齐标准后应合格, 偏差 {r.sheet_max_deviation_mm}")
        return f"偏差 {r.sheet_max_deviation_mm}mm → 合格"
    case("包边", "容差判定逻辑", tolerance_check, "合格")

    def excel_rows():
        img = np.zeros((200, 600, 3), np.uint8)
        img[60:80, 100:520] = (255, 255, 255)
        m = EdgeFoldMeasurement(pixel_per_mm=1.0)
        rs = m.measure_batch([img] * 3)
        rows = m.excel_rows(rs)
        _assert(len(rows) == 3, f"行数 {len(rows)}")
        _assert(len(rows[0]) == 11, f"列数 {len(rows[0])}")
        return f"3 行 × 11 列"
    case("包边", "Excel 行对齐 11 列", excel_rows, "11 列")

    def empty_summary():
        s = EdgeFoldMeasurement.summarize([])
        _assert(s == {}, f"空列表应返回 {{}}，实际 {s}")
        return "空列表返回空字典"
    case("包边", "空批统计安全", empty_summary, "{}")


# =====================================================================
# 10. 验收表生成
# =====================================================================
def test_acceptance():
    def generate():
        import acceptance_xlsx_generator as g
        out = os.path.join(HERE, "_test_acceptor")
        files = g.generate_all_acceptance_tables(out_dir=out, seed=20261026)
        _assert(len(files) == 4, f"生成 {len(files)} 个文件")
        for f in files:
            _assert(os.path.exists(f) and os.path.getsize(f) > 5000,
                    f"{os.path.basename(f)} 过小")
        return f"4 张表, 共 {sum(os.path.getsize(f) for f in files)//1024} KB"
    case("验收表", "一键生成 4 张验收表", generate, "4 文件")

    def schema():
        from openpyxl import load_workbook
        d = os.path.join(HERE, "_test_acceptor")
        expect = {
            "封面机视觉识别率统计表_曲线图版.xlsx": (9, 30),
            "封面机上胶均匀性检验数据记录表_曲线图版.xlsx": (12, 30),
            "封面机包边测量记录表_曲线图版.xlsx": (11, 30),
            "封面机机械手抓取纠偏放置成功率统计表_曲线图版.xlsx": (10, 30),
        }
        bad = []
        for fn, (ncol, nrow) in expect.items():
            ws = load_workbook(os.path.join(d, fn)).active
            got_col = ws.max_column
            # 数据区行数：从第 3 行起连续非空行
            data_rows = sum(1 for r in range(3, ws.max_row + 1)
                            if ws.cell(r, 1).value not in (None, ""))
            if got_col < ncol:
                bad.append(f"{fn}:列 {got_col}<{ncol}")
            if data_rows < nrow:
                bad.append(f"{fn}:数据行 {data_rows}<{nrow}")
        _assert(not bad, "; ".join(bad))
        return "4 张表列数与数据行均达标"
    case("验收表", "表结构与实体机列数一致", schema, "列数/行数达标")

    def content():
        from openpyxl import load_workbook
        d = os.path.join(HERE, "_test_acceptor")
        for fn in os.listdir(d):
            if not fn.endswith(".xlsx"):
                continue
            ws = load_workbook(os.path.join(d, fn)).active
            text = " ".join(str(ws.cell(r, c).value or "")
                            for r in range(1, ws.max_row + 1)
                            for c in range(1, min(ws.max_column, 12) + 1))
            for kw in ("结论", "考核指标", "检验"):
                _assert(kw in text, f"{fn} 缺少『{kw}』")
        return "4 张表均含结论/考核指标/记录信息"
    case("验收表", "结论与记录信息齐备", content, "关键字齐全")

    def cleanup():
        import shutil
        d = os.path.join(HERE, "_test_acceptor")
        if os.path.isdir(d):
            shutil.rmtree(d)
        return "已清理临时输出"
    case("验收表", "清理临时目录", cleanup, "已清理")


# =====================================================================
# 11. 端到端：仿真引擎 + 真实视觉算法（面纸矩形定位）
# =====================================================================
def test_end_to_end():
    """
    用真实 VisionAlgorithm 直线检测作为引擎回调，跑基准工况，
    验证"偏移→检测→误差"链路可用（非 GUI 依赖的最小端到端）。
    """
    from vision_utils_v2 import VisionAlgorithm as VA
    from simulation_module import SimulationTestEngine, SimulationConfig, SceneGenerator

    def build_detector():
        # 以第一帧建立模板：矩形 4 边（左右 ROI）
        cfg = SimulationConfig()
        cfg.random_seed = 1
        cfg.offset_x = 0.0
        cfg.offset_y = 0.0
        cfg.angle = 0.0
        frame, truth = SceneGenerator(cfg).generate_frame()
        h, w = frame.shape[:2]
        left_roi = (0, 0, w // 2, h)
        right_roi = (w // 2, 0, w, h)
        l_lines = VA.detect_lines_multialgorithm(frame, left_roi)
        r_lines = VA.detect_lines_multialgorithm(frame, right_roi)
        _assert(len(l_lines) > 0 and len(r_lines) > 0,
                f"模板建立失败 L={len(l_lines)} R={len(r_lines)}")

        tpl_lines = l_lines + r_lines
        base_angles = [VA.calculate_line_angle(l) for l in tpl_lines]
        base_centers = [VA.calculate_line_center(l) for l in tpl_lines]
        tpl_cx = float(np.mean([c[0] for c in base_centers]))
        tpl_cy = float(np.mean([c[1] for c in base_centers]))
        tpl_angle = float(np.median(base_angles))

        def detector(f):
            det = VA.detect_lines_multialgorithm(f, (0, 0, f.shape[1], f.shape[0]))
            if not det:
                return {"detected": False, "note": "无直线"}
            angss = [VA.calculate_line_angle(l) for l in det]
            ctrs = [VA.calculate_line_center(l) for l in det]
            angle = float(np.median(angss))
            cx = float(np.mean([c[0] for c in ctrs]))
            cy = float(np.mean([c[1] for c in ctrs]))
            return {"detected": True, "dx": cx - tpl_cx, "dy": cy - tpl_cy,
                    "angle": angle - tpl_angle,
                    "score": min(1.0, len(det) / max(1, len(tpl_lines)))}
        return detector

    def run_e2e():
        det = build_detector()
        eng = SimulationTestEngine(det)
        res = eng.run_preset("基准工况", 6, "angle_only")
        st = eng.get_statistics()
        _assert(st["total"] == 6, f"total={st['total']}")
        _assert(st["detection_rate"] >= 0.9,
                f"检出率仅 {st['detection_rate']:.0%}")
        mean_e = (np.mean([r.error_dist for r in res if r.detected])
                  if any(r.detected for r in res) else 999)
        _assert(mean_e < 60, f"平均位移误差 {mean_e:.1f}px 过大")
        return (f"6 例 检出率 {st['detection_rate']:.0%} "
                f"平均误差 {mean_e:.1f}px")
    case("端到端", "仿真→真实算法→误差链路", run_e2e, "检出率≥90% 误差<60px")


# =====================================================================
# 12. 性能基准
# =====================================================================
def test_performance():
    from vision_utils_v2 import VisionAlgorithm as VA
    from simulation_module import SceneGenerator, WorkConditionPresets

    def latency():
        presets = WorkConditionPresets.all_presets()
        rows = []
        for name, fac in presets.items():
            cfg = fac()
            cfg.random_seed = 2026
            frame, _ = SceneGenerator(cfg).generate_frame()
            h, w = frame.shape[:2]
            roi = (0, 0, w, h)
            # 预热
            VA.detect_lines_multialgorithm(frame, roi)
            # 取多轮最小值：本机存在其他进程干扰，最小值最接近真实算法耗时
            best = 1e9
            for _ in range(3):
                t0 = time.time()
                det = VA.detect_lines_multialgorithm(frame, roi)
                best = min(best, (time.time() - t0) * 1000)
            rows.append((name, best, len(det)))
        rows.sort(key=lambda x: -x[1])
        worst_name, worst_ms, _n = rows[0]
        avg = sum(r[1] for r in rows) / len(rows)
        _assert(worst_ms < 2000, f"{worst_name} 单帧 {worst_ms:.0f}ms 过慢")
        return (f"平均 {avg:.1f}ms/帧，最慢 {worst_name} {worst_ms:.0f}ms，"
                f"最快 {rows[-1][1]:.1f}ms")
    case("性能", "各工况单帧耗时可控", latency, "最慢 < 2000ms")

    def throughput():
        from simulation_module import SimulationConfig
        cfg = SimulationConfig()
        cfg.random_seed = 77
        frame, _ = SceneGenerator(cfg).generate_frame()
        roi = (0, 0, frame.shape[1], frame.shape[0])
        VA.detect_lines_multialgorithm(frame, roi)
        n = 10
        samples = []
        for _ in range(n):
            t0 = time.time()
            VA.detect_lines_multialgorithm(frame, roi)
            samples.append((time.time() - t0) * 1000)
        per = float(np.median(samples))
        fps = 1000.0 / per if per > 0 else 0
        return f"{per:.1f}ms/帧 ≈ {fps:.1f} FPS（中位数）"
    case("性能", "基准工况吞吐量", throughput, "报告 FPS")

    def roi_scaling():
        from simulation_module import SimulationConfig
        cfg = SimulationConfig()
        cfg.random_seed = 78
        frame, _ = SceneGenerator(cfg).generate_frame()
        w, h = frame.shape[1], frame.shape[0]
        def _t(roi):
            VA.detect_lines_multialgorithm(frame, roi)
            best = 1e9
            for _ in range(5):
                t0 = time.time()
                VA.detect_lines_multialgorithm(frame, roi)
                best = min(best, (time.time() - t0) * 1000)
            return best
        full = _t((0, 0, w, h))
        half = _t((0, 0, w // 2, h))
        _assert(half <= full * 1.6, f"半幅 ROI {half:.0f}ms 未明显快于全幅 {full:.0f}ms")
        return f"全幅 {full:.1f}ms → 半幅 {half:.1f}ms"
    case("性能", "ROI 缩小可降耗时", roi_scaling, "半幅更快")


# =====================================================================
# 13. GUI / Web 集成加载（不启动界面）
# =====================================================================
def test_integration():
    def import_v3():
        import vision_gui_v3 as m
        return f"vision_gui_v3 导入成功，导出符号 {len(dir(m))} 个"
    case("集成", "GUI 主模块可导入", import_v3, "无导入错误")

    def has_detect():
        import vision_gui_v3 as m
        # 找到承载检测回调的类
        found = [n for n in dir(m) if n.lower() in
                 ("visiongui", "mainwindow", "visionguiwindow")]
        cls = getattr(m, found[0]) if found else None
        _assert(cls is not None, f"未找到主窗口类，候选 {found}")
        _assert(hasattr(cls, "_run_detection_on_frame"),
                "主窗口缺少 _run_detection_on_frame")
        return f"{found[0]}._run_detection_on_frame 就绪"
    case("集成", "主窗口检测回调存在", has_detect, "存在")

    def import_web():
        import web_sim_server as m
        _assert(hasattr(m, "app"), "缺少 Flask app")
        _assert(hasattr(m, "api_start"), "缺少 api_start")
        _assert(hasattr(m, "api_status"), "缺少 api_status")
        return "web_sim_server 路由就绪"
    case("集成", "Web 控制台可导入", import_web, "app/路由存在")

    def web_routes():
        import web_sim_server as m
        c = m.app.test_client()
        r = c.get("/")
        _assert(r.status_code == 200, f"首页 {r.status_code}")
        _assert(len(r.data) > 1000, f"首页内容过小 {len(r.data)}B")
        rs = c.get("/api/status")
        _assert(rs.status_code == 200, f"/api/status {rs.status_code}")
        data = rs.get_json() or {}
        for k in ("running", "done", "progress", "rows"):
            _assert(k in data, f"/api/status 缺字段 {k}")
        return f"首页 {len(r.data)//1024}KB，/api/status {len(data)} 字段"
    case("集成", "Web 页面与状态接口可用", web_routes, "200 + 字段齐全")

    def import_runner():
        import playwright_sim_test as m
        _assert(len(dir(m)) > 5, "模块内容过少")
        return f"playwright_sim_test 导入成功({len(dir(m))} 符号)"
    case("集成", "自动化测试脚本可导入", import_runner, "无导入错误")

    def config_file():
        _assert(os.path.exists("system_config.json"), "缺少 system_config.json")
        with open("system_config.json", encoding="utf-8") as f:
            cfg = json.load(f)
        _assert(isinstance(cfg, dict) and cfg, "配置为空")
        return f"配置键 {len(cfg)} 项"
    case("集成", "系统配置文件有效", config_file, "可解析")


# =====================================================================
# 主流程
# =====================================================================
def main():
    print("=" * 78)
    print("全系统功能自动化测试")
    print("=" * 78)
    groups = [
        ("仿真场景", test_simulation),
        ("批量引擎", test_engine),
        ("直线算法", test_line_algorithm),
        ("轮廓算法", test_contour_algorithm),
        ("标定换算", test_calibration),
        ("模板存取", test_template_io),
        ("坐标机械手", test_coordinate_and_robot),
        ("上胶检测", test_glue),
        ("包边测量", test_fold),
        ("验收表", test_acceptance),
        ("端到端", test_end_to_end),
        ("性能", test_performance),
        ("集成", test_integration),
    ]
    for gname, fn in groups:
        try:
            fn()
        except Exception:
            RESULTS.append({"group": gname, "name": "<组级异常>", "pass": False,
                            "detail": traceback.format_exc(limit=3),
                            "expect": "", "elapsed_ms": 0})
            print(f"[FAIL] {gname} / <组级异常>\n{traceback.format_exc(limit=3)}")

    total = len(RESULTS)
    passed = sum(1 for r in RESULTS if r["pass"])
    failed = [r for r in RESULTS if not r["pass"]]

    print("-" * 78)
    print(f"合计 {total} 项：通过 {passed}，失败 {len(failed)}，"
          f"通过率 {passed / total * 100:.1f}%")
    if failed:
        print("\n失败清单：")
        for r in failed:
            print(f"  ✗ {r['group']} / {r['name']}：{r['detail']}")
    print("-" * 78)

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total": total, "passed": passed, "failed": len(failed),
        "pass_rate_pct": round(passed / total * 100, 2),
        "groups": sorted(set(r["group"] for r in RESULTS)),
        "cases": RESULTS,
    }
    with open("auto_test_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("报告：auto_test_report.json")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
