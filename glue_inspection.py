#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
上胶均匀性视觉检测 — 浩达验收 stub
======================================

对齐实体机第 2 张验收表（封面机上胶均匀性检验数据记录表）口径：
  - 横向 5 个检测区（左区 0~20% / 左中区 20~40% / 中区 40~60% / 右中区 60~80% / 右区 80~100%）
  - 每区上胶覆盖率 (%)，单区 ≥95% 合格
  - 漏胶斑点检测（可见白点）
  - 抽样 30 件，判定"单件是否全合格" → 汇总合格率

实体机结果参考：
  合格率 100%，全批平均覆盖率 97.8%，最低覆盖率 91.0%
  合格线 95% / 估读精度 ±1%
"""

import cv2
import numpy as np
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any


# =====================================================================
# 上胶区域检测核心
# =====================================================================

@dataclass
class GlueZoneResult:
    zone_index: int              # 1~5
    zone_name: str               # 左区 / 左中区 / 中区 / 右中区 / 右区
    coverage_pct: float = 0.0    # 上胶覆盖率 %
    has_visible_white_dots: bool = False   # 可见漏胶斑点
    qualified: bool = True


@dataclass
class GlueSheetResult:
    zones: List[GlueZoneResult] = field(default_factory=list)
    mean_coverage_pct: float = 0.0
    min_coverage_pct: float = 0.0
    max_coverage_pct: float = 0.0
    leak_zone_count: int = 0     # 漏胶区数
    qualified: bool = True       # 单件合格？
    sample_id: str = ""


ZONE_NAMES = ["左区(0~20%)", "左中区(20~40%)", "中区(40~60%)", "右中区(60~80%)", "右区(80~100%)"]


class GlueUniformityInspector:
    """
    上胶均匀性 5 区视觉检测
    
    用法：
        inspector = GlueUniformityInspector(min_coverage_pct=95.0)
        result = inspector.inspect(image, sheet_mask=None)
    """

    def __init__(self, min_coverage_pct: float = 95.0,
                 glue_color_hsv_range: Tuple[Tuple[int, int, int], Tuple[int, int, int]] =
                     ((80, 50, 50), (130, 255, 255))):
        # 胶液颜色范围（默认蓝紫色，可按实际胶液调整 HSV）
        self.hsv_lo = glue_color_hsv_range[0]
        self.hsv_hi = glue_color_hsv_range[1]
        self.min_coverage = min_coverage_pct

    def _extract_sheet(self, image: np.ndarray,
                       mask: Optional[np.ndarray] = None) -> np.ndarray:
        """提取面纸区域（自动或 mask）"""
        if mask is not None:
            masked = cv2.bitwise_and(image, image, mask=mask)
            return masked
        # 自动提取：找最大白色/浅色矩形轮廓
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        return image[y:y + h, x:x + w]

    def inspect(self, image: np.ndarray, sheet_mask: np.ndarray = None,
                sample_id: str = "") -> GlueSheetResult:
        sheet = self._extract_sheet(image, sheet_mask)
        h, w = sheet.shape[:2]
        total_area = h * w
        hsv = cv2.cvtColor(sheet, cv2.COLOR_BGR2HSV)
        glue_mask = cv2.inRange(hsv, np.array(self.hsv_lo), np.array(self.hsv_hi))

        zones: List[GlueZoneResult] = []
        leak_count = 0

        for i in range(5):
            x1 = int(w * i / 5)
            x2 = int(w * (i + 1) / 5)
            zone_mask = np.zeros_like(glue_mask)
            zone_mask[:, x1:x2] = 255
            zone_glue = cv2.bitwise_and(glue_mask, zone_mask)
            zone_area = (x2 - x1) * h
            glue_area = cv2.countNonZero(zone_glue)
            coverage = (glue_area / zone_area * 100.0) if zone_area > 0 else 0.0

            # 漏胶斑点检测（本区内上胶掩码的"洞"）
            zone_inv = cv2.bitwise_not(zone_glue)
            zone_inv = cv2.bitwise_and(zone_inv, zone_mask)
            _, zone_inv_thr = cv2.threshold(zone_inv, 200, 255, cv2.THRESH_BINARY)
            white_dot_count, _ = cv2.connectedComponents(zone_inv_thr)
            # 如果漏胶斑点数量异常（远大于胶液正常边缘像素数 → 视为漏胶）
            has_leak = coverage < self.min_coverage or (
                coverage >= self.min_coverage and white_dot_count > max(50, glue_area * 0.02)
            )

            qualified = coverage >= self.min_coverage and not has_leak
            zones.append(GlueZoneResult(
                zone_index=i + 1, zone_name=ZONE_NAMES[i],
                coverage_pct=round(coverage, 2),
                has_visible_white_dots=has_leak, qualified=qualified
            ))
            if not qualified:
                leak_count += 1

        coverages = [z.coverage_pct for z in zones]
        return GlueSheetResult(
            zones=zones,
            mean_coverage_pct=round(np.mean(coverages), 2),
            min_coverage_pct=round(np.min(coverages), 2),
            max_coverage_pct=round(np.max(coverages), 2),
            leak_zone_count=leak_count,
            qualified=(leak_count == 0),
            sample_id=sample_id
        )

    def inspect_batch(self, images: List[np.ndarray],
                      masks: List[np.ndarray] = None,
                      sample_ids: List[str] = None) -> List[GlueSheetResult]:
        return [self.inspect(img, masks[i] if masks else None,
                             sample_ids[i] if sample_ids else f"P{i+1}")
                for i, img in enumerate(images)]

    @staticmethod
    def summarize(results: List[GlueSheetResult],
                  min_coverage_pct: float = 95.0) -> Dict[str, Any]:
        """对齐实体机口径：合格率 / 全批最低覆盖率 / 全批平均覆盖率"""
        if not results:
            return {}
        total = len(results)
        qualified = sum(1 for r in results if r.qualified)
        mean_all = np.mean([r.mean_coverage_pct for r in results])
        min_all = np.min([r.min_coverage_pct for r in results])
        return {
            "sample_count": total,
            "qualified_count": qualified,
            "failure_count": total - qualified,
            "pass_rate_pct": round(qualified / total * 100, 2),
            "mean_coverage_pct_all": round(mean_all, 2),
            "min_coverage_pct_all": round(min_all, 2),
            "max_coverage_pct_all": round(max(r.max_coverage_pct for r in results), 2),
            "min_coverage_threshold_pct": min_coverage_pct,
            "caliber": "浩达 ZFM-500B 上胶均匀性验收口径对齐",
            "reference": "封面机上胶均匀性检验数据记录表_曲线图版.xlsx",
        }
