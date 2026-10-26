#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机械手坐标输出接口层 — 浩达 ZFM-500B 对齐 stub
=================================================

实体封面机验收口径：
  - 视觉输出坐标点位 → 机械手纠偏参考（坐标偏差平均 0.085mm / 最大 0.112mm）
  - 坐标系：世界坐标 mm，估读精度 ±0.001mm
  - 考核指标：抓取成功率 ≥99.8%，定位误差 ≤0.1mm

本模块提供：
  1. Modbus/TCP 协议打包 → 供工业控制器/PLC 直接消费
  2. 串口 RS-232 协议打包 → 供传统机械手（如三菱 RV-N、发那科）
  3. 机械手纠偏仿真 stub → 闭环仿真：视觉坐标 → 模拟机械手纠偏 → 放置精度
  4. 抓取成功率/定位误差 的 mm 空间统计（对齐实体机第 4 张验收表）
"""

import math
import time
import struct
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional, Dict, Any


# =====================================================================
# 1. 协议打包
# =====================================================================

def pack_modbus_tcp_register(x_mm: float, y_mm: float, z_mm: float, a_deg: float,
                              score: float, seq: int = 1) -> bytes:
    """
    Modbus/TCP 风格寄存器打包（保持 4 坐标 + 1 评分 + 1 帧序号）
    
    格式：[MBAP 头 7字节] + [PDU 功能码 0x10] + [寄存器地址 2字节] + [数量 2字节]
          + [字节计数 1字节] + [4个 float32 共 16字节] + [score float32 4字节] + [校验 2字节]
    
    真实部署时请把寄存器地址改成你的 PLC 实际映射表。
    """
    payload = struct.pack("<fffff", x_mm, y_mm, z_mm, a_deg, score)
    length = 1 + 1 + 2 + 2 + 1 + len(payload)
    mbap = struct.pack(">HHH", seq, 0x0000, length)
    pdu = bytes([0x10, 0x00, 0x00, 0x00, 0x0A, 0x14]) + payload  # funcode 0x10 = write multiple registers
    # 简化：只返回 MBAP + PDU（真实 Modbus/TCP 不需要 CRC）
    return mbap + pdu


def pack_serial_csv(x_mm: float, y_mm: float, z_mm: float, a_deg: float,
                    valid: bool, score: float,
                    frame_id: int = 0) -> str:
    """
    RS-232 串口协议：单行 CSV，供传统机械手控制器
    
    格式：<帧头>,X_mm,Y_mm,Z_mm,A_deg,Valid,Score,<校验>,<帧尾>
    例：$POSE,123.456, 45.678, 0.000, 90.000, 1, 0.987, A3,;
    """
    body = f"POSE,{x_mm:.3f},{y_mm:.3f},{z_mm:.3f},{a_deg:.3f},{int(valid)},{score:.3f}"
    checksum = sum(ord(c) for c in body) & 0xFF
    return f"${body},{checksum:02X};\n"


# =====================================================================
# 2. 机械手纠偏仿真 stub（闭环）
# =====================================================================

@dataclass
class RobotSimulationConfig:
    """机械手仿真参数 — 对齐实体机抓取纠偏放置成功率验收口径"""
    # 伺服纠偏能力（对视觉坐标的响应精度）
    servo_latency_ms: float = 15.0      # 伺服延迟
    servo_error_mm_sigma: float = 0.02   # 伺服定位误差标准差（mm）
    # 真空吸盘抓取
    pickup_success_rate: float = 0.998   # 单次抓取成功率（考核线 ≥99.8%）
    # 纠偏补偿
    correction_gain: float = 0.95       # 纠偏伺服增益（小于1 避免过冲）
    # 放位精度
    placement_error_mm_sigma: float = 0.03  # 最终放置位置误差（mm）
    # 工装基准偏移
    fixture_offset_mm: Tuple[float, float] = (0.0, 0.0)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RobotCycleResult:
    """单次 抓取→纠偏→放置 循环结果"""
    visual_x_mm: float = 0.0
    visual_y_mm: float = 0.0
    pickup_success: bool = True
    corrected_x_mm: float = 0.0
    corrected_y_mm: float = 0.0
    placed_x_mm: float = 0.0
    placed_y_mm: float = 0.0
    final_error_mm: float = 0.0
    cycle_time_ms: float = 0.0
    timestamp: float = 0.0


class RobotSimulator:
    """
    机械手抓取—纠偏—放置 闭环仿真器
    
    实体机验收口径（来自 封面机机械手抓取纠偏放置成功率统计表）：
      - 30 状态 × 50 次/状态 = 1500 次循环
      - 考核指标：抓取成功率 ≥99.8%，定位误差 ≤0.1mm
      - 实机结果：抓取成功率 平均 99.8%、最低 96%
      - 定位误差：平均 0.063mm / 最大 0.083mm
    """

    def __init__(self, cfg: RobotSimulationConfig = None, seed: int = None):
        self.cfg = cfg or RobotSimulationConfig()
        import random
        self._rng = random.Random(seed)

    def single_cycle(self, visual_pose_mm: Tuple[float, float],
                     target_pose_mm: Tuple[float, float]) -> RobotCycleResult:
        """
        执行一次 抓取→纠偏→放置 循环
        
        参数：
            visual_pose_mm: (x, y) 视觉输出坐标（mm）
            target_pose_mm: (x, y) 期望放置的目标点位（mm）
            
        返回 RobotCycleResult
        """
        import math
        t0 = time.time()
        pickup_ok = self._rng.random() < self.cfg.pickup_success_rate

        if not pickup_ok:
            return RobotCycleResult(
                visual_x_mm=visual_pose_mm[0], visual_y_mm=visual_pose_mm[1],
                pickup_success=False,
                cycle_time_ms=(time.time() - t0) * 1000
            )

        # 纠偏：视觉给的坐标 → 伺服补偿 → 带误差的修正坐标
        # 期望位移差 = target - visual（需要补多少）
        desired_dx = target_pose_mm[0] - visual_pose_mm[0]
        desired_dy = target_pose_mm[1] - visual_pose_mm[1]
        # 伺服增益（小于1）+ 伺服误差（高斯）
        servo_dx = desired_dx * self.cfg.correction_gain + \
                   self._rng.gauss(0, self.cfg.servo_error_mm_sigma)
        servo_dy = desired_dy * self.cfg.correction_gain + \
                   self._rng.gauss(0, self.cfg.servo_error_mm_sigma)
        corrected_x = visual_pose_mm[0] + servo_dx
        corrected_y = visual_pose_mm[1] + servo_dy

        # 放置：再叠加一次放置误差（机械臂末端累积）
        placed_x = corrected_x + self._rng.gauss(0, self.cfg.placement_error_mm_sigma)
        placed_y = corrected_y + self._rng.gauss(0, self.cfg.placement_error_mm_sigma)
        # 工装基准偏移
        placed_x += self.cfg.fixture_offset_mm[0]
        placed_y += self.cfg.fixture_offset_mm[1]

        final_err = math.sqrt(
            (placed_x - target_pose_mm[0]) ** 2 +
            (placed_y - target_pose_mm[1]) ** 2
        )

        return RobotCycleResult(
            visual_x_mm=visual_pose_mm[0], visual_y_mm=visual_pose_mm[1],
            pickup_success=True,
            corrected_x_mm=corrected_x, corrected_y_mm=corrected_y,
            placed_x_mm=placed_x, placed_y_mm=placed_y,
            final_error_mm=final_err,
            cycle_time_ms=(time.time() - t0) * 1000 + self.cfg.servo_latency_ms,
            timestamp=time.time()
        )

    def batch(self, visual_poses: List[Tuple[float, float]],
              target_poses: List[Tuple[float, float]]) -> List[RobotCycleResult]:
        assert len(visual_poses) == len(target_poses)
        return [self.single_cycle(v, t) for v, t in zip(visual_poses, target_poses)]

    @staticmethod
    def summarize(results: List[RobotCycleResult],
                  pickup_rate_threshold: float = 0.998,
                  pos_error_threshold_mm: float = 0.1) -> Dict[str, Any]:
        """
        汇总 → 对齐实体机第 4 张验收表口径
        
        返回字段与实体机一致：
          抓取成功率(%) / 平均定位误差(mm) / 最大定位误差(mm) / 判定
        """
        if not results:
            return {}
        import statistics as st
        n = len(results)
        successes = [r for r in results if r.pickup_success]
        fail_because_pickup = n - len(successes)
        pickup_rate = len(successes) / n if n else 0.0

        if successes:
            pos_errors = [r.final_error_mm for r in successes if r.pickup_success]
            mean_err = st.mean(pos_errors) if pos_errors else 0.0
            max_err = max(pos_errors) if pos_errors else 0.0
            pos_pass_count = sum(1 for e in pos_errors if e <= pos_error_threshold_mm)
        else:
            mean_err = max_err = 0.0
            pos_pass_count = 0

        pickup_pass = pickup_rate >= pickup_rate_threshold
        pos_pass = pos_pass_count == len(successes) if successes else True
        overall_pass = pickup_pass and pos_pass

        return {
            "cycle_count": n,
            "pickup_success": len(successes),
            "pickup_fail": fail_because_pickup,
            "pickup_success_rate_pct": round(pickup_rate * 100, 2),
            "pickup_threshold_pct": pickup_rate_threshold * 100,
            "mean_positioning_error_mm": round(mean_err, 4),
            "max_positioning_error_mm": round(max_err, 4),
            "positioning_threshold_mm": pos_error_threshold_mm,
            "positioning_pass_count": pos_pass_count,
            "positioning_total_count": len(successes),
            "pickup_pass": pickup_pass,
            "positioning_pass": pos_pass,
            "overall_pass": overall_pass,
            "caliber": "浩达 ZFM-500B 机械手抓取纠偏放置验收口径对齐",
            "reference": "封面机机械手抓取纠偏放置成功率统计表_曲线图版.xlsx",
        }

    def run_by_state(self, visual_pose_factory, target_pose_factory,
                     num_states: int = 30, cycles_per_state: int = 50) -> List[Dict[str, Any]]:
        """
        按"状态编号 × 每次 N 次"分组执行 → 对齐实体机 Excel 第 4 张表口径
        
        实体机口径：30 状态 × 50 次/状态 = 1500 次循环
        
        参数：
            visual_pose_factory(state_idx) → (x,y) mm 视觉坐标（每个状态可返回不同视觉分布）
            target_pose_factory(state_idx) → (x,y) mm 目标放置点位
        """
        import statistics as st
        state_rows = []
        for s in range(num_states):
            visual = [visual_pose_factory(s) for _ in range(cycles_per_state)]
            target = [target_pose_factory(s) for _ in range(cycles_per_state)]
            cycles = self.batch(visual, target)
            succ = [c for c in cycles if c.pickup_success]
            fail = cycles_per_state - len(succ)
            rate = len(succ) / cycles_per_state * 100.0
            if succ:
                pos_errs = [c.final_error_mm for c in succ]
                mean_err = st.mean(pos_errs)
                max_err = max(pos_errs)
            else:
                mean_err = max_err = 0.0
            state_rows.append({
                "state_id": s + 1,
                "state_label": f"状态{s+1}",
                "total": cycles_per_state,
                "success": len(succ),
                "fail": fail,
                "success_rate_pct": round(rate, 2),
                "mean_pos_error_mm": round(mean_err, 4),
                "max_pos_error_mm": round(max_err, 4),
            })
        return state_rows

    @staticmethod
    def excel_rows(state_rows: List[Dict[str, Any]],
                   pickup_rate_threshold_pct: float = 99.8,
                   pos_error_threshold_mm: float = 0.1) -> List[List]:
        """
        状态分组行 → 对齐实体机 Excel 表头
        
        列：[状态编号, 抓取总次数, 抓取成功(次), 抓取失败(次), 抓取成功率(%),
             平均定位误差(mm), 最大定位误差(mm), 判定, 考核线 99.8%, 合格线 0.1mm]
        """
        rows = []
        for r in state_rows:
            qualified = (r["success_rate_pct"] >= pickup_rate_threshold_pct
                         and r["max_pos_error_mm"] <= pos_error_threshold_mm)
            rows.append([
                r["state_label"], r["total"], r["success"], r["fail"],
                r["success_rate_pct"], r["mean_pos_error_mm"], r["max_pos_error_mm"],
                "合格" if qualified else "不合格",
                pickup_rate_threshold_pct, pos_error_threshold_mm,
            ])
        return rows
