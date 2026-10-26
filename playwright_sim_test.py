#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Playwright 可视化自动化：模拟测试参数设置 + 批量测试 + 结果校验
================================================================
流程：
1. 自动启动 Web 控制台（web_sim_server.py）
2. 打开 Chromium（有头模式，可观看自动化过程；同步录制视频）
3. 场景A：基准工况 × 30样本（仅角度随机）——填参数、点开始、等待完成、校验合格率
4. 场景B：全部工况 × 5样本 ——同上
5. 每个场景截图保存至 sim_ui_report/，结束后输出断言结论
运行：python playwright_sim_test.py
"""

import os
import socket
import subprocess
import sys
import time

from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:5188"
REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim_ui_report")
SERVER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web_sim_server.py")

PASS_THRESHOLD = 99.5  # 考核合格率阈值（%），项目指标 ≥99.5%


def wait_port(port: int, timeout: float = 90.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except OSError:
            time.sleep(0.6)
    return False


def parse_pass_rate(overall_text: str) -> float:
    """从整体统计文本中解析『考核合格率 xx.xx%』"""
    key = "考核合格率"
    idx = overall_text.find(key)
    if idx < 0:
        return -1.0
    seg = overall_text[idx + len(key):]
    num = ""
    for ch in seg:
        if ch.isdigit() or ch == ".":
            num += ch
        elif num:
            break
    return float(num) if num else -1.0


def shot(page, filename: str):
    """截图前摘除 MJPEG 流（无限流会阻塞 fonts.ready 导致截图挂起），截完恢复"""
    page.evaluate(
        "const i=document.getElementById('preview'); if(i) i.removeAttribute('src')")
    page.screenshot(path=filename)
    page.evaluate(
        "const i=document.getElementById('preview');"
        " if(i && !i.getAttribute('src')) i.src='/video_feed'")


def run_scenario(page, tag: str, scope: str, preset: str, samples: int,
                 random_mode: str, shots: dict):
    print(f"\n===== 场景{tag}：scope={scope} preset={preset} "
          f"samples={samples} mode={random_mode} =====")

    # —— 可视化设置参数 ——
    if scope == "current":
        page.check("input[name=scope][value=current]")
        page.select_option("#preset", label=preset)
    else:
        page.check("input[name=scope][value=all]")
    page.fill("#samples", str(samples))
    label = {"angle_only": "仅角度随机（标准位）", "all": "位置+角度随机",
             "position_only": "仅位置随机", "fixed": "固定不随机"}[random_mode]
    page.select_option("#random-mode", label=label)
    shot(page, os.path.join(REPORT_DIR, shots["params"]))

    # —— 启动批量测试 ——
    page.click("#btn-start")
    page.wait_for_timeout(1200)  # 等首批帧到达，截"进行中"画面
    shot(page, os.path.join(REPORT_DIR, shots["running"]))

    # —— 等待完成（进度标签出现"测试完成"） ——
    page.wait_for_selector("#progress-label:has-text('测试完成')", timeout=300000)

    # —— 读取整体统计并校验 ——
    overall_text = page.inner_text("#overall")
    pass_rate = parse_pass_rate(overall_text)
    shot(page, os.path.join(REPORT_DIR, shots["result"]))
    ok = pass_rate >= PASS_THRESHOLD
    print(f"整体统计：{overall_text.splitlines()[0] if overall_text else '（空）'}")
    print(f"考核合格率 = {pass_rate:.2f}%  →  "
          f"{'PASS' if ok else 'FAIL'}（阈值 {PASS_THRESHOLD}%）")
    return ok, pass_rate


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    proc = subprocess.Popen([sys.executable, SERVER],
                            cwd=os.path.dirname(SERVER))
    try:
        if not wait_port(5188):
            print("错误：Web 控制台启动超时")
            return 1

        results = []
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False, slow_mo=120)
            context = browser.new_context(
                viewport={"width": 1180, "height": 1500},
                record_video_dir=os.path.join(REPORT_DIR, "video"),
                record_video_size={"width": 1180, "height": 1500},
                locale="zh-CN",
            )
            page = context.new_page()
            # 页面含无限 MJPEG 流，load 事件永不触发，须等待 domcontentloaded
            page.goto(BASE, wait_until="domcontentloaded")
            page.wait_for_selector("#btn-start")

            # 场景A：基准工况 × 30（仅角度随机）
            results.append(("A", run_scenario(
                page, "A", "current", "基准工况", 30, "angle_only",
                {"params": "01_场景A_参数设置.png",
                 "running": "02_场景A_批量进行中.png",
                 "result": "03_场景A_测试结果.png"})))

            # 场景B：全部工况 × 5
            results.append(("B", run_scenario(
                page, "B", "all", "", 5, "angle_only",
                {"params": "04_场景B_参数设置.png",
                 "running": "05_场景B_批量进行中.png",
                 "result": "06_场景B_测试结果.png"})))

            video = page.video
            context.close()
            if video is not None:
                path = video.path()
                browser.close()
                dst = os.path.join(REPORT_DIR, "自动化过程录像.webm")
                try:
                    os.replace(path, dst)
                    print(f"录像已保存：{dst}")
                except OSError:
                    print(f"录像位于：{path}")
            else:
                browser.close()

        print("\n========== 自动化测试结论 ==========")
        all_ok = True
        for tag, (ok, rate) in results:
            print(f"场景{tag}: 合格率 {rate:.2f}%  {'PASS' if ok else 'FAIL'}")
            all_ok = all_ok and ok
        print(f"截图与录像目录：{REPORT_DIR}")
        return 0 if all_ok else 2
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
