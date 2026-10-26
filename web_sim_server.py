#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模拟测试 Web 控制台（供 Playwright 可视化自动化驱动）
=====================================================
说明：
- Playwright 只能驱动浏览器，故将模拟测试的参数设置/批量测试/实时预览/统计结果
  做成与桌面 GUI 同风格的 Web 页面，后端直接复用 vision_gui_v3 的检测逻辑与
  simulation_module 的 BatchTestThread / SimulationTestEngine。
- 页面：http://127.0.0.1:5188
  API：
    GET  /                控制页
    GET  /video_feed      MJPEG 实时预览流（批量逐帧画面）
    GET  /api/status      轮询：进度/当前用例/整体统计/按工况表
    POST /api/start       开始批量 {scope, preset, samples, random_mode}
    POST /api/stop        停止批量
- 运行：python web_sim_server.py
"""

import io
import sys
import threading
import time

import cv2
from flask import Flask, Response, jsonify, render_template_string
from PyQt5.QtCore import QObject, pyqtSlot
from PyQt5.QtWidgets import QApplication

import vision_gui_v3 as G
from simulation_module import BatchTestThread, SimulationTestEngine, WorkConditionPresets

HOST, PORT = "127.0.0.1", 5188

app = Flask(__name__)

# ---------------- 共享状态（批量线程写 / HTTP 线程读） ----------------
LOCK = threading.Lock()
STATE = {
    "running": False,
    "done": False,
    "error": "",
    "preset_label": "",          # 进度条显示文本
    "progress": {"cur": 0, "total": 0},
    "case": None,                # 当前用例信息 dict（供页面渲染）
    "frames_done": 0,
    "jpeg": None,                # 最新预览帧（JPEG bytes）
    "overall": None,             # 整体统计 dict
    "rows": [],                  # 按工况统计行
    "engine": None,
    "thread": None,
}

PASS_DIST_PX = 50.0
PASS_ANGLE_DEG = 5.0

# 桌面 GUI 使用的基准模板（与 GUI「模拟测试」页一致）
TEMPLATES = {
    "left_roi": (355, 150, 430, 570),
    "right_roi": (850, 150, 925, 570),
    "left_line_long": ((390, 195), (390, 525)),
    "left_line_short": ((395, 190), (500, 190)),
    "right_line_long": ((890, 195), (890, 525)),
    "right_line_short": ((780, 530), (885, 530)),
}


def _setup_templates(win):
    c = win.image_canvas
    c.left_roi = TEMPLATES["left_roi"]
    c.right_roi = TEMPLATES["right_roi"]
    c.left_line_long = TEMPLATES["left_line_long"]
    c.left_line_short = TEMPLATES["left_line_short"]
    c.right_line_long = TEMPLATES["right_line_long"]
    c.right_line_short = TEMPLATES["right_line_short"]
    win.current_mode = "line"


# ---------------- 批量线程回调 ----------------

def _on_progress(preset, cur, total):
    with LOCK:
        STATE["preset_label"] = f"工况【{preset}】进度 {cur}/{total}"
        STATE["progress"] = {"cur": STATE["frames_done"], "total": STATE["progress"]["total"]}


def _on_case_done(_result):
    with LOCK:
        STATE["frames_done"] += 1
        STATE["progress"] = {"cur": STATE["frames_done"], "total": STATE["progress"]["total"]}


def _on_frame(payload):
    """逐帧：编码 JPEG 供 MJPEG 流 + 组装当前用例信息"""
    frame = payload.get("frame")
    truth = payload.get("truth") or {}
    res = payload.get("result") or {}
    preset = payload.get("preset", "")

    err_dist = res.get("error_dist")
    err_ang = res.get("error_angle")
    detected = bool(res.get("detected"))
    ok = bool(detected and err_dist is not None and err_ang is not None
              and err_dist <= PASS_DIST_PX and abs(err_ang) <= PASS_ANGLE_DEG)
    verdict = "合格" if ok else ("漏检" if not detected else "超差")
    vcolor = "#00FF96" if ok else "#FF5566"
    if detected and res.get("det_dx") is not None:
        case = {
            "preset": preset,
            "case_id": res.get("case_id"),
            "verdict": verdict,
            "vcolor": vcolor,
            "truth": [truth.get("offset_x", 0), truth.get("offset_y", 0),
                      truth.get("angle_deg", 0)],
            "det": [res.get("det_dx"), res.get("det_dy"), res.get("det_angle")],
            "err_dist": err_dist,
            "err_angle": err_ang,
            "elapsed": res.get("elapsed_ms"),
        }
    else:
        case = {"preset": preset, "case_id": res.get("case_id"),
                "verdict": "漏检", "vcolor": "#FF5566",
                "truth": [truth.get("offset_x", 0), truth.get("offset_y", 0),
                          truth.get("angle_deg", 0)],
                "det": None, "err_dist": None, "err_angle": None, "elapsed": None}

    if frame is not None:
        small = cv2.resize(frame, (800, 450), interpolation=cv2.INTER_AREA)
        okb, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 82])
        jpeg = buf.tobytes() if okb else None
    else:
        jpeg = None

    with LOCK:
        STATE["case"] = case
        if jpeg is not None:
            STATE["jpeg"] = jpeg


def _on_finished(engine: SimulationTestEngine):
    stats = engine.get_statistics()
    rows = []
    for name, st in engine.get_statistics_by_preset().items():
        rows.append({
            "name": name,
            "det_rate": st.get("detection_rate", st.get("success_rate", 0)) * 100,
            "det_text": f"{st.get('detected', 0)}/{st.get('total', 0)}",
            "pass_rate": st.get("pass_rate", 0) * 100,
            "pass_text": f"{st.get('pass_count', 0)}/{st.get('total', 0)}",
            "mean_err": st.get("mean_error_dist_px", 0),
            "rmse": st.get("rmse_dist_px", 0),
            "mean_ang": st.get("mean_error_angle_deg", 0),
            "mean_ms": st.get("mean_time_ms", 0),
        })
    with LOCK:
        STATE["running"] = False
        STATE["done"] = True
        STATE["engine"] = engine
        STATE["overall"] = stats
        STATE["rows"] = rows
        STATE["preset_label"] = (
            f"测试完成：共 {stats.get('total', 0)} 次，"
            f"合格 {stats.get('pass_count', 0)} 次，"
            f"合格率 {stats.get('pass_rate', 0) * 100:.2f}%"
        )


def _on_error(err: str):
    with LOCK:
        STATE["running"] = False
        STATE["done"] = False
        STATE["error"] = err


class _WebBridge(QObject):
    """
    信号桥接器（必须 living 在 Qt 主线程）：
    BatchTestThread 在工作线程发出信号，若直接连接到普通 Python 函数，
    PyQt 会把调用排队到 connect 时所在线程（Flask 线程，无事件循环），
    回调将永远得不到执行。经由此 QObject 的槽，信号被排队投递到 Qt 主循环，
    与桌面 GUI 的连接机制一致。
    """
    @pyqtSlot(str, int, int)
    def on_progress(self, name, cur, total):
        _on_progress(name, cur, total)

    @pyqtSlot(object)
    def on_case(self, result):
        _on_case_done(result)

    @pyqtSlot(object)
    def on_frame(self, payload):
        _on_frame(payload)

    @pyqtSlot(object)
    def on_finished(self, engine):
        _on_finished(engine)

    @pyqtSlot(str)
    def on_error(self, err):
        _on_error(err)


BRIDGE = None  # main() 中在主线程创建


# ---------------- API ----------------

@app.get("/")
def index():
    presets = list(WorkConditionPresets.all_presets().keys())
    resp = Response(render_template_string(PAGE_HTML, presets=presets))
    # 样式迭代期禁止缓存，避免浏览器展示旧 CSS
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    return resp


@app.get("/video_feed")
def video_feed():
    def gen():
        boundary = b"--frame\r\n"
        while True:
            with LOCK:
                jpeg = STATE["jpeg"]
            if jpeg is None:
                time.sleep(0.05)
                continue
            yield (boundary +
                   b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
            time.sleep(0.04)  # ~25fps 上限
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/status")
def api_status():
    with LOCK:
        return jsonify({
            "running": STATE["running"],
            "done": STATE["done"],
            "error": STATE["error"],
            "preset_label": STATE["preset_label"],
            "progress": STATE["progress"],
            "case": STATE["case"],
            "overall": STATE["overall"],
            "rows": STATE["rows"],
        })


@app.post("/api/start")
def api_start():
    data = request_json()
    scope = data.get("scope", "current")
    preset = data.get("preset", "基准工况")
    samples = max(1, min(200, int(data.get("samples", 10))))
    random_mode = data.get("random_mode", "angle_only")

    with LOCK:
        if STATE["running"]:
            return jsonify({"ok": False, "msg": "批量测试进行中"}), 409
        win = APP_CTX["win"]
        if scope == "all":
            names = list(WorkConditionPresets.all_presets().keys())
        else:
            names = [preset if preset in WorkConditionPresets.all_presets() else "基准工况"]
        engine = SimulationTestEngine(win._run_detection_on_frame)
        th = BatchTestThread(engine, names, samples, random_mode)
        # 连接到主线程的桥接器槽：信号经 Qt 队列投递到主循环，确保回调被执行
        th.progress.connect(BRIDGE.on_progress)
        th.case_done.connect(BRIDGE.on_case)
        th.frame_tested.connect(BRIDGE.on_frame)
        th.finished_all.connect(BRIDGE.on_finished)
        th.error.connect(BRIDGE.on_error)
        STATE.update({
            "running": True, "done": False, "error": "", "engine": engine,
            "overall": None, "rows": [], "case": None, "frames_done": 0,
            "progress": {"cur": 0, "total": samples * len(names)},
            "preset_label": "批量测试进行中…",
        })
        th.start()
        STATE["thread"] = th
    return jsonify({"ok": True, "total": samples * len(names)})


@app.post("/api/stop")
def api_stop():
    with LOCK:
        th = STATE.get("thread")
        if th is not None and STATE["running"]:
            th.abort()
            return jsonify({"ok": True})
    return jsonify({"ok": False, "msg": "当前无进行中的批量测试"})


def request_json():
    from flask import request
    try:
        return request.get_json(force=True, silent=True) or {}
    except Exception:
        return {}


APP_CTX = {"win": None}


# ---------------- 页面（与桌面 GUI 同风格的深色主题） ----------------

PAGE_HTML = r"""
<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>模拟测试 Web 控制台</title>
<style>
  :root { --bg:#05080F; --line:#0066CC; --txt:#E0E6F0;
          --sub:#7E93B8; --cyan:#00FFFF; --green:#00FF96; --red:#FF5566;
          --orange:#FF9600; --yellow:#FFD24D; }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:var(--bg); color:var(--txt); font:13px Consolas,"Microsoft YaHei",monospace; padding:18px; }
  /* 页头：与桌面 QTabBar::tab 同质感 */
  h1 { display:inline-block; color:var(--cyan); font:bold 15px Consolas,"Microsoft YaHei";
       background:rgba(0,50,100,.59); border:2px solid var(--line); border-radius:6px;
       padding:10px 15px; margin-bottom:8px; }
  .sub { color:var(--sub); font-size:12px; margin-bottom:14px; }
  /* 分组框：与桌面 QGroupBox 一致（2px #0066CC 边框、8px 圆角；标题浮于边框线上） */
  .panel { position:relative; background:rgba(10,20,40,.85); border:2px solid var(--line);
           border-radius:8px; padding:18px 14px 14px; margin:16px 0 14px; }
  /* 标题本身透明底；::before 仅在边框线位置画一条页面底色细条遮断边框
     （QGroupBox 经典效果），避免在框内留下整块黑带。
     CJK 字形视觉中心高于 em 框中心约 3px，故文字整体下移 3px，
     遮断条反向偏移 3px 独立对齐边框线 */
  .panel h2 { position:absolute; top:-8px; left:10px; z-index:0;
              padding:0 6px; color:var(--cyan); font:bold 13px Consolas,"Microsoft YaHei";
              background:transparent; }
  .panel h2::before { content:""; position:absolute; z-index:-1; left:0; right:0;
                      top:50%; height:4px; transform:translateY(calc(-50% - 3px));
                      background:var(--bg); }
  /* 字段标签：#9FD8FF；单选/复选文字：#00FFFF（与模拟页容器样式一致） */
  label { color:#9FD8FF; font:13px Consolas,"Microsoft YaHei"; margin-right:6px; }
  label.opt { color:var(--cyan); }
  /* 输入控件：与桌面 QComboBox/QSpinBox 一致（半透明蓝底 + 2px #0088FF 边框） */
  select, input[type=number] {
      background:rgba(0,50,100,.59); color:var(--cyan); border:2px solid #0088FF;
      border-radius:6px; padding:6px 8px; font:13px Consolas,"Microsoft YaHei";
      min-width:150px; min-height:28px; }
  select:focus, input[type=number]:focus { border-color:var(--cyan); outline:none; }
  input[type=number] { min-width:90px; }
  .row { display:flex; align-items:center; gap:14px; margin-bottom:10px; flex-wrap:wrap; }
  input[type=radio] { accent-color:var(--cyan); width:15px; height:15px; }
  /* 按钮：与桌面 QPushButton 一致（半透明蓝底、青色文字、2px 语义色描边） */
  button { background:rgba(0,50,100,.59); color:var(--cyan); border:2px solid #00AA66;
           border-radius:6px; padding:8px 22px; font:bold 13px Consolas,"Microsoft YaHei";
           cursor:pointer; min-height:28px; }
  button:hover:enabled { background:rgba(0,100,200,.78); }
  button:disabled { background:rgba(50,50,50,.59); color:#666666; border-color:#333333;
                    cursor:not-allowed; }
  #btn-stop { border-color:#FF6666; }
  /* 进度条：与桌面 QProgressBar 一致（#0A1428 底、4px 圆角、白色居中文字） */
  .bar { height:18px; background:#0A1428; border:1px solid var(--line); border-radius:4px;
         overflow:hidden; text-align:center; }
  .bar > div { height:100%; width:0%; background:#00AA66; transition:width .2s;
               color:#fff; font:12px Consolas,"Microsoft YaHei"; line-height:18px; }
  #progress-label { color:var(--sub); font:12px Consolas,"Microsoft YaHei"; margin-top:6px; }
  /* 预览框/统计框：与桌面 sim_preview、整体统计面板一致（#0A1428 底 + #0066CC 边框） */
  #preview { width:800px; max-width:100%; border:2px solid var(--line); border-radius:8px;
             background:#0A1428; display:block; }
  #case-info { margin-top:10px; font:12.5px Consolas,"Microsoft YaHei"; line-height:1.7;
               color:#9FD8FF; min-height:88px; }
  #overall { background:#0A1428; border:1px solid var(--line); border-radius:6px;
             padding:10px; font:13px Consolas,"Microsoft YaHei"; line-height:1.8;
             color:var(--txt); min-height:52px; }
  /* 表格：与桌面 QTableWidget 一致（青色文字、表头半透明蓝底） */
  table { width:100%; border-collapse:collapse; font:12.5px Consolas,"Microsoft YaHei"; margin-top:8px; color:var(--cyan); }
  th, td { border:1px solid #12315E; padding:6px 8px; text-align:center; }
  th { color:var(--cyan); background:rgba(0,50,100,.78); border:1px solid var(--line);
       font:bold 12.5px Consolas,"Microsoft YaHei"; padding:4px; }
  .legend { color:var(--sub); font-size:12px; margin-top:6px; }
  .legend b.c { color:#00FFFF; } .legend b.g { color:var(--green); } .legend b.r { color:var(--red); }
</style>
</head>
<body>
  <h1>模拟测试 Web 控制台</h1>
  <div class="sub">Playwright 可视化自动化入口 · 与桌面「模拟测试」页同风格同引擎（复用批量线程与检测逻辑）</div>

  <div class="panel">
    <h2>① 批量自动化测试（需先在基准工况设置好模板）</h2>
    <div class="row">
      <label class="opt"><input type="radio" name="scope" value="current" checked> 仅当前工况</label>
      <label class="opt"><input type="radio" name="scope" value="all"> 全部工况</label>
      <label>工况:</label>
      <select id="preset">
        {% for p in presets %}<option value="{{ p }}">{{ p }}</option>{% endfor %}
      </select>
    </div>
    <div class="row">
      <label>每工况样本数:</label>
      <input type="number" id="samples" min="1" max="200" value="10">
      <label>随机模式:</label>
      <select id="random-mode">
        <option value="angle_only">仅角度随机（标准位）</option>
        <option value="all">位置+角度随机</option>
        <option value="position_only">仅位置随机</option>
        <option value="fixed">固定不随机</option>
      </select>
    </div>
    <div class="row">
      <button id="btn-start" onclick="startBatch()">▶ 开始批量测试</button>
      <button id="btn-stop" onclick="stopBatch()" disabled>■ 停止</button>
    </div>
    <div class="bar"><div id="bar-fill">0%</div></div>
    <div id="progress-label">等待开始批量测试</div>
  </div>

  <div class="panel">
    <h2>② 真实值 / 检测值 实时对比（批量过程逐帧）</h2>
    <img id="preview" alt="实时预览">
    <div class="legend"><b class="c">青色圆圈十字=真实位姿</b>　<b class="g">绿色=检测合格</b>　<b class="r">红色=超差/漏检</b></div>
    <div id="case-info">当前用例：—</div>
  </div>

  <div class="panel">
    <h2>③ 测试统计</h2>
    <div id="overall">整体统计：尚未执行批量测试</div>
    <table id="stats-table">
      <thead><tr>
        <th>工况</th><th>检出率</th><th>合格率(考核)</th><th>平均位置误差(px)</th>
        <th>RMSE(px)</th><th>平均角度误差(°)</th><th>平均耗时(ms)</th>
      </tr></thead>
      <tbody></tbody>
    </table>
  </div>

<script>
function rateColor(rate){ return rate>=99.5?'#00FF96':(rate>=95?'#FFD24D':'#FF5566'); }
function fmt(v,d=3){ return (v==null)?'--':Number(v).toFixed(d); }

async function startBatch(){
  const scope=document.querySelector('input[name=scope]:checked').value;
  const body={scope:scope, preset:document.getElementById('preset').value,
              samples:parseInt(document.getElementById('samples').value||'10'),
              random_mode:document.getElementById('random-mode').value};
  const r=await fetch('/api/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const j=await r.json();
  if(!j.ok){ alert(j.msg||'启动失败'); return; }
  document.getElementById('btn-start').disabled=true;
  document.getElementById('btn-stop').disabled=false;
  document.getElementById('case-info').textContent='当前用例：—';
  document.getElementById('overall').textContent='批量测试进行中…';
}

async function stopBatch(){
  await fetch('/api/stop',{method:'POST'});
}

function renderCase(c){
  if(!c){ document.getElementById('case-info').textContent='当前用例：—'; return; }
  let html=`工况：${c.preset}　用例#${c.case_id??''}　判定：<span style="color:${c.vcolor};font-weight:bold">${c.verdict}</span><br>`
          +`真值 dx/dy/角度：${fmt(c.truth?.[0],1)} / ${fmt(c.truth?.[1],1)} / ${fmt(c.truth?.[2],2)}°`;
  if(c.det){
    html+=`<br>检测 dx/dy/角度：${fmt(c.det[0],1)} / ${fmt(c.det[1],1)} / ${fmt(c.det[2],2)}°`
         +`<br>位置误差：<b>${fmt(c.err_dist,2)}px</b>　角度误差：${fmt(Math.abs(c.err_angle),2)}°　耗时：${fmt(c.elapsed,1)}ms`;
  }
  document.getElementById('case-info').innerHTML=html;
}

function renderOverall(s){
  const el=document.getElementById('overall');
  if(!s||!s.total){ el.textContent='整体统计：暂无数据'; return; }
  const pr=s.pass_rate*100, dr=s.detection_rate*100, col=rateColor(pr);
  el.innerHTML=`<b style="color:#00FFFF">整体统计结果</b>　总用例 <b>${s.total}</b>`
    +`　检出 <b>${s.detected}</b>（${dr.toFixed(1)}%）`
    +`　漏检 <b style="color:#FF5566">${s.miss_count}</b>`
    +`　检出超差 <b style="color:#FF9600">${s.out_of_tol_count}</b>`
    +`　合格 <b style="color:${col}">${s.pass_count}</b>`
    +`　<b style="color:${col}">考核合格率 ${pr.toFixed(2)}%</b><br>`
    +`平均位置误差 <b>${fmt(s.mean_error_dist_px)}px</b>　最大 <b>${fmt(s.max_error_dist_px)}px</b>`
    +`　RMSE <b>${fmt(s.rmse_dist_px)}px</b>　平均角度误差 <b>${fmt(s.mean_error_angle_deg)}°</b>`
    +`　最大 <b>${fmt(s.max_error_angle_deg)}°</b>　平均耗时 <b>${fmt(s.mean_time_ms,2)}ms</b>`;
}

function renderRows(rows){
  const tb=document.querySelector('#stats-table tbody');
  tb.innerHTML='';
  for(const r of rows){
    const tr=document.createElement('tr');
    tr.innerHTML=`<td>${r.name}</td>`
      +`<td>${r.det_rate.toFixed(1)}% (${r.det_text})</td>`
      +`<td style="color:${rateColor(r.pass_rate)}">${r.pass_rate.toFixed(2)}% (${r.pass_text})</td>`
      +`<td>${fmt(r.mean_err)}</td><td>${fmt(r.rmse)}</td><td>${fmt(r.mean_ang)}</td><td>${fmt(r.mean_ms,2)}</td>`;
    tb.appendChild(tr);
  }
}

async function poll(){
  try{
    const s=await (await fetch('/api/status')).json();
    const total=s.progress?.total||0, cur=s.progress?.cur||0;
    const pct= total? Math.round(cur/total*100):0;
    document.getElementById('bar-fill').style.width=pct+'%';
    document.getElementById('bar-fill').textContent=pct+'%';
    document.getElementById('progress-label').textContent=s.preset_label||'等待开始批量测试';
    renderCase(s.case);
    if(s.overall) renderOverall(s.overall);
    if(s.rows&&s.rows.length) renderRows(s.rows);
    const running=s.running&&!s.error;
    document.getElementById('btn-start').disabled=running;
    document.getElementById('btn-stop').disabled=!running;
    if(s.error) document.getElementById('progress-label').textContent='批量测试错误：'+s.error;
  }catch(e){}
}
setInterval(poll,400);
poll();

// MJPEG 为无限流：必须在文档 load 完成后再挂载，
// 否则 window.load/document.fonts.ready 永不就绪（阻塞截图等自动化操作）
function startFeed(){
  const img=document.getElementById('preview');
  if(img && !img.getAttribute('src')) img.src='/video_feed';
}
if(document.readyState==='complete'){ startFeed(); }
else{ window.addEventListener('load', startFeed); }
</script>
</body>
</html>
"""


def main():
    global BRIDGE
    qt_app = QApplication(sys.argv)
    win = G.MainWindow()
    win.showMinimized()
    _setup_templates(win)
    APP_CTX["win"] = win
    BRIDGE = _WebBridge()  # 在 Qt 主线程创建，保证信号排队投递到主循环

    th = threading.Thread(
        target=lambda: app.run(host=HOST, port=PORT, threaded=True,
                               debug=False, use_reloader=False),
        daemon=True)
    th.start()
    print(f"模拟测试 Web 控制台已启动：http://{HOST}:{PORT}")
    sys.exit(qt_app.exec_())


if __name__ == "__main__":
    main()
