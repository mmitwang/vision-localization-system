# Web 控制台样式与桌面 GUI 统一 — 修复说明

> 适用文件：`web_sim_server.py`（模拟测试 Web 控制台，http://127.0.0.1:5188）
> 对标对象：`vision_gui_v3.py` 桌面端「模拟测试」页
> 更新日期：2026-10-08

---

## 一、问题现象

Web 控制台参数设置区域与桌面 GUI 视觉不一致；样式代码修改后用户浏览器仍显示旧样式，经多轮反馈未收敛。

## 二、根因

1. **运行实例未同步**：长生命周期的服务进程仍在运行旧代码。Playwright 自动化脚本每次自启新服务，掩盖了用户手动访问的旧实例问题。
2. **浏览器缓存**：页面未设置禁缓存响应头，旧 CSS 被浏览器复用。
3. **样式细节缺失**：控件质感（底色/边框/字色）、分组标题压线效果与桌面 QGroupBox 不一致。

## 三、桌面 → Web 样式 Token 对照表

| 控件 | 桌面 Qt 样式（来源） | Web CSS 实现 |
|------|---------------------|--------------|
| 页面底色 | `#05080F` | `body { background:#05080F }` |
| 分组框 | `QGroupBox`：2px `#0066CC`、圆角 8px、`rgba(10,20,40)` 底 | `.panel` 同值；标题浮于边框线上（见第四节） |
| 分组标题 | 青色加粗 Consolas，压在边框线上 | `.panel h2` 绝对定位 + 边框遮断 |
| 字段标签 | `QLabel`：`#9FD8FF` 13px Consolas | `label { color:#9FD8FF }` |
| 单选/复选文字 | `QRadioButton`：`#00FFFF` | `label.opt { color:#00FFFF }`，指示器 15px |
| 下拉框/数字框 | `rgba(0,50,100,150)` 底、2px `#0088FF` 边框、青字、圆角 6px、min-height 28px | `select,input[type=number]` 同值；`:focus` 边框转 `#00FFFF` |
| 开始按钮 | 蓝底 + 2px 绿边 `#00AA66` + 青字，hover 提亮 | `#btn-start`（`#btn-stop` 红边 `#FF6666`） |
| 禁用按钮 | 灰底 `rgba(50,50,50,150)`、灰字 `#666`、边框 `#333` | `button:disabled` 同值 |
| 进度条 | `QProgressBar`：`#0A1428` 底、1px `#0066CC`、圆角 4px、chunk `#00AA66` | `.bar` / `.bar>div` 同值 |
| 预览/统计面板 | `sim_preview`：`#0A1428` 底、2px `#0066CC`、圆角 8px | `#preview` / `#overall` 同值 |
| 表格 | `QTableWidget`：青字、表头 `rgba(0,50,100)` 底 | `table` 青字、`th` 半透明蓝底 |
| 字体 | Consolas 优先 | `Consolas,"Microsoft YaHei"` |
| 分组命名 | ① 批量自动化测试（需先在基准工况设置好模板）/ ② 真实值 / 检测值 实时对比 / ③ 测试统计 | Web 三个面板标题同名；标签统一半角冒号 |

按钮语义色约定（全局）：绿 `#00AA66`=启动、红 `#FF6666`=停止、蓝 `#0088FF`=描边/应用、橙 `#FF9600`=导出。

## 四、QGroupBox 标题压线 CSS 配方（关键）

桌面 QGroupBox 标题浮于边框线上、线在文字处断开。Web 实现要点：

```css
.panel {
  position: relative;
  border: 2px solid #0066CC;
  border-radius: 8px;
  padding: 18px 14px 14px;
  margin: 16px 0 14px;
}
.panel h2 {
  position: absolute;
  top: -8px;            /* CJK 字形视觉中心高于 em 框中心约 3px，需下移补偿 */
  left: 10px;
  background: transparent;   /* 标题透明底，不可整块涂黑，否则框内留黑带 */
}
.panel h2::before {          /* 仅在边框线处画细条遮断边框 */
  content: "";
  position: absolute;
  left: 0; right: 0;
  top: 50%;
  height: 4px;
  transform: translateY(calc(-50% - 3px));  /* 与文字下移量反向，独立对齐边框线 */
  background: #05080F;       /* = 页面底色，不是面板底色 */
}
```

注意：
- 遮断条高度取 4px（边框 2px，略宽以消除抗锯齿残影）；
- 文字下移补偿值随字体族/字号变化，换字体后需用 `Range.getBoundingClientRect()` 重新测量字形中心与 `panel.getBoundingClientRect().top` 的差值。

## 五、运行实例与缓存同步（改完样式必做）

```powershell
# 1. 杀掉旧服务实例
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -like "*web_sim_server*" } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force }

# 2. 以新代码重启
Start-Process python -ArgumentList "web_sim_server.py" -WorkingDirectory "<工程目录>" -WindowStyle Minimized
```

服务端已对首页响应设置（`web_sim_server.py` 的 `index()`）：

```python
resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
resp.headers["Pragma"] = "no-cache"
```

浏览器端如仍见旧样式：Ctrl+F5 强制刷新。

## 六、验证清单

1. 编译：`python -m py_compile web_sim_server.py`
2. 真实浏览器打开页面逐项核对第三节对照表（分组标题、标签色、输入框质感、按钮描边、进度条）；
3. 放大标题区域确认：蓝线穿过文字视觉中心、文字处边框干净断开、框内无黑色横带；
4. 功能回归：`python playwright_sim_test.py`，两场景考核合格率均应为 100%（样式改动不得影响批量逻辑）；
5. 产物：`sim_ui_report/` 下截图（参数设置/进行中/结果）与 `自动化过程录像.webm`。

## 七、最近一次验证结果（2026-10-08）

- 场景A 基准工况 ×30（仅角度随机）：检出 30/30，合格率 100.00%
- 场景B 全部 11 种工况 ×5（共 55 例）：检出 55/55，合格率 100.00%
- 真实浏览器样式逐项检查：PASS
