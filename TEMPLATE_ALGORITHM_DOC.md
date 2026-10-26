# 双直线模板与外形模板算法原理及设置过程文档

## 目录
1. 概述
2. 双直线模板
   2.1 算法原理
   2.2 设置过程
   2.3 数据结构
3. 外形模板
   3.1 算法原理
   3.2 设置过程
   3.3 数据结构
4. 边缘检测算法详解
5. 模板匹配算法
6. 动态检测流程

---

## 1. 概述

本视觉定位系统支持两种模板类型：

| 模板类型 | 适用场景 | 核心特征 |
|---------|---------|---------|
| 双直线模板 | 矩形面纸定位 | 每侧两条垂直线（长边+短边） |
| 外形模板 | 异形轮廓识别 | 任意形状轮廓匹配 |

两种模板均遵循"先设置模板，再进行检测"的工作流程。

---

## 2. 双直线模板

### 2.1 算法原理

双直线模板基于**边缘检测+直线提取**的计算机视觉技术，用于精确定位矩形面纸的四个角点。

#### 2.1.1 核心算法流程

```
输入图像 → ROI截取 → 灰度化 → 高斯模糊 → Canny边缘检测 → Hough直线检测 → 直线匹配
```

#### 2.1.2 算法步骤详解

**步骤1：ROI区域截取**
```python
x1, y1, x2, y2 = roi  # 用户定义的检测区域
roi_img = frame[y1:y2, x1:x2]
```

**步骤2：灰度化**
- 将彩色图像转换为灰度图像，减少计算量
- 公式：`gray = 0.299*R + 0.587*G + 0.114*B`

**步骤3：高斯模糊**
```python
blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
```
- 作用：平滑图像，减少噪声干扰
- 参数：5x5高斯核，标准差1.4

**步骤4：Canny边缘检测**
```python
edges = cv2.Canny(blurred, canny_low, canny_high)
```
- **算法原理**：
  1. 计算图像梯度（Sobel算子）
  2. 非极大值抑制（保留边缘局部最大值）
  3. 双阈值检测（高阈值确定强边缘，低阈值连接弱边缘）

- **阈值说明**：
  - `canny_low`（低阈值）：低于此值的像素被抑制
  - `canny_high`（高阈值）：高于此值的像素被确认为边缘

**步骤5：Hough直线检测**
```python
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold, 
                        minLineLength=30, maxLineGap=10)
```
- **参数说明**：
  - `rho=1`：极坐标距离分辨率（像素）
  - `theta=np.pi/180`：角度分辨率（弧度）
  - `threshold`：累加器阈值
  - `minLineLength`：最小线段长度
  - `maxLineGap`：线段允许的最大间隙

### 2.2 设置过程

#### 2.2.1 设置流程图

```
打开摄像头 → 选择双直线模板 → 设置左侧长边 → 设置左侧短边 → 
设置左侧ROI → 设置右侧长边 → 设置右侧短边 → 设置右侧ROI → 
保存模板 → 手动测试
```

#### 2.2.2 详细步骤

**步骤1：启动摄像头**
- 在左侧面板选择"摄像头"标签页
- 选择可用摄像头并点击"打开摄像头"

**步骤2：选择双直线模板**
- 点击右侧面板的"双直线模板"标签

**步骤3：设置左侧模板**
1. 点击"框选长边"按钮
2. 在图像上拖动鼠标框选左侧长边区域
3. 系统自动检测并显示检测到的直线
4. 点击"框选短边"按钮，重复上述操作
5. 点击"设置左监测区域"，框选左侧检测区域

**步骤4：设置右侧模板**
- 重复步骤3，设置右侧的长边、短边和监测区域

**步骤5：验证模板完整性**
- 模板状态显示"完成"表示所有设置已完成
- 需设置：左长边、左短边、左ROI、右长边、右短边、右ROI

**步骤6：保存模板**
- 点击"保存当前模板"按钮
- 可选择保存路径和文件名

**步骤7：手动测试**
- 点击"手动测试"按钮启动实时检测
- 检测区域和边缘线会随纸张移动而动态更新

### 2.3 数据结构

```python
@dataclass
class DualLineTemplate:
    left_line_long: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
    left_line_short: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
    right_line_long: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
    right_line_short: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
    left_roi: Optional[Tuple[float, float, float, float]] = None  # (x1, y1, x2, y2)
    right_roi: Optional[Tuple[float, float, float, float]] = None
    valid: bool = False
```

**直线表示**：`((x1, y1), (x2, y2))` - 直线的两个端点坐标

**ROI表示**：`(x1, y1, x2, y2)` - 矩形区域的左上角和右下角坐标

---

## 3. 外形模板

### 3.1 算法原理

外形模板基于**轮廓提取+模板匹配**技术，用于识别不规则形状的物体。

#### 3.1.1 核心算法流程

```
输入图像 → ROI截取 → 灰度化 → 高斯模糊 → Canny边缘检测 → 
轮廓提取 → 轮廓特征计算 → 模板匹配
```

#### 3.1.2 轮廓提取算法

**步骤1：边缘检测**
- 使用Canny算子检测图像边缘

**步骤2：轮廓提取**
```python
contours, hierarchy = cv2.findContours(
    binary_image, 
    cv2.RETR_EXTERNAL,  # 只检测最外层轮廓
    cv2.CHAIN_APPROX_SIMPLE  # 压缩轮廓点
)
```

**步骤3：轮廓特征计算**
- **面积**：`cv2.contourArea(contour)`
- **周长**：`cv2.arcLength(contour, closed=True)`
- **矩**：`cv2.moments(contour)` - 用于计算质心
- **Hu矩**：`cv2.HuMoments(moments)` - 用于形状匹配

**步骤4：模板匹配**
```python
match_score = cv2.matchShapes(template_contour, target_contour, 
                               cv2.CONTOURS_MATCH_I1, 0)
```
- 匹配分数越小，相似度越高
- 通常阈值设为0.7，小于此值认为匹配成功

### 3.2 设置过程

#### 3.2.1 设置流程图

```
打开摄像头 → 选择外形模板 → 设置左侧轮廓 → 设置左侧ROI → 
设置右侧轮廓 → 设置右侧ROI → 保存模板 → 手动测试
```

#### 3.2.2 详细步骤

**步骤1：启动摄像头**
- 在左侧面板选择"摄像头"标签页
- 选择可用摄像头并点击"打开摄像头"

**步骤2：选择外形模板**
- 点击右侧面板的"外形模板"标签

**步骤3：设置参数（可选）**
- **Canny低阈值**：调整边缘检测灵敏度（范围：10-200）
- **Canny高阈值**：调整边缘确认阈值（范围：50-300）
- **匹配阈值**：轮廓匹配相似度阈值（范围：0.1-1.0）

**步骤4：设置左侧轮廓**
- **方式一：自动识别**
  1. 点击"自动识别"按钮
  2. 系统自动检测左侧区域的轮廓
  
- **方式二：手绘轮廓**
  1. 点击"手绘左轮廓"按钮
  2. 在图像上用鼠标绘制轮廓
  3. 双击完成绘制

**步骤5：设置左侧区域**
- **方式一：自动识别区域**
  1. 点击"自动识别区域"按钮
  
- **方式二：手绘区域**
  1. 点击"手绘左区域"按钮
  2. 拖动鼠标绘制矩形区域

**步骤6：设置右侧轮廓和区域**
- 重复步骤4-5，设置右侧的轮廓和监测区域

**步骤7：保存模板**
- 点击"保存当前模板"按钮
- 可选择保存路径和文件名

**步骤8：手动测试**
- 点击"手动测试"按钮启动实时检测

### 3.3 数据结构

```python
@dataclass
class ShapeTemplate:
    left_contour: Optional[np.ndarray] = None  # 轮廓点集 (N, 1, 2)
    right_contour: Optional[np.ndarray] = None
    left_roi: Optional[Tuple[float, float, float, float]] = None
    right_roi: Optional[Tuple[float, float, float, float]] = None
    center: Tuple[float, float] = (0, 0)      # 轮廓中心点
    angle: float = 0.0                        # 轮廓角度
    area: float = 0.0                         # 轮廓面积
    name: str = ""                            # 模板名称
    valid: bool = False
```

**轮廓表示**：`np.ndarray` 形状为 `(N, 1, 2)`，包含N个轮廓点的坐标

---

## 4. 边缘检测算法详解

系统支持多种边缘检测算法，可通过算法选择下拉菜单切换：

### 4.1 算法对比

| 算法名称 | 特点 | 适用场景 | 参数 |
|---------|------|---------|------|
| **canny_standard** | 标准Canny检测 | 一般场景 | canny_low=50, canny_high=150 |
| **canny_low** | 低阈值，更敏感 | 弱边缘检测 | canny_low=30, canny_high=100 |
| **sobel** | Sobel梯度算子 | 快速检测 | threshold=50 |
| **scharr** | Scharr算子，更精确 | 高精度检测 | threshold=40 |
| **fusion** | 多算法融合 | 复杂场景 | 自动选择最优 |

### 4.2 算法实现

#### 4.2.1 Canny标准算法
```python
def detect_lines_canny_standard(gray, offset_x, offset_y, canny_low=50, canny_high=150):
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
    edges = cv2.Canny(blurred, canny_low, canny_high)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 40, minLineLength=25, maxLineGap=10)
    # 转换坐标并过滤短线条...
```

#### 4.2.2 Sobel算法
```python
def detect_lines_sobel(gray, offset_x, offset_y, threshold=50):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edges = cv2.magnitude(sobelx, sobely)
    _, binary = cv2.threshold(edges, threshold, 255, cv2.THRESH_BINARY)
    lines = cv2.HoughLinesP(binary, 1, np.pi/180, 35, minLineLength=25, maxLineGap=10)
```

#### 4.2.3 Scharr算法
```python
def detect_lines_scharr(gray, offset_x, offset_y, threshold=40):
    scharrx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
    scharry = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
    edges = cv2.magnitude(scharrx, scharry)
    _, binary = cv2.threshold(edges, threshold, 255, cv2.THRESH_BINARY)
    lines = cv2.HoughLinesP(binary, 1, np.pi/180, 35, minLineLength=25, maxLineGap=10)
```

---

## 5. 模板匹配算法

### 5.1 直线匹配算法

```python
def find_best_matching_lines(detected_lines, template_lines):
    matched_lines = []
    for template_line in template_lines:
        # 计算模板直线特征
        template_angle = calculate_line_angle(template_line)
        template_length = calculate_line_length(template_line)
        template_center = calculate_line_center(template_line)
        
        best_match = None
        best_score = float('inf')
        
        for detected_line in detected_lines:
            # 计算检测直线特征
            detected_angle = calculate_line_angle(detected_line)
            detected_length = calculate_line_length(detected_line)
            detected_center = calculate_line_center(detected_line)
            
            # 综合评分（角度权重更高）
            angle_diff = min(abs(detected_angle - template_angle), 360 - abs(detected_angle - template_angle))
            length_diff = abs(detected_length - template_length) / max(template_length, 1)
            dist = distance(detected_center, template_center)
            
            score = angle_diff * 0.5 + length_diff * 0.3 + dist * 0.01
            
            if score < best_score and score < 50:
                best_score = score
                best_match = detected_line
        
        if best_match:
            matched_lines.append(best_match)
    
    return matched_lines
```

### 5.2 匹配评分公式

```
score = angle_diff × 0.5 + length_diff × 0.3 + distance × 0.01
```

| 特征 | 权重 | 说明 |
|-----|------|------|
| 角度差异 | 0.5 | 最重要，角度匹配是直线识别的关键 |
| 长度差异 | 0.3 | 次要因素，允许一定的长度变化 |
| 位置距离 | 0.01 | 最不重要，允许位置漂移 |

---

## 6. 动态检测流程

### 6.1 检测流程图

```
开始检测 → 获取当前帧 → 检查模板完整性 → 计算模板中心点 → 
使用模板区域检测 → 模板匹配 → 计算偏移量 → 变换检测区域 → 
二次检测 → 合并结果 → 更新显示 → 计算机械臂偏移 → 同步UI
```

### 6.2 关键步骤说明

**步骤1：偏移量计算**
```python
# 计算检测中心点
det_center_x = mean(line_center_x for line in detected_lines)
det_center_y = mean(line_center_y for line in detected_lines)

# 计算偏移
dx = det_center_x - template_center_x
dy = det_center_y - template_center_y
```

**步骤2：区域变换**
```python
# 模板区域随纸张移动
x1, y1, x2, y2 = template_roi
transformed_roi = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
```

**步骤3：实时数据同步**
- 更新机械臂偏移参数（X/Y/Z/A）
- 更新实时检测数据（角度、长度、中心点）
- 更新UI显示

### 6.3 相对位置保持原则

1. **区域与纸张相对位置不变**：检测区域始终覆盖纸张对应位置
2. **区域与边缘线相对位置不变**：线条在区域内的相对坐标关系稳定
3. **检测与变换同时进行**：确保实时性和准确性

---

## 附录：坐标系统说明

### A.1 图像坐标系
- 原点：左上角
- X轴：向右为正
- Y轴：向下为正

### A.2 物理坐标系（标定后）
- 原点：标定板左下角
- X轴：向右为正
- Y轴：向上为正
- 单位：毫米（mm）

### A.3 像素-物理转换
```python
# 标定后通过像素/毫米比例转换
pixel_per_mm_x = fx / work_distance
pixel_per_mm_y = fy / work_distance

x_mm = x_pixel / pixel_per_mm_x
y_mm = y_pixel / pixel_per_mm_y
```

---

**文档版本**: v1.0  
**创建日期**: 2026-05-24  
**适用系统**: 视觉定位系统 v3.x