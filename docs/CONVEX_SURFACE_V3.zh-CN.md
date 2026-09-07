# convex_surface：凸包外部避障碰撞球

本轮新增独立 `convex-surface-v3-experimental`，不替换 material-v1 或 joint-v2。
目标为抓取后外部搬运避障：先求真实输入几何的凸包，再在凸包表面放球。
不保留空腔、不要求球心位于原材料内，也不运行旧材料深度／空腔自由空间评估。

**默认最多 64 球，可降到 32；半径策略与数量无关。稀疏球并集并不等于完整凸包，不保证局部零漏检。**
下面给出实际数值、失败案例、原生 cuRobo 验证与人工查看命令。

## 1. 直接看结果

```bash
conda activate fastsim_vnext
cd asset-collision-spheres  # 从本地工作区根目录进入
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

关闭一个 Isaac 窗口后再打开下一个；静态预览不需要按 Play。

### 方盒：上限 64，实际 56 球

```bash
python -m asset_collision_spheres.preview.inspect \
  --usd outputs/convex_v3/box/E_auto_64/preview.usda \
  --display convex_spheres --view close
```

### 碗：上限 64，实际 64 球

```bash
python -m asset_collision_spheres.preview.inspect \
  --usd outputs/convex_v3/bowl/E_auto_64/preview.usda \
  --display convex_spheres --view close
```

上面两条命令将路径中的 `E_auto_64` 改为 `E_auto_32` 即可看低预算结果。
碗为 32 球，box_167 为 26 球；文件后缀表示上限，不保证实际数量相同。

### 杯子：机器人 attach 后的原生 64 球

```bash
python -m asset_collision_spheres.adapters.workspace.mug_auto \
  --method convex_surface --geometry-config configs/demo/convex_surface.json \
  --diagnostic-budget 64 --output outputs/convex_v3/native_mug_64 \
  --open-existing --hide-visual --view close
```

预览球读取的是 native kinematics 实际槽位经过 FK 的结果。青色是抓后凸包，紫色仍是原位置 mesh。
去掉 `--hide-visual` 显示杯子实体。这里不是动态抓取或完整 1e 任务回放。

32 球版本：后端仍保留 64 个槽位，只有 32 个激活。

```bash
python -m asset_collision_spheres.adapters.workspace.mug_auto \
  --method convex_surface --geometry-config configs/demo/convex_surface_32.json \
  --diagnostic-budget 64 --output outputs/convex_v3/native_mug_32 \
  --open-existing --hide-visual --view close
```

### 看布点原因

单资产 `preview.inspect` 的 `--display` 可改为：

- `convex_features`：合并平面、真实棱线、闭合轮廓、拟合轴／proxy 和采样点。
- `convex_hull`：凸包实体。
- `convex_sites`：小采样点、凸包线框和轮廓。
- `convex_spheres`：最终球与凸包线框。

红色=网格角来源，青色=棱来源，金色=闭合轮廓来源，绿色=网格面来源，蓝色=通用采样。
这些是调试来源，不是人工语义标签。尤其“网格角”是映射后的参数化角位置，不承诺原 hull 具有理想尖角。
可在 Stage 面板中单独切换 `/World/ReferenceMesh`、`HullSolid`、`HullWire`、`Patches`、
`HullFeatures`、`Proxy`、`Sites`、`Spheres`。

已实际渲染并检查的截图：

- [方盒 56 球](../outputs/convex_v3/box/E_auto_64/close.png)
- [碗 64 球](../outputs/convex_v3/bowl/E_auto_64/close.png)
- [机器人持杯，原生 64 球](../outputs/convex_v3/native_mug_64/close.png)
- [碗分区／轮廓顶视图](../outputs/convex_v3/user_bowl_32/features_top.png)

## 2. 用户配置和重新生成

生成器最简配置：

```json
{"mode": "convex_surface"}
```

可选字段只有：

| 字段 | 默认 | 作用 |
| --- | --- | --- |
| `shape_hint` | `auto` | `auto`／`open_box`／`generic`；帮助选择网格或通用策略 |
| `max_spheres` | `64` | 正整数 1–64；限制数量，不强制用满 |
| `seed` | `5` | 复现参考采样、FPS 和消融结果 |

`backend_sphere_capacity` 仅由实际适配层注入，不是日常资产或 Mission 配置字段。
旧模式的 `sphere_budget`、`outward_offset_mode`、`system_policy`、`radius_mode` 等不能混入新配置。
没有每类别半径、杯口方向、面框、rim 配额或棱线标注。

```json
{"mode": "convex_surface", "max_spheres": 32}
```

改变上限会重新布点；32 球不保证是 64 球的子集，但同一几何的半径完全相同。

### 任意现有 USD preset：只复用资产输入

```bash
# 默认凸包策略；覆盖为最多 32 球。旧 preset 的 generation 部分不参与。
python -m asset_collision_spheres.preview.convex \
  --preset configs/local/bowl/bowl_001_auto_geometry.json \
  --max-spheres 32 --output outputs/convex_v3/user_bowl_32 --build-only

python -m asset_collision_spheres.preview.convex \
  --preset configs/local/bowl/bowl_001_auto_geometry.json \
  --max-spheres 32 --output outputs/convex_v3/user_bowl_32 \
  --open-existing --display spheres --view close
```

换盒子时使用 `configs/local/box/box_167_auto_geometry.json` 并换一个输出目录。
这个新入口支持 `--config`、`--max-spheres`、`--build-only`、`--open-existing`、
`--view close|top|front`、`--display overlay|spheres|original|hull|features|sites`。
`--display original` 只看原 mesh；`overlay` 同时看原实体与球；`spheres` 隐藏实体。
拟合和已有文件核验都在独立进程完成，避免在 Kit 启动前加载独立 USD 库。

既有 `preview.mesh` 仍是旧材料预览入口；新模式请使用 `preview.convex`。
`preview.inspect` 是直接查看给定诊断 USD，不做源哈希核验；`preview.convex --open-existing` 会核验。

OBJ／PLY／STL：

```bash
asset-spheres --mesh /absolute/path/object.obj --unit-scale-m 1 \
  --config configs/demo/convex_surface.json --output outputs/my_object_convex.json
```

Python：

```python
from asset_collision_spheres import generate_convex_surface_spheres, load_mesh

mesh = load_mesh("object.obj", unit_scale_m=1, require_material=False)
result = generate_convex_surface_spheres(mesh, {"max_spheres": 32})
print(result.spheres.shape)  # [N,4]，物体局部坐标，米：cx,cy,cz,r
```

单位、USD prim 和变换必须正确。默认 loader 仍检查材料网格；仅新入口关闭旧材料合法性要求。
新 USD 加载选项对非三角面只为 debug 做扇形三角化，不把它用于凸包优化；原 USD 不改写。
凹多边形的这种 debug 三角化不等于高质量重新网格化，凸包只读取转换后的顶点。

### Mission / cuRobo

在现有 cuRobo solver config 中：

```yaml
attached_object_sphere_method: mesh
attached_object_sphere_fallback: error
attached_object_sphere_budget: 64  # 后端预留槽位
attached_object_mesh_configs:
  mug:                          # 场景对象 ID，不是类别标签
    mode: convex_surface
    max_spheres: 32              # 可省略；省略默认64
```

有效上限=min(用户上限、真实后端容量、64)。后端容量小于 64 时自动进一步收紧。
不接受 `max_spheres >64`，也不接受 `null`。旧 joint-v2 仍可使用自身上限，未全局改为64。
Mission 共享 schema 不能表达此跨字段条件，构建 cuRobo 配置和 prepare_request 时严格验证新模式，
而不是把共享 `max_spheres` schema 全局改为64破坏旧模式。

## 3. 实际调用链与修改文件

```text
资产 loader：单位／缩放／作者变换／坐标轴
  → api.generate_convex_surface_spheres
  → algorithms/hull_structure.py：凸包、平面、边界、box frame
  → algorithms/convex_surface.py：规则网格或 feature+FPS、统一半径
  → algorithms/hull_diagnostics.py：独立 hull 诊断
  → MeshRegionSphereResult，物体局部 [N,4]
  → adapters/fastsim.py（新 mode 独立分支）
  → Mission 原有 object-to-link 变换、缓存
  → AttachmentManager.update → native slots → FK／collision query
```

实际读取：工作区/Plugins AGENTS、快速指南、迁移说明、REORGANIZATION、现有 auto_geometry、
joint_selection、box_features、loader、API、preview、fastsim adapter、mug native 验证、
Mission schema/backend/config/attestation、配置与测试。cuRobo 仅阅读 AGENTS 并运行既有测试。

本仓库新增：

- `algorithms/hull_structure.py`、`convex_surface.py`、`hull_diagnostics.py`。
- `preview/convex.py`。
- `examples/freeze_convex_reference.py`、`run_convex_experiments.py`。
- `configs/demo/convex_surface.json`、`convex_surface_32.json`。
- `tests/test_convex_surface.py`、本文档。

本仓库增量修改：`api.py`、`__init__.py`、`cli.py`、`loaders/mesh.py`、`loaders/usd.py`、
`adapters/fastsim.py`、`adapters/workspace/mug_auto.py`、`mug.py`、`preview/inspect.py`、README。

Mission 必要接线修改：`backend.py` 仅增加新模式指标到旧 attestation 字段的显式映射；
`config.py` 新模式输入校验；`alignment.py` 接受 `mesh_convex_surface` 类型；
`catalog/mission.yaml` mode枚举、`index.yaml` 哈希和相关测试。
没有重写 attach/update/detach、object-to-link 或 CUDA collision kernel。

未修改旧算法函数体、task2sim 转发文件、正式 1e run、物理碰撞体、安全 margin 或 ignore。
工作树原本就有迁移及上一轮未提交修改，本轮没有回滚、提交或推送它们。

## 4. 凸包与结构检测细节

### 凸包

精确去重有效顶点；中心化、按几何半径归一化后调用 `scipy.spatial.ConvexHull`。
根据 Qhull 外向平面法线校正新凸包面的绕序，移除未参与 hull 的内部顶点。
原 mesh 的 winding、材料体积、开口和水密性不作为输入条件。

拒绝少于4点、非有限值、尺度<1e-9 m、SVD最小/最大奇异值比<1e-6、Qhull失败或不稳定体积。
不使用 QJ 随机扰动，不为共面输入制造厚度。原始 mesh 和 fingerprint 始终保留。

### 平面和边界

尺度 `L=sqrt(hull.area/(2*pi))`。
相邻三角面采用相对 seed 的10°法线约束、0.004L面距约束增长，避免沿曲面逐步漂移。
随后对相邻候选 patch 合并；整个合并组需同时满足12°法线约束、0.012L最大面距。
初次顶视检查发现0.008L把碗封口分成两块并产生伪轮廓弦，已修正并加入完整封口面积/边界转折回归。
建立真实 patch adjacency，同 patch 内三角形对角线全部抑制。

真实边界中的28°以上二面角作为 sharp-edge 候选；同 patch 对的边界按degree-two链合并，
保留真实折线，长度低于0.065L的碎片过滤。不是拟合无限直线交点，也不是原材料短桥算法。
三条独立 sharp 方向在真实 hull 顶点交汇时记录尖角；扫描圆角可能不产生8个尖角。

### box-like

从主要 patch 发现三组无符号法线，要求近似正交、各轴两侧均有面积支持。
用全部 hull facet 的面积加权法线迭代细化正交坐标系，不用世界 XYZ 或单纯 OBB 判定。
14°内轴向法线解释面积占比达到0.80时 auto 启用；`open_box` 门槛为0.66，
`generic` 禁用网格。hint 不参与半径公式或固定球数。

实际 box_167 旋转和平移前后56点双向最近邻最大误差 **2.04e-14 m**。
初版基于 patch seed 的轴曾出现约3 mm差异，已用全 hull 法线细化修正并加入实资产回归。

### 规则网格

在三个几何轴上搜索 `nx,ny,nz>=2`，只保留边界节点：

```text
N = nx*ny*nz - max(nx-2,0)*max(ny-2,0)*max(nz-2,0)
```

穷举不超过 cap 的小整数，最小化：

```text
std(log(L_axis/(n_axis-1))) + 0.28*(cap-N)/cap
```

相对间距均衡为主，预算利用率为辅；不是强制用满。
正方体上限64→4×4×4边界56点：8角、24棱内部、24面内部。
长方体自动增加长轴点数；测试0.4×0.2×0.12 m长方体使用64点，拟合轴顺序下节点3×4×6。
浅托盘上限64→2×4×7边界56点，不会在短厚度方向浪费很多层。
上限<8时不能保留全部角，退到通用布点并标记review。

所有 proxy target 用实际 hull triangle 最近点映射，球心必须在 hull 表面。
误差>0.08L的局部 target 不保留，用通用 FPS 补足相应数量，标记review；不制造理想尖角。
共享边／角只创建一次并去重。圆角盒例子进入网格后有局部替换，**没有强留假的尖角球**。
最终用凸多面体全部半空间与支持平面联合校验点在表面；避免扫描细长三角形最近点的纳米级误差。

## 5. 通用 FPS、rim 和数量

参考集由8192个按面积分配的三角面内样本加实际 hull vertices 组成，seed默认5。
普通选点按距离已选集合最远的参考点逐步加入（欧氏表面距离近似）。
主选择器不优化接触探针收益，也不使用旧材料 patch 贪心；没有无约束3D k-means。
从 hull 惯性主轴补充至多6个外轮廓极值点，保证面积采样不单独决定最外端支撑。
记录每4点的参考集最近距离 median/p95/max 曲线。
第一版未找到跨资产稳定的提前停止阈值，因此 generic 充分使用有效上限，最多64。
没有Lloyd/CVT、split/merge、半径升级或90/133球式增长。

### rim-like loop

面积≥hull总面积3.5%的近平面 patch，提取其真实闭合边界链。
长度≥0.4L，并要求至少55%的边界长度存在法线转折证据：直接二面角≥15°，
或0.065L邻域中相对 patch 法线转折≥25°。后者用于扫描资产的渐变倒角。
不要求圆、水平、Z最大，不依赖“碗／杯子”类别；底部边界也可能被识别为loop。
这不是原 mesh 的拓扑开边界。

全局特征间距 `h_feature=0.8*sqrt(hull.area/cap)`；每环候选数由 `ceil(loop_length/h_feature)` 得到。
所有特征共享 `floor(0.45*cap)` 上限，环配额不足时按请求比例缩减、余数分配；少于3点的环明确不采样。
剩余额度照顾真实尖角和长边，至少55%留给普通表面（普通极值点也可能恰好落在轮廓上）。
沿真实折线等弧长布点，去重；不写rim=12之类资产常数。

局限：近邻转折证据是空间局部近似，极薄／复杂局部可能混入另一侧的法线变化；
不把检测出的每个大平面边界都宣称为语义杯沿。误识别/漏识别仍需人工看features视图。

## 6. 统一半径政策及实际数量

```text
L = sqrt(hull.area / (2*pi))
r = min(0.10*L, 0.006 m + 0.02*L, 0.012 m)
```

同一资产所有球先使用同一半径。0.10L约束避免微小物体被固定6mm项支配；12mm绝对上限避免大物体巨球。
不使用spacing/2，不按类别，不随球数改变。
比较过 `min(0.06L,12mm)`：杯子仅5.16mm，比旧认可中位数8.56mm偏小；因此使用共享有界仿射公式。
不是全局最优半径标定，只是本轮三个资产量级对齐的简单系统政策。

| 资产 | 上限32实际数量 | 上限64实际数量 | 本轮半径 min/p25/median/p75/max（mm） | 旧v2 32球半径中位数（mm） |
| --- | ---: | ---: | --- | ---: |
| mug | 32 | 64 | 全部7.720 | 8.556 |
| bowl_001 | 32 | 64 | 全部9.859 | 9.521 |
| box_167 | 26 | 56 | 全部11.344 | 8.485 |

盒子的球比旧固定6mm外扩实验中位数大约34%，不是完全相同半径；这是统一公式结果。
新模式各消融之间，以及同资产32/64上限之间，半径完全一致。
历史参考使用旧实际半径，不冒充同半径消融；完整分位数见 `historical_reference.json`。

## 7. 消融与诊断数值

10种几何×2档上限×4种采样=80条结果。
几何包括cube、长方体、box_167、旋转box_167、浅托盘、圆角盒、bowl_001、
程序生成的更平滑碗、mug、同长方体细分三角网格。
更平滑碗是明确标注的合成几何，不是另一只下载的真实扫描资产。

每组B/C/D/E使用同一hull、同一上限、同一实际数量、同一半径、同一独立12000点评估集（seed20260919）。
box上限64的最终规则网格是56点，因此随机/FPS/feature对照也都使用56点。

| 资产 | B面积随机 p95 | C普通FPS p95 | D特征+FPS p95 | E自动最终 p95 |
| --- | ---: | ---: | ---: | ---: |
| box（56球） | 76.74 mm | 55.28 mm | 61.92 mm | 54.39 mm |
| bowl（64球） | 58.45 mm | 37.99 mm | 42.20 mm | 42.20 mm |
| mug（64球） | 25.47 mm | 17.26 mm | 18.18 mm | 18.18 mm |

这是独立 hull 表面点到最近采样球心的距离，不是碰撞误差保证。
杯/碗特征约束牺牲少量全表面均匀度，换来闭合轮廓稳定布点；不能宣称所有指标都优于普通FPS。

最终结果的距离分布（median/p95/max，mm）：

| 资产 | 上限32 | 上限64 |
| --- | --- | --- |
| box | 53.84 / 81.25 / 96.06 | 36.31 / 54.39 / 66.58 |
| bowl | 36.10 / 59.15 / 82.60 | 24.84 / 42.20 / 55.28 |
| mug | 15.79 / 26.14 / 32.37 | 11.05 / 18.18 / 23.48 |

### 低维特征单独评价

每条棱／环用513个等弧长点评价nearest-site距离，另对球与每段折线求精确交区间，计算未覆盖弧长。
区分“球心准确在环上”和“球覆盖到环”，不会把几毫米投影误差算成整个特征漏掉。

- bowl64：两个主要环的实际在环球心20/10个（包括落在环上的普通极值点），最大中心弧长间隔52.5/52.0mm；
  球并集仍留下约32.8/32.2mm最大未覆盖弧段。
- mug64：两个环10/16个在环球心，最大中心间隔19.2/21.0mm；最大未覆盖弧段约3.72/5.50mm。
- box64：规则网格8角来源、24棱来源、24面来源；扫描hull仅检测4个高置信度尖角，4个均被球覆盖，
  网格球心距这些尖角约0.68–3.20mm，不声称是4个精确锚点。
- box的扫描近平面分区边界有些在理想网格棱内侧；因此某些rim“精确在环球心”为0不等于完全无球。
  报告同时给出sphere support、nearest distance和真实未覆盖弧长，不能只读一个计数。

各主要patch的p95、每条edge的最大和p95 gap、整段无球支持标记均在 `result.json`，没有靠面积采样代替边沿评价。

### 外轮廓 support error

```text
h_H(n) = max_vertex(n·v)
h_S(n) = max_sphere(n·c+r)
error = h_S-h_H
```

方向包括均匀球面方向、主轴、box轴、patch/feature法线。
正值=提前碰撞；负值=可能晚碰撞。报告完整方向和每方向值、正负两侧max/p95。
上限64最终结果：box、bowl、mug在所测方向中的最大晚接触均为0；提前接触最大分别约11.34、9.86、7.72mm。
这是大平面接近的诊断；**即使所有测试方向都不晚接触，小障碍物仍可能漏检**。

### 独立有限障碍物接近

每资产18方向×2横向偏移×box/cylinder=72个有限障碍物测试，另有18个大平面接近测试。
hull接触位置用线性规划解凸多面体相交的最早平移位置，sphere-vs-box/cylinder为解析距离解；不启动Isaac。
圆柱的hull参考用外切64边形，记录径向近似上界；不是声称连续圆柱严格真值。
有限障碍物半宽/半径0.09L，半高0.04L；偏移0和0.25L，所有资产共用。

上限64最终结果：

| 资产 | hull碰到、球沿整条接近线完全没碰到 | 有碰到的测试中最大晚接触 |
| --- | ---: | ---: |
| box | 37 / 72 | 326.1 mm |
| bowl | 3 / 72 | 292.6 mm |
| mug | 0 / 72 | 121.6 mm |

很大的晚接触表示障碍物从前面的球间隙穿进去，直到后侧球才碰到。
所以mug“0个完全miss”绝不等于0漏检或适合所有局部避障任务。
**本轮达到了结构化布点和原生接线目标，没有达到完整凸包碰撞等价目标。**
不擅自增加半径或突破64掩盖这个问题；是否接受应按后续搬运任务的障碍物尺寸决定。

## 8. 原生验证及测试

真实1e G2／CUDA中分别验证64球和32球，后端均预留64槽：

- object→link→native FK最大中心误差：64球约0.393µm，32球约0.387µm。
- 实际world collision kernel的payload-only sentinel代价约0.00872044；机器人本体为0，detach后为0。
- 64→32诊断子集→64，以及32→16诊断子集→32：碰撞响应消失再恢复，剩余槽radius为负。
  这里的减半子集专用于sentinel测试，不冒充重新优化的32/16球模型。独立生成的32球也另跑完整backend。
- detach世界移除状态恢复，`mug/collision`重新成为世界障碍物；缓存reattach命中。
- 两环境使用不同关节状态、不同物体姿态，真实indexed native FK闭合误差最大约0.233µm；
  减少球及detach对两环境均清空尾槽。
- 预览最终球来自native实际槽位，而非仅显示generator副本。
- 本机缓存reattach约23–25ms；隔离FK+碰撞查询+回读约0.38–0.41ms。
  不是完整轨迹规划性能，也不是实时保证。

测试记录：独立包175通过；相关Mission附着/可视化/接线40通过、2个其他G2 dish/fork用例跳过；
Mission schema/manifest选测10通过、2项无关测试排除；cuRobo attachment manager既有19项通过。
两项被跳过的既有用例不代替本轮实际G2 mug CUDA验证，也不声称运行了全部Plugins测试。

隔离导出的1e配置已通过：

```bash
fastsim config validate \
  outputs/convex_v3/native_mug_64/offline_runtime/runs/run.convex_surface-786482ddede9a7b9.yaml \
  --project ../FastSim-Demo/.fastsim-demo/runtime/project.yaml --offline --json
```

原正式1e run未改写。通过schema仅证明配置可编译，不等于整任务成功。
复跑native：

```bash
python -m asset_collision_spheres.adapters.workspace.mug_auto \
  --method convex_surface --geometry-config configs/demo/convex_surface.json \
  --diagnostic-budget 64 --output outputs/convex_v3/native_mug_64 \
  --build-only --extended-native-checks --write-run-config
```

## 9. 产物、回归与未运行项

全部新结果在 `outputs/convex_v3/`，旧历史结果不覆盖。

- `frozen_manifest.json`：330项旧配置/JSON/USD/截图哈希，前后核验。
- `ablation_summary.json`：80条数值对照；各资产各变体目录下有result.json和preview.usda。
- `<asset>/historical_reference.json`：v1/v2旧球的真实半径和独立hull重评估，旧文件原字节不动。
- `<asset>/obstacle_approaches.json`：逐方向有限障碍物接触与平面测试。
- `radius_calibration.json`：共享半径公式对比；`rotation_parity.json`：真实盒旋转一致性。
- `failure_cases.json`：空点集、共面、近退化点集明确拒绝；没有返回假球。
- `native_mug_32/`、`native_mug_64/`：原生球、FK、sentinel、多环境、配置及截图。

复现实验：

```bash
python examples/freeze_convex_reference.py
python examples/run_convex_experiments.py
python -m pytest -q
```

已知限制／未运行：

1. generic目前用满有效上限，尚无稳定跨资产的提前停止阈值；没有CVT/Lloyd对照。
2. box局部proxy偏差超限会替换成FPS并review；圆角盒已覆盖此边界，不保证所有歪斜/圆角盒网格对称。
3. 近平面/loop阈值是一套共享经验值，不是任意扫描资产稳定识别证明。
4. 未对任意mesh做修复或高质量重网格；原始顶点噪声仍会影响凸包、分区和轮廓。
5. 用户减少球数不会增大半径，所以局部球间隙可能明显变大。
6. 小障碍物完全漏检/严重晚接触的反例已经保留；support良好并不能消除它们。
7. 未运行完整动态1e抓取/搬运/放下、自碰撞可行性验收或两环境Isaac物理任务。
8. 未改全局margin、ignore、物理collider、native CUDA kernel，也没有“任务成功”宣称。

下一步建议先由用户检查这三个资产的32/64预览，明确搬运任务可接受的最小障碍物尺寸与漏检范围，
再决定是否需要更密球预算或另一种能保守覆盖外形的碰撞表示；不把当前稀疏表面模型包装成完整凸包碰撞器。
