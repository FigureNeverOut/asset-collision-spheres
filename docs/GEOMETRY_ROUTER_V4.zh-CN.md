# 几何路由与原表面分球 v4：实现、验证、运行命令

本轮是增量实现，不替换已认可的凸包 v3。实现版本为
`geometry-policy-v4-experimental` 和 `original-surface-v4-experimental`。
自动路由判断的是“是否有足够证据允许整体凸包化作为外部避障代理”，不是物体语义分类。

## 1. 先看结果

当前真实 mug、bowl、box_167，以及合成浅盘自动选凸包；实心盒、球、直杆、弯杆、
U/L 形件、圆环、分叉件保留原表面。四个边界例也保守保留原表面。
16个几何 × 32/64上限，共32组最终实验，全部符合本次样例预期。
这不是未知资产准确率，也不是通用容器识别保证。

真实1e杯子的原表面32/64球、自动路由32球均经过native attachment验证。
没有运行完整物理抓取/搬运任务；静态机器人姿态不表示碰撞可行的运动轨迹。

旧配置与旧产物未改写。495个历史文件（配置、v2/v3输出及5个旧数值算法文件）
有冻结哈希，改造前后检查一致；原v3生成器、hull结构算法、hull诊断的源文件没有修改。
自动路由选凸包的8个实验，其球数组与直接调用旧v3逐元素完全相同。

## 2. 最少需要什么输入

算法需要物体三角网格，坐标为物体局部坐标、单位米。
外部文件还需要真实单位、缩放和源坐标变换；USD需要明确选定mesh prim。
这些是物体来源信息，不是杯沿/把手/角点标注。

新路由最简配置：

```json
{"mode":"auto_geometry","geometry_policy":"auto"}
```

改成最多32球：

```json
{"mode":"auto_geometry","geometry_policy":"auto","max_spheres":32}
```

| 字段 | 默认值/要求 | 含义 |
|---|---|---|
| geometry_policy | 在旧入口中必须显式提供才启用v4 | auto / convex_hull / original_surface |
| mode | 新入口默认auto_geometry | 新策略只允许auto_geometry，不混入旧convex_surface模式 |
| shape_hint | auto，可选 | auto / open_box / box / generic；只是结构提示，不控制路由 |
| max_spheres | 64，可选 | 整数1–64；32是上限，不承诺必须用满 |
| seed | 5，可选 | 确定性采样种子 |
| backend_sphere_capacity | backend内部提供 | 不应写到任务YAML；有效上限=min(max_spheres, native容量,64) |

不需要 sphere_budget、手工区域、每区配额、开口轴、杯沿/杯底高度、把手框、每类半径。
这些旧材料模式参数混进v4配置会报错，避免语义悄悄变更。
`shape_hint=box`在新层映射到旧结构分析的auto；`open_box`保留旧的宽松盒状候选阈值。
两者都不是强制路由，仍需对应目标表面的结构证据。

三种策略：

- convex_hull：明确强制调用已有convex_surface；不经过自动资格判断。
- original_surface：明确强制保留原三角形表面；不先把物体整体填实。
- auto：先构造分析用hull，只有组合证据充分才选凸包，否则保留原表面。

直接调用新的 `generate_geometry_spheres(mesh)` 默认auto。
旧的 `generate_auto_geometry_spheres` 只有看到geometry_policy字段才转到v4；
旧 `{"mode":"convex_surface"}` 和 material-v1/joint-v2 配置继续走原分支。

## 3. 真实调用链

```text
离线mesh/USD loader → 米制原网格
  → generate_geometry_spheres / 旧auto入口的显式v4分派
  → geometry router 或人工指定策略
    ├─ convex_hull → 既有convex_surface v3
    └─ original_surface → 原表面结构分析 → 盒网格 / surface graph FPS
  → 共享半径公式 → 物体局部米制 [N,4]

FastSim运行时：
attached_object_mesh_configs[object_id]
  → prepare_request（实际容量、规范化配置、版本/配置hash）
  → mesh缓存查找 → fit_mesh_attachment → 上述新入口
  → object-to-link变换 → AttachmentManager.update
  → native kinematics attached_object槽位 → CUDA FK/碰撞检查
```

没有用预计算球冒充native生成器运行。mug预览的 `*_native_input.json` 仅保存输入证据；
真正backend根据配置再次生成/命中自己的缓存。没有修改cuRobo kernel或AttachmentManager。

任务配置示意（资产/robot_config等现有字段仍保留）：

```yaml
attached_object_sphere_method: mesh
attached_object_sphere_budget: 64  # native预留容量，不是必须生成64个
attached_object_sphere_fallback: error
attached_object_mesh_configs:
  mug:
    mode: auto_geometry
    geometry_policy: auto
    max_spheres: 32
```

本轮只导出独立实验run.yaml，不覆盖正式1e任务，不改全局碰撞忽略或碰撞margin。
缓存包含geometry-policy-v4版本和规范化配置；不同策略/上限/seed/native容量不会混用。

## 4. 自动路由：证据、公式、阈值

源码：`algorithms/geometry_router.py`，阈值是统一的系统策略，不是单资产标注。

设原网格M、分析凸包H，长度尺度 `L=sqrt(area(H)/(2*pi))`。

1. 使用4096个确定性分层面积样本，计算H表面到M实际三角形的距离。
   距离超过 `0.025L` 视为原表面不支持的hull新增区域。
2. 使用旧hull相连平面patch分析。候选cap要求patch面积至少占H的4%，
   该patch至少55%的样本是不支持区域。最多精查新增面积最大的4个候选。
3. 填充率 `F=volume(M)/volume(H)` 仅在M为正体积、封闭、绕序一致时有效。
   无可靠材料体积时不自动批准凸包，保留原表面并标记review。
4. 候选cap的新增面积集中度 `C=该patch不支持样本数/全部不支持样本数`，要求C≥0.30。
5. 从候选cap的最多25个位置沿向内法线发射射线。至少65%的射线应在
   候选轴向总深度的12%–101%范围遇到材料；候选中心射线也应满足此深度条件。
   无底的圆环/弯杆间隙通常在此失败。101%只是数值/扫描容差。
6. 在中心可用深度的20%、40%、60%处计算真实mesh截面。
   每层检查24个周向射线：至少95%的方向需要有两次以上材料交点，
   并且至少95%的方向为偶数交点（中心位于自由空间）。三层必须全部通过。
   只有侧边几根实心臂、没有周向围壁的U/L形空间通常在此失败。
   这里采用24方向的截面奇偶一致性，而不是容易在薄扫描壁上抖动的单条3D contains射线。
7. 对面积采样点协方差，取最大两个特征值，细长度 `E=sqrt(lambda1/lambda2)`。
   E≥8不批准整体凸包。另输出 `clip((E-4)/4,0,1)` 反证强度。
   弯曲/分叉反证主要还来自多patch新增区域、围壁/底证据失败；没有实现完整骨架分支检测。
8. 汇总分数：

```text
score = 0.25*min(C/0.60,1)
      + 0.20*cap不支持比例
      + 0.25*截面通过比例
      + 0.25*底部射线通过比例
      + 0.05*clip(1-F,0,1)
```

只有全部硬条件通过且score≥0.78，才批准convex_hull。
score不是经过统计校准的概率，不能把0.84解释为84%的安全性。
体积填充率只有5%权重，绝不是 `if hollow then convex_hull`。

`geometry_routing`输出requested/selected、score、evidence、每个候选cap、
各层截面统计、失败gate、reason/fallback_reason、review_required及分析hull。
显式强制策略的score为null，表示没有声称做过自动判断。
无有效hull时尝试原表面；原始数据连可用三角形都没有则报错，不造假厚度或假球。

## 5. 原表面如何分球

只焊接坐标完全相同的导出接缝，丢弃数值退化三角形和孤立无引用顶点。
不把空间上很近但不同的两臂焊起来，不填洞，不做整体凸化。
USD新入口要求已经三角化；旧凸包显示用的多边形扇形剖分不能拿来当原表面材料。

### 实心盒/近盒：规则表面网格

独立对M分析平面、角棱和盒坐标系，复用旧网格节点分配。
不能只因存在三个正交法线方向就把U/L形当盒子：额外检查M接近其包络。
封闭材料的体积比需≥0.85；原表面到包络距离p95≤0.025L、最大≤0.08L。
没有可靠体积时要求面积比在0.85–1.10且p95≤0.015L，并保留review标记。

沿三轴自动选(nx,ny,nz)，共享角/棱/面节点，总数：
`N=nx*ny*nz-max(nx-2,0)*max(ny-2,0)*max(nz-2,0)`。
枚举预算内候选，最小化 `std(log(axis_spacing))+0.28*(cap-N)/cap`。
实际投影目标是M三角形，不是H；投影偏差大的点以原表面graph采样补位。
方块64上限→4×4×4边界→56球；不同长宽高会选不同节点数，并非强行凑满64。

### 通用非凸表面：近似测地线graph FPS

图节点包括原顶点、共享边中点、面中心、4096个面积样本及必要的补位点。
边只能连接同一真实三角形内的点，边权是该三角形内线段的米制长度。
面内多个面积样本再建立二维局部Delaunay连接，减少细长三角形上的绕路误差。
没有任何跨空气的空间kNN边。

使用Dijkstra最短路更新“到已选球心的最近图距离”，每次取最远节点。
先给连通分量种子（容量不足时按面积优先），再细化；未表示分量写进诊断，
adapter会拒绝把整个分量完全遗漏的球集直接挂载。
输出每个连通分量的面积、球数、参考图距离、不可达比例及面内剖分失败数。
连通的U形两臂必须沿底部绕行，不能通过空气互相“覆盖”。

这仍是近似图测地线，不是精确曲面测地线。不同剖分不承诺球位置完全相同。
测试了U形细分/旋转后的表面位置与距离质量；规则盒不同剖分/刚体变换则检查对应点集一致。

## 6. 半径、诊断与碰撞边界

共享原来的公式和12mm硬半径上限：
`r=min(0.10L, 0.006m+0.02L, 0.012m)`。
凸包路线L来自H面积；原表面路线L来自M面积。因此同物体两条路线半径可不同，
同路线32/64上限的半径相同，不会用巨大球弥补预算下降。
例如mug：凸包r≈7.72044mm，原表面r≈8.09719mm。

路由诊断与采样诊断分开：后者有actual_count、sampling_strategy、radius_m、selection、
surface_uniformity、feature_diagnostics、connectivity_fairness、review_required。
12000个独立种子的目标表面样本报告最近球心距离及球体并集缺口。
support_error只度量外方向支持范围，不能检测U内部凹陷，不能替代原表面距离与图公平性。

native既有attestation字段是固定标量；新原表面路径明确写入metric_mapping：
protrusion字段是保守上界，半径是距离到原表面的上界，不冒充材料穿出精确测量。
体积比是球体积之和/材料体积，不是并集覆盖；无有效体积时标量0并另标记不可用。

球心落在原表面不等于球体不伸进凹槽。所有稀疏表面球都可能存在局部漏检、
球半径造成的槽口收缩和过度保守；本轮没有证明零漏检、连续碰撞或轨迹安全。

## 7. 实际路由实验

最终结果：`outputs/geometry_v4/regression_surface_refined/summary.json`。
每个`<case>/32`与`<case>/64`都有result.json和preview.usda。
前面的regression_01、regression_final保留为过程记录，最终以surface_refined为准。

| 样例 | 来源 | 自动路径 | score约值 | 备注 |
|---|---|---|---|---|
| mug | 原1e真实资产 | convex_hull | .837 | 32/64球；与旧v3相同 |
| bowl | 原bowl_001真实资产 | convex_hull | .946 | 32/64球；与旧v3相同 |
| box | 原box_167真实开口盒 | convex_hull | .971 | 26/56球；与旧v3相同 |
| tray | 合成浅盘 | convex_hull | .958 | 有开口、围壁与底 |
| solid_box | 合成长方体 | original_surface | 0 | 原表面规则网格 |
| sphere | 合成球面三角网格 | original_surface | 0 | 原表面graph FPS |
| straight_rod | 合成实心圆柱杆 | original_surface | 0 | 没有容器cap，且细长 |
| curved_rod | 合成弯实心杆 | original_surface | .422 | 有hull新增面但缺容器证据 |
| u_solid / l_solid | 合成实心U/L形件 | original_surface | .440 / .506 | 围壁/集中度等gate失败 |
| ring_solid | 合成实心圆环 | original_surface | .425 | 孔贯通，缺底 |
| branched_solid | 合成实心分叉件 | original_surface | .436 | 非容器围壁 |
| hollow_bent_tube | 合成空心弯管 | original_surface | .429 | 保形回退，需review |
| narrow_neck | 合成细颈容器 | original_surface | 0 | 小开口不满足cap条件，允许漏识别，需review |
| irregular_scan | 合成缺一个三角形的盒面 | original_surface | 0 | 无可靠材料体积，需review；不代表任意扫描噪声测试 |
| rounded_box | 合成圆角超椭球盒 | original_surface | 0 | 本例用generic，未强行套网格 |

U/L/弯杆的review表示检测到候选但拒绝凸包，需要人确认回退，不表示它们不该保留原形。
没有候选cap的正常实心体可以直接保留原表面；不确定且新增面较多则标记review。

## 8. 实际测试与native验收

包原基线175项通过；新增路由/强制策略/图距离/盒网格/旋转细分/配置与USD验证测试。
最终212 passed（新增37项），记录在 `outputs/geometry_v4/package_tests_complete.log`。
Mission附件/可视化/接线选择：43 passed，2 skipped；两项是既有的G2 dish/fork条件测试。
schema/manifest选择：13 passed，2 deselected。
cuRobo既有AttachmentManager：19 passed；未改cuRobo源文件。

三份最终native报告：

- `native_auto_mug_32/auto_geometry_native.json`：自动选凸包，32球/容量64，FK最大误差3.8735e-7m。
- `native_original_mug_32_final/auto_geometry_native.json`：强制原表面32球/容量64，误差3.9075e-7m。
- `native_original_mug_64_final/auto_geometry_native.json`：强制原表面64球/容量64，误差3.9631e-7m。

每份均真实执行：object-to-link、native槽位、CUDA FK、payload-only sentinel、
detach后槽位负半径/碰撞消失、reattach cache hit、full→减少一半→full的尾槽清理，
以及两个不同姿态环境的indexed CUDA FK与清理。哨兵不碰机械臂；原表面payload代价约
0.00909719，arm和detach后为0。双环境是native G2验证，不是两个Isaac物理场景。

原表面32/64与auto32的隔离run已通过FastSim config validate；这只证明配置/编译成功。
截图已实际用Isaac渲染并检查：U形32球、bowl路由截面证据、mug自动32球。
没有把仅生成USD当作已经看过所有资产的交互预览。

## 9. 直接运行的命令

先进入正确项目并激活环境：

```bash
cd asset-collision-spheres  # 从本地工作区根目录进入
# 先在终端初始化本机conda
conda activate fastsim_vnext
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

看U形原表面32球（换目录32为64即可看64上限）：

```bash
python -m asset_collision_spheres.preview.inspect \
  --usd outputs/geometry_v4/regression_surface_refined/u_solid/32/preview.usda \
  --display geometry_spheres --view top
```

看碗的自动32球：

```bash
python -m asset_collision_spheres.preview.inspect \
  --usd outputs/geometry_v4/regression_surface_refined/bowl/32/preview.usda \
  --display geometry_spheres --view close
```

把 `--display geometry_spheres` 改成 `--display geometry_analysis` 看路由证据。
正常图中青色线是选中的凸包，紫色线是原表面；分析图中橙色是分析hull/截面，
绿色轴是通过的cap、红色轴是未通过的cap。分析hull默认隐藏，不冒充选中目标。
其他资产只替换上述case名，如curved_rod、l_solid、solid_box。

看1e机器人杯子自动32球（已native验证）：

```bash
python -m asset_collision_spheres.adapters.workspace.mug_auto \
  --method auto_geometry --geometry-config configs/demo/geometry_auto_32.json \
  --output outputs/geometry_v4/native_auto_mug_32 \
  --open-existing --view close --hide-visual
```

强制原表面32球对照：

```bash
python -m asset_collision_spheres.adapters.workspace.mug_auto \
  --method auto_geometry --geometry-config configs/demo/geometry_original_32.json \
  --output outputs/geometry_v4/native_original_mug_32_final \
  --open-existing --view close --hide-visual
```

紫色杯子仍是原始位置的参照mesh，按之前要求保留；不是当前球发生了位置偏移。
删掉 `--hide-visual` 可显示抓起后的实体杯子，但一些球会被杯壁遮挡。

新资产只看自动分球，无需造场景或让机器人抓取：

```bash
python -m asset_collision_spheres.preview.geometry \
  --preset configs/local/bowl/bowl_001_auto_geometry.json \
  --geometry-policy auto --max-spheres 32 \
  --output outputs/geometry_v4/my_bowl_test
python -m asset_collision_spheres.preview.inspect \
  --usd outputs/geometry_v4/my_bowl_test/preview.usda --display geometry_spheres
```

preset只读取source，不使用其历史generation配置。普通OBJ/PLY/STL则使用：
`--mesh /绝对路径/object.obj --unit-scale-m 1`，替换--preset；必须提供正确单位。
新生成器拒绝覆盖既有输出目录；修改32为64时另选目录，例如my_bowl_test_64。

重新跑native原表面32验证（另选新目录，不覆盖已验收证据）：

```bash
python -m asset_collision_spheres.adapters.workspace.mug_auto \
  --method auto_geometry --geometry-config configs/demo/geometry_original_32.json \
  --diagnostic-budget 64 --output outputs/geometry_v4/my_native_original_32 \
  --build-only --extended-native-checks --write-run-config
```

完整离线样例回归：

```bash
python examples/run_geometry_experiments.py --output outputs/geometry_v4/my_regression
python examples/freeze_geometry_reference.py
python -m pytest -q
```

在本机无图形会话终端启动Isaac时，可在命令前加
`DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority`；普通桌面终端通常不用额外设置。
所有预览无需按Play，不执行物理。

## 10. 阅读和修改范围

实际检查：根AGENTS与仓库状态、项目快速上手/仓库位置/vNext迁移说明、迁移文档，
当前凸包现状文档与v3/v2说明；api、auto_geometry、convex_surface、hull_structure、
hull_diagnostics、loaders、FastSim adapter、CLI、mug/native验证、preview链及相关测试；
Mission配置/manifest/index、backend与attestation类型。

新增（相对asset-collision-spheres）：

- `src/asset_collision_spheres/algorithms/{geometry_policy,geometry_router,original_surface,surface_graph}.py`
- `src/asset_collision_spheres/preview/geometry.py`
- `examples/{geometry_cases,run_geometry_experiments,freeze_geometry_reference}.py`
- `configs/demo/{geometry_auto,geometry_auto_32,geometry_original,geometry_original_32}.json`
- `tests/test_geometry_policy.py`，本说明文档。

修改既有接线：

- `src/asset_collision_spheres/{api,__init__,cli}.py`、`algorithms/auto_geometry.py`
- `adapters/fastsim.py`、`adapters/workspace/{mug_auto,mug}.py`
- `loaders/usd.py`、`preview/{inspect,isaac}.py`、README。
- FastSim-Plugins Mission中：curobo `backend.py/config.py/alignment.py`，
  catalog `mission.yaml/index.yaml`，`test_sphere_package_integration.py/test_artifacts.py`。
  更新本地manifest对应SHA，未改归档版本；未提交/推送。

不改旧数值算法、正式1e资产/任务、物理参数、全局碰撞忽略和cuRobo内核。
工作树已有大量迁移/其他未提交改动，原样保留，没有reset或清理。

## 11. 尚未运行、已知风险与下一步

- 未运行完整动态抓取/搬运、所有Plugins测试、两环境Isaac物理、未知资产大批评测。
- 未在U/L/弯杆资产上建立真实机械臂抓取场景；它们验证的是离线几何，
  新原表面native路径用真实mug验证，不混淆这两种证据。
- 4个正例与合成反例远不足以估计泛化准确率；阈值仍需用未参与开发的资产验证。
- 多开口、多腔、偏心/不规则容器、开口在小颈处、带复杂突出部可能回退或误判。
- 局部精查最多4个cap，24条截面方向与有限采样不能证明真实自由空间拓扑。
- 原表面graph距离仍受剖分和采样影响；细长/各向异性三角形可导致分布不够均匀。
  面内补连接后，直杆64球的表面最近球心p95仅从约39.59mm变为39.11mm，改善有限；
  不把这次补连接宣传为解决了所有网格质量问题。
- 默认64只是数量上限，没有自动判断“多少球已经足够安全”的停止准则。
  通用表面一般用满有效容量，盒网格可少于上限。
- 首选下一步：用户检查U/L、圆环与新资产；建立保留集统计“错误凸包化”，优先压低它；
  再改网格重采样/长边细分与图距离近似，最后讨论带明确误差目标的自动球数选择。
