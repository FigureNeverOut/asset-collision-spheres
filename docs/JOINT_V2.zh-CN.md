# v2 增量实验：方盒结构与球大小／数量联合选择

2026-09-07。本轮实现位于迁移后的 `asset-collision-spheres`，不再往 task2sim 的旧转发文件里加算法。

这是可运行的实验分支，不是任意物体通用识别器，也没有零漏检保证。旧配置仍走 `auto-geometry-material-v1-experimental`；出现新策略字段才进入 `auto-geometry-joint-v2-experimental`。已认可的 mug／bowl／box 配置和历史 JSON、USD、截图保留，61 项 SHA256 核对通过。

## 先看效果

以下命令在本项目使用 `fastsim_vnext`，无需重新建立场景、重新抓取或点击 Play。

```bash
# 先在终端初始化本机conda
conda activate fastsim_vnext
cd asset-collision-spheres  # 从本地工作区根目录进入
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

### 方盒：同样 32 球、同样最多外扩 6 毫米

先关掉一个 Isaac 窗口，再开另一个即可。

```bash
# 上轮 auto geometry；不是 cuRobo 自带的原始分球算法
python -m asset_collision_spheres.preview.inspect \
  --usd outputs/structure_v2/ablations/box/A_legacy_32/preview.usda --display spheres

# 本轮：自动识别结构＋多半径，仍为 32 球／6 mm
python -m asset_collision_spheres.preview.inspect \
  --usd outputs/structure_v2/ablations/box/G_auto_32/preview.usda --display spheres

# 单独看识别结果：彩色=平面片，灰色=未归类材料，青色=棱，红色=角邻域
python -m asset_collision_spheres.preview.inspect \
  --usd outputs/structure_v2/features/box/features.usda --display features
```

将路径中的 `_32` 改成 `_64` 可看同条件 64 球。橙色是球，紫色是同位置的原材料 mesh 线框；按住鼠标调整视角也不会改变球数据。

下面这一组**改变了外扩上限**，不能与 6 mm 结果直接归因为“算法更好”：

```bash
# 自动尺度上限，64 球硬上限；本盒算出的外扩搜索上限为 20 mm
python -m asset_collision_spheres.preview.inspect \
  --usd outputs/structure_v2/ablations/box/J_scale_aware_64/preview.usda --display spheres

# 不指定用户数量上限，受系统 256 球上限约束；本次自动停在 133 球
python -m asset_collision_spheres.preview.inspect \
  --usd outputs/structure_v2/ablations/box/L_scale_system_64/preview.usda --display spheres
```

最后路径的 `_64` 是实验编号中保留的旧预算字段；在 adaptive 模式下它不是实际球数。实际数量必须看 `result.json`，本次为 133。

### 1e 杯子：显示真正写入 cuRobo 并经过 FK 的球

```bash
python -m asset_collision_spheres.adapters.workspace.mug_auto \
  --method auto_geometry --geometry-config configs/demo/joint_v2_fixed_32.json \
  --diagnostic-budget 64 --output outputs/structure_v2/native_joint_32 \
  --open-existing --hide-visual --view close
```

这里的 `64` 是后端预留槽位，实际只有 **32 球**。去掉 `--hide-visual` 可同时显示抓取后的杯子实体；紫色原位置 mesh 仍保留。颜色代表几何 patch，不是人工标出的杯沿、杯壁或把手。

```bash
# 少参数自动配置；本次 128 个后端槽位中实际使用 90 球
python -m asset_collision_spheres.adapters.workspace.mug_auto \
  --method auto_geometry --geometry-config configs/demo/joint_v2_adaptive.json \
  --diagnostic-budget 128 --output outputs/structure_v2/native_joint_adaptive \
  --open-existing --hide-visual --view close
```

### 碗：同条件对照

```bash
python -m asset_collision_spheres.preview.inspect \
  --usd outputs/structure_v2/ablations/bowl/A_legacy_32/preview.usda --display spheres
python -m asset_collision_spheres.preview.inspect \
  --usd outputs/structure_v2/ablations/bowl/G_auto_32/preview.usda --display spheres
```

当前 32 球／6 mm 约束下，碗的新旧结果相同，这是有效的对照结果，不是没有运行新代码。

## 配置怎么写

固定数量例子见 [joint_v2_fixed_32.json](../configs/demo/joint_v2_fixed_32.json)。为控制变量，它显式保留了旧采样数和探针尺寸。

日常少参数入口见 [joint_v2_adaptive.json](../configs/demo/joint_v2_adaptive.json)：

```json
{
  "mode": "auto_geometry",
  "shape_hint": "auto",
  "sphere_count_mode": "adaptive",
  "max_spheres": null,
  "outward_offset_mode": "scale_aware",
  "max_outward_offset_m": null,
  "extra_face_spheres": 0,
  "seed": 5
}
```

| 字段 | 含义 |
|---|---|
| `shape_hint: auto` | 尝试可信 box-like 几何增强；失败时继续 generic，不强套方盒 |
| `open_box` | 降低尝试结构解释的置信度门槛；不能捏造 AABB 角、放宽材料约束 |
| `generic` | 关闭 box 特征；在 v2 中仍使用通用多半径／数量策略 |
| `sphere_count_mode: fixed` | 使用 `sphere_budget`；不超过预算，正常候选充足时达到指定数量 |
| `adaptive` | 初始目标、实际数量、用户上限、后端容量分别记录；不追求 100% 探针命中 |
| `max_spheres` | 用户硬上限；省略／null 仍受系统上限和真实后端容量约束 |
| `outward_offset_mode: fixed` | 必须明确给出 `max_outward_offset_m` |
| `scale_aware` | 根据几何计算搜索上限；如果同时给了明确上限，取更严格的那个 |
| `extra_face_spheres` | 在同一候选池上追加补面球，不重新随机化原球；容量不足明确报告未补足 |

`extra_face_spheres` 不能突破 fixed 的 `sphere_budget`。因此已用满 32 球的 fixed 结果请求补 8 球，仍是 32 球并报告 8 球未补足；希望追加时使用有剩余容量的 adaptive 模式。[补面配置](../configs/demo/joint_v2_adaptive_faces.json) 与普通自动配置只差这一项。

`--budget` 只适用于 fixed；adaptive 要改 `max_spheres`。普通物体不需要类别名称、杯口方向、把手框、面 AABB 或每个区域的球数。

输入仍需要正确的真实材料网格、单位和坐标系。USD 资产选择／缩放属于 loader 的 `source`，不是分球参数；不要把 `configs/box` 等资产预设文件整体作为生成配置传入。

### 从现有盒子资产重新生成

```bash
python -m asset_collision_spheres.preview.mesh \
  --preset configs/local/box/box_167_auto_geometry.json \
  --config configs/demo/joint_v2_fixed_32.json --budget 64 \
  --output outputs/structure_v2/my_box_fixed64 --build-only

python -m asset_collision_spheres.preview.mesh \
  --preset configs/local/box/box_167_auto_geometry.json \
  --config configs/demo/joint_v2_fixed_32.json --budget 64 \
  --output outputs/structure_v2/my_box_fixed64 --open-existing --display spheres
```

`preview.mesh --open-existing` 检查配置、源 mesh 和 USD 摘要；`preview.inspect --usd` 是显式打开诊断文件的快捷查看器，不做这些来源核验。两者都把拟合和 Isaac 启动隔离，避免提前加载独立 USD／数值库导致窗口卡死。

## 实际调用链和代码位置

```text
真实材料 mesh（物体局部坐标，米）
  → api.generate_auto_geometry_spheres
  → algorithms/auto_geometry.py（旧字段走 v1；新字段转 joint_selection）
  → joint_selection.py：复用旧候选／patch／诊断＋可选 box_features
  → [N,4]：cx, cy, cz, radius

Mission YAML schema
  → CuroboBackend._fit_full_mesh_attachment
  → 旧 mesh_attachment 转发模块 → adapters/fastsim.py
  → prepare_request 注入真实容量／缓存摘要 → fit_mesh_attachment
  → object-to-link 变换 → native AttachmentManager.update
  → kinematics 实际槽位和 FK → 原生碰撞查询
```

主要新增（算法／预览／适配器目录均在 `src/asset_collision_spheres/` 下）：

- `algorithms/box_features.py`：材料支持的平面、短桥接邻接、棱和角邻域。
- `algorithms/joint_selection.py`：新策略解析、多半径、加球／放大／有限修补／补面、诊断。
- `preview/features.py`、`preview/inspect.py`：结构分区和已有 USD 查看。
- `adapters/workspace/native_validation.py`：真实槽位减少、模型切换 sentinel、双环境 FK。
- `examples/run_joint_experiments.py`、`tests/test_joint_selection.py`、三个 `configs/demo/joint_v2_*.json`。

已有文件的增量接线：`algorithms/auto_geometry.py`、`adapters/fastsim.py`、`preview/mesh.py`、`adapters/workspace/mug.py` 和 `mug_auto.py`、`cli.py`。旧模块身份和 task2sim／Mission 转发关系不变。

项目外层只有两处必要接线修正：

1. `FastSim-Plugins/.../catalog/mission.yaml` 增加 v2 参数声明，更新 `index.yaml` 摘要并补 schema 测试。没有改 Mission 物理／规划策略。
2. `curobo_v2/curobo/_src/collision/attachment_manager.py` 取多环境多工具 link pose 时使用已有的 `make_contiguous=True`，补真实 CUDA 回归测试。没有改 CUDA 碰撞内核、球的几何或碰撞余量。

本轮未修改用户已有的 `curobo_v2/.../sphere_fit/fit_spheres.py` 调试改动，也不调用其中的 containment 补丁。

## 算法实际做了什么

### 复用的 v1

原实现确实按面积采样、沿内法线寻找材料球心、近似测地 patch 平衡、基于接触收益贪心，并进行有限同预算修补。原候选每个球心只有一个半径，近似为 `depth + cap - margin`：上限同时几乎直接决定半径。自由空间的超允许带惩罚通常为零，不能代替允许带内的外扩代价。

训练默认 6000 个表面样本；独立诊断默认 12000 个样本、seed=20260908，未拿独立诊断去选球。v2 继续复用这些入口。

### box 特征不等于语义分类

在三角形实际邻接图中聚合近共面区域，按面积拟合平面，保留残差、面积、法线、支持面和邻接。
检查至少三个近正交方向，再沿短材料表面桥寻找平面片接口；接口必须有实际共享边支持，不是无限平面求交。
短桥允许扫描噪声、窄倒角和口沿，但不连接隔着空气的独立面。内外壁不能仅因平行就合并。

角来自至少两个不平行的可信棱在局部相遇，最终投影回材料。圆角盒可以只留下可信棱段、没有锐角 anchor。它不证明“开口盒”语义，也没有完整孔洞拓扑分析。

角／棱附近增加不同方向、不同深度的少量球心，全部重新查询原材料；包括靠近特征的浅球心，不只保留最深点。再与普通候选去重，每个中心最多四个半径版本，最终只能选其中一个。

角要求表示一个小邻域：anchor 加两个材料方向的见证点，跨度为尺度长度的 1.5%；不让一个几乎看不见的小球仅碰到单点就算“保护好了”。主要棱分三段，对中段给予独立的递减收益。保护只使用有限资源，余下继续表示普通面／patch；adaptive 不允许仅选完保护球就因为收益阈值停掉，把面预算全部吃掉。

修补时可替换原保护球，但不能破坏已经建立的角或棱中段要求。独立特征评估可以由多个球联合支持角邻域；当前选择器的保护候选仍偏保守，这是大尺度／紧外扩约束下的限制之一。

### 通用大小—数量策略

合法中心深度为 `d`，外扩上限为 `cap`：

```text
e ∈ {0, 0.35, 0.65, 1.0} × cap
r = d + e - numerical_margin
```

同中心版本互斥。选择器比较新增接触／patch／特征收益，扣除重复覆盖、外扩、球数和超允许带自由空间代价。可以新增，也可以放大已选球；最多有限轮一换一／adaptive 一换二尝试。

默认软外扩代价为 `0.0015 × (e/cap)^2`，不是精确外凸体积。球数代价也是 0.0015；在固定预算下加球成本为常数，不会突破预算。大球确实有显著收益时允许用满 cap：不是为了显得“自适应”就强行缩小每个球。

自动模式不设用户上限时，离线最多 256；挂载时再取真实后端容量的最小值。停止原因包括收益太小、容量、迭代、时间、候选不足。时间上限在迭代边界检查，不是可强制中断所有几何查询的实时期限。返回不满预算／重要结构缺失的结果会报告需检查；没有合法材料球心则明确报错。

`fit_mesh_attachment` 在整个材料组件、已检测的重要角／主棱／大平面完全缺失时拒绝直接 native 挂载；普通稀疏空隙和不可信检测后的 generic 回退不触发这个门槛。小扫描平面片遗漏仍保守报告 review，但不冒充“整面没有球”。

### 尺度上限

采用 `L_ref = sqrt(material_surface_area / (2π))`，包含内外材料表面。

```text
derived_cap = min(0.06 × L_ref, 0.020 m)
effective_cap = min(derived_cap, 用户显式 cap)  # 用户未给时用 derived_cap
```

选择面积等效长度是因为：bbox 对细长突出物敏感且轴对齐 bbox 随旋转改变；robust extent 需要额外采样／分位数和可能忽略真实突出部；面积长度与刚体变换、平面细分不变，定义简单。不过它也会受重复表面、扫描毛刺面积影响，不是离群点免疫估计。达到 20 mm 绝对上限后，不再保证按比例扩大 cap。

显式 `fixed + 6 mm` 始终是 6 mm；不会自动变成上面的 20 mm。

## 系统默认值

以下是一套共享策略，不按 mug／bowl／box 改值。普通用户不用逐资产填写。

| 项目 | 默认 |
|---|---|
| 面积尺度比例／绝对外扩上限 | 0.06／20 mm |
| 系统球数上限／初始目标 | 256／8 |
| 外扩权重／指数 | 0.0015／2 |
| 球数成本／最小边际收益 | 0.0015／0.00015 |
| 特征收益权重／保护资源比例 | 0.06／0.35 |
| 最大选择迭代／时间检查阈值 | 512／30 s |
| 同中心外扩等级 | 0、0.35、0.65、1 |
| 特征候选最大附加规模／内移比例 | 2400／0.001、0.003、0.007、0.015 × L_ref |
| 平面区域最小面积／种子法线夹角 | 总面积 0.8%／16° |
| 生长平面距离／拟合 RMS 残差上限 | 0.007／0.005 × L_ref |
| 平面可信方向／每方向面积 | 13°／总面积至少 1.5% |
| auto／open_box 结构分数门槛 | 0.58／0.48 |
| 材料桥距离上限 | 0.045 × L_ref；三角最小高度加权的图近似 |
| 合并片法线点积／平面距离 | ≥0.99／<0.009 × L_ref，并要求材料桥邻接 |
| 棱接口法线点积 | 近垂直绝对值≤0.30，或相反≤−0.94 |
| 棱最小长度／线拟合 90% 残差 | 0.09／0.025 × L_ref |
| 角端点聚合／非平行判据 | 0.05 × L_ref／方向绝对点积≤0.8 |
| 角邻域跨度／命中数值容差 | 0.015／0.001 × L_ref |
| 棱训练／复检样本；表示邻域 | 每段 13／41 点；0.006 × L_ref |
| 大平面诊断 | 面积≥总面积 3%；独立 2048 样本，seed=20260909 |
| 主棱诊断 | 长度≥最长可信棱的一半，单独报告完全没有表示的主棱 |

基础候选、patch、探针等默认值继续来自 v1：6000／64／seed=5，balance=1.5、free-space=0.2、redundancy=0.02；默认 repair=0，控制变量实验明确设为 1。未显式给探针时，半径取旧 characteristic length 的 3%，穿入深度取探针半径的 25%。因此少参数 native 演示与固定 6 mm 探针的消融表不是完全同一配置。

参数依据不是普适最优证明：48 组跨三资产的共享代价比较选择了温和的平方惩罚。0.0005 往往全部选最大；0.006 使盒子明显过小；0.0015 在保留现有结果和允许不同半径之间较温和。具体完整记录在 `outputs/structure_v2/cost_calibration.json`。检测阈值经过理想盒、扫描盒、浅托盘、旋转／细分、圆角和曲面负例检查，仍是实验默认值。

## 实测结果与解释

完整 70 组消融记录在 `outputs/structure_v2/ablation_summary.json`，各目录另有完整 `result.json`、`generation.json` 和 USD。所有半径分位数、外扩分布、候选来源、修补、耗时和自由空间诊断都在 JSON 中。

### 同 mesh、seed=5、外扩 6 mm、同预算

| 资产／方法 | 球数 | 半径 min／P25／中位／P75／max（mm） | 独立接触命中 | 角邻域支持 |
|---|---:|---|---:|---:|
| 盒 v1 | 32 | 7.25／8.31／8.50／8.56／10.99 | 4.79% | 1/8 |
| 盒：仅完整特征增强，单半径 | 32 | 7.02／8.10／8.50／9.62／10.99 | 5.03% | 8/8 |
| 盒：auto＋多半径 | 32 | 7.02／7.96／8.49／9.13／10.99 | 4.97% | 8/8 |
| 盒 v1 | 64 | 7.25／8.15／8.50／8.50／10.99 | 9.71% | 2/8 |
| 盒：auto＋多半径 | 64 | 5.54／7.68／8.45／8.50／10.99 | 8.82% | 8/8 |
| 杯 v1 → v2 | 32 | 相同：7.55／7.92／8.56／9.13／11.02 | 49.44% → 49.44% | 不套 box |
| 杯 v1 → v2 | 64 | v2：5.75／7.57／8.08／9.09／11.02 | 83.42% → 83.64% | 不套 box |
| 碗 v1 → v2 | 32 | 相同：8.16／9.50／9.52／10.29／13.04 | 13.20% → 13.20% | 不套 box |
| 碗 v1 → v2 | 64 | 相同：7.76／9.28／9.52／9.52／13.04 | 25.28% → 25.28% | 不套 box |

盒子 32 球的棱分布分数由 0.148 提升到 0.420，64 球由 0.272 到 0.568；但 32 球的最大未表示棱间隙仍约 26.5 cm。**角邻域被照顾到了，不代表棱已经形成连续碰撞链。** 普通面的覆盖也仍非常稀疏，需要你看图确认。

候选-only 与 selection-only 分开跑过：当前数据中评分选择的影响明显大于单纯增加候选；加候选仍有助于稳定角附近的合法位置。另有 `A0_single_control` 检查新选择器自身与旧版的差异；新修补方法与旧修补不是完全同一个实现，不能将所有差异都归给特征。

### 自动尺度＋系统上限，不强行固定球数

下表控制实验仍用显式 6 mm 接触探针；实际 native 少参数杯子演示使用继承的默认探针，数量为 90，不是表中的 81。

| 资产 | 自动 cap | 实际球数 | 半径中位数 | 加 8 个补面球后 | 停止原因 |
|---|---:|---:|---:|---:|---|
| 杯 | 6.29 mm | 81 | 7.86 mm | 89 | 边际收益不足 |
| 碗 | 13.05 mm | 116 | 16.28 mm | 124 | 边际收益不足 |
| 盒 | 20 mm | 133 | 22.50 mm | 141 | 边际收益不足 |

三者都使用同一组系统参数。补面结果保留之前球数组作为前缀；在 `max_spheres=64` 且已满的另一组试验中，8 个补面球全部明确报告为未补足。cap=20 mm 不是半径=20 mm：半径还包括球心到材料边界的深度。

### 测试和真正挂载

- 独立包完整回归：138 项；原 113 项保持通过，另有 25 项新增策略／硬限制测试。
- 原生 cuRobo attachment 测试文件：19 项通过；新增的多工具双环境用例修复前可复现 contiguous 报错，修复后通过。
- Mission 参数 schema／转发／catalog 摘要的定向测试：9 项通过。
- 10 个特征预览案例、28 组旋转／细分／种子／缩放质量诊断、48 组共享外扩代价比较、70 组消融已真实运行。
- 先验证冻结 64 球杯子，再验证新 32 球／64 槽位和新 adaptive 90 球／128 槽位。
- 新 32 球 object→link→native FK 闭合误差约 0.38 微米。模型切换 32→16→32 后 sentinel 代价为 0.01013→0→0.01013；未使用槽位 radius 均为负值。
- attach 时 mug 不再作为 world obstacle，detach 后真实 mesh 重回 materialized world；另测了 detach、缓存 reattach。
- 两环境使用不同关节／物体位姿；native G2 indexed FK 对独立矩阵参考的误差约 0.12／0.23 微米，减少球和 detach 后两环境尾槽均为负。
- 原生槽位回读被写入最终杯子 USD，不是只显示生成器副本。
- 生成一般约 1–3 s（本机、单线程 BLAS）；32 球缓存 reattach 约 24 ms，隔离的 FK＋碰撞查询＋回读约 0.42 ms。不是完整轨迹规划耗时，也不是严格实时保证。

测试仍暴露问题：2 倍盒子＋固定物理 6 mm 约束的 adaptive 结果为 71 球、角邻域只支持 4/8，因此**不通过重要结构验收**，不能直接 native attach。几何及误差同比缩放的另一类测试保持了数量／半径比例；不要混淆这两类缩放实验。

圆角／扫描小平面片仍可能不完整。普通遗漏允许存在，但这不是任意大尺度物体自动可靠的证明。

## 接入真实任务与复现

独立实验任务配置已生成：

```text
outputs/structure_v2/native_joint_adaptive/run.auto_geometry.yaml
outputs/structure_v2/native_joint_adaptive/offline_runtime/runs/run.auto_geometry-ad9157165f74ce1c.yaml
```

只改变碰撞球策略／预留容量，不改原 1e `run.yaml`、物理碰撞体、全局 margin 或 ignore。FastSim 的 schema 子集不支持 nullable 数值，因此导出 YAML 时自动省略 JSON 中的 null；省略与生成器中的 null 语义相同。手写 Mission YAML 也请省略这两个 null 字段，而不是写 `max_spheres: null`。

```yaml
attached_object_sphere_method: mesh
attached_object_sphere_fallback: error
attached_object_sphere_budget: 128   # 后端容量，不是强制用满
attached_object_mesh_configs:
  mug:
    mode: auto_geometry
    shape_hint: auto
    sphere_count_mode: adaptive
    outward_offset_mode: scale_aware
    extra_face_spheres: 0
    seed: 5
```

配置已用已有本地规范化资产的 runtime project 离线校验通过：

```bash
fastsim config validate \
  outputs/structure_v2/native_joint_adaptive/offline_runtime/runs/run.auto_geometry-ad9157165f74ce1c.yaml \
  --project ../FastSim-Demo/.fastsim-demo/runtime/project.yaml --offline --json
```

不能换用未物化 SVN 资源的 `.fastsim/project.yaml` 冒充同一配置校验；没有重新下载或重做物理资产。

重跑本轮诊断／真实挂载：

```bash
python examples/run_joint_experiments.py --stage all
python examples/run_joint_experiments.py --stage cost-calibration --cases mug bowl box
python -m pytest -q

python -m asset_collision_spheres.adapters.workspace.mug_auto \
  --method auto_geometry --geometry-config configs/demo/joint_v2_fixed_32.json \
  --diagnostic-budget 64 --output outputs/structure_v2/native_joint_32 \
  --build-only --extended-native-checks
```

本轮未运行完整 1e 动态抓取／倒水任务，也未测完整轨迹规划耗时或双环境 Isaac 物理任务。通过的是配置、真实挂载数据链、原生碰撞核和静态显示，不能写成“整个任务已成功”。

## 仍需人工确认的边界

1. 杯／碗只是本次负例，不代表已经实现所有空腔的语义识别。
2. 真实材料网格必须闭合、朝向一致且有正体积；没有自动修复、自交穷尽检查或凸包兜底。
3. 原始三角剖分会影响近似图距离、扫描平面片边界和 tie-break；面积加权及同比缩放检查降低了敏感性，没有证明完全不变。
4. 超允许带自由空间探针均未发现侵入，但允许带本身可能占用空腔边缘；小孔、窄通路仍可能受影响，没有完整自由空间拓扑／通行性证明。
5. 半径／数量是离散贪心启发式，不是全局最优。紧物理容差下的大物体，尤其多个小球协作表示角邻域，仍需继续优化。
6. 所有球都可能在某个小预算／紧 cap 条件下合理地选最大版本；这不等于没有多半径机制。反过来，不能为了降低“最大球比例”而无谓缩球。
7. 下一步应优先让你检查盒子的棱中段和大面，而不是仅看角支持 8/8 就宣布碰撞检测已经足够。
