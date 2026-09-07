# Asset Collision Spheres

将完整物体三角网格转换为碰撞球的独立 Python 接口与命令行工具。
核心生成只使用 NumPy、SciPy、Trimesh、Rtree；无需 FastSim、cuRobo、Isaac Sim、机器人或 GPU。
USD 输入可选依赖 OpenUSD (`usd-core`)。

当前为实验版本：支持自动几何分球、手动区域分球，以及底层解析几何接口。
输入必须是封闭、朝外且绕序一致的材质边界；带杯腔的实体杯壁可以满足这一条件。
不自动填洞或用凸包替换原网格。默认入口不是 HTTP 服务，可由其他项目直接 import。

## 安装与最小示例

Python 3.11+，在本仓库根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -c constraints-tested.txt -e '.[dev]'
python examples/create_demo_assets.py
asset-spheres --mesh outputs/demo/cup.obj --unit-scale-m 1 \
  --config examples/configs/demo_auto.json --output outputs/cup_spheres.json
python -m pytest -q
```

示例脚本自行生成杯、碗和盒子，无需下载资产。输出路径已存在时 CLI 会拒绝覆盖。
`python -m asset_collision_spheres` 与 `asset-spheres` 等价。

## Python 接口

```python
import json
from pathlib import Path
from asset_collision_spheres import load_mesh, generate_auto_geometry_spheres, make_report

mesh = load_mesh("outputs/demo/cup.obj", unit_scale_m=1.0)
config = {
    "sphere_budget": 32,
    "max_outward_offset_m": 0.004,
    "candidate_sample_count": 6000,
    "seed": 5,
}
result = generate_auto_geometry_spheres(mesh, config)
print(result.spheres.shape)  # (32, 4): x, y, z, radius，全部为米
Path("outputs/api_result.json").write_text(
    json.dumps(make_report(mesh, result, config), indent=2, allow_nan=False)
)
```

也可直接传入已在所需局部坐标系下、以米为单位的 `trimesh.Trimesh`。
直接传入网格时生成器不变换坐标；`load_mesh` 保留文件轴向并应用显式单位比例。
OBJ/PLY/STL 读取会合并完全相同的顶点，恢复 STL 等格式的共享顶点拓扑；不会进行近似焊接或几何修补。
多几何场景应先显式导出为一个材质网格。

## 参数与诊断

| 参数 | 含义 |
| --- | --- |
| `sphere_budget` | 显式球数，1–256；候选不足时报错 |
| `max_outward_offset_m` | 显式外扩上限，无默认值；示例值不是通用规划安全距离 |
| `candidate_sample_count` | 面积采样数，500–20000，默认 6000 |
| `patch_count` | 三角形邻接图上的局部分块数，默认 64 |
| `balance_strength` | 局部覆盖均衡权重，默认 1.5 |
| `mode` | `auto_geometry`；`global` 是共享候选池的消融对照 |
| `repair_rounds` | 同预算替换修复轮数，0–4，默认 0 |
| `seed` | 固定随机种子；复现比较时也应固定依赖版本和网格顶点顺序 |

报告包含 `spheres_m`、坐标系、网格指纹、配置、区域及评估信息。
关注 `missing_material_components`、`zero_contact_patch_ids`、`surface_coverage`、
`ordinary_contact_misses` 与 `free_probe_intrusions_beyond_allowance`。
`finite_checks_passed` 只表示有限几何检查通过；不证明全表面覆盖或连续运动无碰撞。
尚无完整自相交检测。自动生成会保留遗漏整块材质的结果并置 `review_required`，供诊断使用。

## 使用现有 bowl / mug 配置

`examples/configs/` 保留已有算法参数。碗和盒子的 USD preset 已将本机绝对路径改为
相对于配置文件的 `../../assets/.../Aligned.usd`；资产本身不在仓库内。
安装 USD 支持后，将资产放在对应目录，或修改 `source.usd_path`：

```bash
python -m pip install -c constraints-tested.txt -e '.[usd]'
asset-spheres --preset examples/configs/bowl_001_auto_geometry.json \
  --budget 32 --output outputs/bowl_001_spheres.json
```

USD loader 保留原静态预览的规则：选择一个明确的三角 Mesh，应用作者变换和显式单位比例，
检查物理尺寸，然后转换到 Z-up。输出标记为 `usd_stage_transformed_z_up_m`，
并记录变换矩阵；它不是自动计算好的机器人 attachment 局部坐标。
`mesh_prim`、`unit_scale_m`、尺寸检查必须与实际资产一致。

Mug 的 auto JSON 是纯生成配置，可配合已导出的完整材质 OBJ 使用 `--mesh --config`。
Mug 手动区域配置带原资产尺寸与坐标假设，不能直接套到任意杯子上。
CLI 在配置含 `regions` 时调用 `generate_mesh_region_spheres`，否则调用自动生成器。

## 模块与接入边界

| 模块 | 作用 |
| --- | --- |
| `auto_geometry_spheres.py` | 自动几何分球与独立采样评估 |
| `mesh_region_spheres.py` | 手动区域预算、材质深度与几何工具 |
| `mesh_attachment.py` | 球合法性、网格指纹、预计算产物校验；不依赖仿真器 |
| `attachment_spheres.py` | 历史解析形状接口（盒、开口盒、无把手杯等） |
| `solid_box_regions.py` | 实心盒区域配置生成 |
| `usd.py` | 可选 USD 加载；从原静态预览提取 |

原生 attachment 使用的 `fastsim/mesh-spheres/1` 保留在底层兼容代码中。
本工具输出 `asset-collision-spheres/1`，两者不能通过改字段名直接互换：
需先在调用方完成坐标转换、网格指纹与球容量核对。
本次未改动原 FastSim 调用路径；原项目继续使用其原有模块，独立包后续修改也不会自动同步回原项目。
Isaac 静态查看器、机器人规划、任务流水线、仿真数据集与运行日志不包含在此包内。

## 来源与发布

提取范围和原文件 SHA256 见 [extraction_manifest.json](docs/extraction_manifest.json)。
五个核心模块逐字复制自本地新增代码，原测试仅替换导入路径。
保留原仓库 LICENSE 的 pending 状态，未擅自授予新的开源许可；公开发布前由权利人确定授权范围与许可证。
GitHub 操作见 [发布指南](docs/GITHUB_PUBLISH.zh-CN.md)。
