# Asset Collision Spheres

把物体三角网格转换为米制碰撞球的 Python 接口。面向已有环境的同事，
**拿到源码即可调用，不需要安装本项目，不提供 HTTP 服务。**

核心只需要现有环境中的 NumPy、SciPy、Trimesh、Rtree；不依赖机器人、GPU、
FastSim、cuRobo 或 Isaac。读取 USD 时才需要 `pxr`。

本轮只整理本地源码，**没有提交或推送 GitHub**。LICENSE 仍待确定；
后续公开仓库前需要确定授权范围与许可证。

## 最简单的调用

同事在自己的程序里指定一次本项目的 `src` 目录：

```python
import sys
sys.path.insert(0, "/path/to/asset-collision-spheres/src")

from asset_collision_spheres import generate_from_file

result = generate_from_file(
    "object.obj",
    unit_scale_m=0.001,        # 原文件是毫米；如果已经是米，填1
    config={"max_spheres": 32},
)
print(result.spheres)          # NumPy [N,4]，每行 x、y、z、半径，全部为米
print(result.selected_policy)  # convex_hull 或 original_surface
print(result.review_required)
result.save_json("outputs/result.json")  # 已有文件不覆盖
```

也可以在环境中设置 `PYTHONPATH=/path/to/asset-collision-spheres/src`，然后直接import。
无需执行 `pip install .`。不要把未转换单位的毫米坐标直接传给米制网格接口。

## 三个入口

```python
from asset_collision_spheres import generate, generate_from_arrays, generate_from_file

# 已有 trimesh.Trimesh，必须已在所需坐标系下且单位为米
result = generate(mesh, config={"max_spheres": 32})

# 已有顶点和三角面数组；faces是从0开始的整数索引
result = generate_from_arrays(vertices, faces, unit_scale_m=0.001,
                              config={"max_spheres": 32})

# OBJ/PLY/STL文件
result = generate_from_file("object.obj", unit_scale_m=0.001,
                            config={"max_spheres": 32})
```

USD额外提供 `mesh_prim="/World/Object/Mesh"`。USD会应用作者变换并转换到Z-up，
请检查 `result.coordinate_frame`；它不是自动计算好的机器人attach局部坐标。

## 配置

| 字段 | 默认值 | 含义 |
|---|---|---|
| geometry_policy | auto | 保守自动路由；或指定convex_hull / original_surface |
| max_spheres | 64 | 整数1–64，上限而非一定用满 |
| shape_hint | auto | auto / box / open_box / generic；结构提示，不强制路由 |
| seed | 5 | 固定输入及依赖版本时可复现 |

不需要人工提供杯沿、把手、角点、区域配额或每类半径。
可不传config使用默认值；也可使用 `GenerationConfig(max_spheres=32)` 获得类型提示。

## 不写程序也能运行

在已有Python环境中：

```bash
cd asset-collision-spheres
python generate_spheres.py --mesh object.obj --unit-scale-m 0.001 \
  --max-spheres 32 --output outputs/result.json
```

根目录入口会自动定位本项目源码，也支持从其他目录用绝对脚本路径执行。
不指定配置文件时默认新auto路由；历史 `--config` / `--preset` 调用保持原语义。

## 结构与兼容性

```text
src/asset_collision_spheres/
├── api.py        # 统一对外导出
├── contracts.py  # GenerationConfig / SphereResult
├── service.py    # 网格、数组、文件入口；只做校验、加载与分派
├── algorithms/   # 既有v1/v2/v3/v4算法，保留分球策略
├── geometry/     # 几何基础工具
├── loaders/      # 文件、单位和坐标处理
├── preview/      # 可选预览
└── adapters/     # 可选FastSim/cuRobo接线
```

旧公开函数、旧模块别名、仿真adapter保留，不用修改原任务。
新统一入口默认v4；旧生成函数没有因为接口整理而切换默认算法。
稀疏球仍有局部漏检和槽口收缩风险，不代表零漏检或完整动态任务成功。

详细说明：[接口与参数](docs/API.zh-CN.md)、[代码分层](docs/ARCHITECTURE.md)、
[源码交接与后续GitHub准备](docs/GITHUB_PUBLISH.zh-CN.md)。

算法与已有实验：[几何路由v4](docs/GEOMETRY_ROUTER_V4.zh-CN.md)、
[凸包v3](docs/CONVEX_SURFACE_V3.zh-CN.md)、[材料v2](docs/JOINT_V2.zh-CN.md)、
[原有目录迁移](docs/REORGANIZATION.zh-CN.md)。

本机资产在 `configs/local/`，结果在 `outputs/`，均不上传；
现有环境/构建配置保留兼容，但本轮不生成安装包或新增安装流程。
