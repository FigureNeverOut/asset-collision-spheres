# 同事直接调用的源码接口

本轮收拢调用入口，保留已认可的分球策略，并兼容不同Trimesh版本的空射线结果。
无需安装本项目、不提供HTTP服务；
同事已有 NumPy、SciPy、Trimesh、Rtree 环境即可。USD输入额外使用已有pxr。

## 1. 源码入口

```python
import sys
sys.path.insert(0, "/path/to/asset-collision-spheres/src")

from asset_collision_spheres import (
    GenerationConfig,
    SphereResult,
    generate,
    generate_from_arrays,
    generate_from_file,
)
```

把路径替换为同事实际存放源码的目录。也可统一设置PYTHONPATH，此时不用写sys.path。
使用insert(0)是为了明确选中这份源码，避免环境里旧版本同名模块优先。
如果进程已导入旧版本，需要重启进程；不要在运行中的仿真程序里强行切换模块。

## 2. 三种输入

### 已经有trimesh对象

```python
result = generate(mesh, config={"max_spheres":32})
```

mesh必须是单个 `trimesh.Trimesh`，单位已经是米，坐标系已经是你希望输出球使用的坐标系。
接口不自动居中、猜单位、合并Scene或变换机器人姿态。

### 已经有顶点和面

```python
result = generate_from_arrays(
    vertices,             # [V,3]
    faces,                # [F,3]，从0开始的整数三角形索引
    unit_scale_m=0.001,    # 输入毫米转米；输入米则为1
    config={"max_spheres":32},
)
```

输入数组不被修改；坐标绕现有原点等比例缩放，不重置物体原点。
非法面索引、非整数索引、非有限顶点、非法单位比例会报ValueError。

### 文件

```python
result = generate_from_file("object.obj", unit_scale_m=0.001,
                            config={"max_spheres":32})
```

支持OBJ/PLY/STL，不支持自动把多个物体的Scene拼成一个物体。
USD / USDA / USDC：

```python
result = generate_from_file(
    "object.usd", unit_scale_m=0.001,
    mesh_prim="/World/Object/Mesh",
    config={"max_spheres":32},
)
```

必须显式选择一个三角Mesh。不猜哪个prim是目标，也不拿显示用多边形扇形剖分充当原表面。
USD应用作者变换和单位，并转换到Z-up；必要时可用up_axis="Y"或"Z"显式覆盖源向上轴。
普通mesh/数组/OBJ文件的坐标标记为 `input_mesh_axes_m`，USD为
`usd_stage_transformed_z_up_m`。后者不是自动生成好的机器人挂载局部坐标。

## 3. 配置就四项

```python
config = GenerationConfig(
    geometry_policy="auto",
    max_spheres=32,
    shape_hint="auto",
    seed=5,
)
result = generate(mesh, config=config)
```

config可为该类型、普通字典或None。None默认auto、64、auto、5。
字典中可省略任何一项；`mode="auto_geometry"`是可选兼容字段。

| 参数 | 允许值 | 默认 | 作用 |
|---|---|---|---|
| geometry_policy | auto / convex_hull / original_surface | auto | 选择自动路由、强制凸包、强制原表面 |
| max_spheres | 整数1–64 | 64 | 数量上限，规则网格可以不用满 |
| shape_hint | auto / box / open_box / generic | auto | 结构提示，不能替代路由策略 |
| seed | 非负整数 | 5 | 固定输入与依赖下复现采样 |

不接受sphere_budget、backend_sphere_capacity、regions或自定义radius等旧模式字段。
如果要用历史材料算法，使用保留的旧接口，不把两套参数混在一起。
不需要杯沿、把手、角点标注；半径沿用已有共享策略。

## 4. 返回值

`SphereResult`提供：

| 属性/方法 | 含义 |
|---|---|
| spheres | float64 NumPy [N,4]，每行cx/cy/cz/r，全部米制 |
| count | 实际球数 |
| selected_policy | 最终convex_hull或original_surface |
| review_required | 是否存在需要人工复核的路由/几何诊断 |
| regions | 每个球的采样来源标签，不是物体语义分类 |
| diagnostics | 完整原算法诊断，包含路由证据和采样指标 |
| coordinate_frame | 明确的输出坐标系标签 |
| mesh_sha256 | 输入米制网格指纹 |
| config | 解析后的GenerationConfig |
| source | 可选来源/单位/变换元数据；不包含本机文件路径 |
| to_dict() | 独立的JSON兼容字典，schema为asset-collision-spheres/1 |
| save_json(path) | 保存该字典；已有文件拒绝覆盖 |

```python
spheres_for_collision = result.spheres
centers_m = result.spheres[:, :3]
radii_m = result.spheres[:, 3]
result.save_json("outputs/object_spheres.json")
```

dataclass属性不可重绑定，但内部数组/字典仍可修改；如需保留原结果请使用copy。
to_dict返回独立数据，修改它不会改变原诊断。
检查通过不等于碰撞完备；球并集有缺口，有限半径也可能缩窄凹槽。

## 5. 命令行：不安装也能跑

```bash
python /path/to/asset-collision-spheres/generate_spheres.py \
  --mesh object.obj --unit-scale-m 0.001 \
  --max-spheres 32 --output outputs/result.json
```

这个根目录脚本只负责定位同目录src并调用CLI，没有第二份算法。
可以从其他目录运行；输入输出相对路径以当前工作目录为准。
普通输入不传config默认v4 auto；可传--geometry-policy、--shape-hint、--seed。
已有--config和--preset调用仍保持历史语义，文件已有generation时不会偷偷改默认路线。

## 6. 仿真接线与兼容

现有FastSim adapter仍调用原内部入口，本轮不用修改任务run.yaml。
新接口只是把同一生成结果整理成方便同事消费的对象，并不是一个新的native attachment。
同事自己接机器人时仍须负责object-to-link变换、native容量、attach/detach生命周期。
通用JSON不是FastSim预计算挂载artifact，不能直接混用两个schema。

旧generate_auto_geometry_spheres、generate_convex_surface_spheres、
generate_geometry_spheres、generate_mesh_region_spheres、make_report、load_mesh及旧模块别名均保留。
新generate默认v4，但旧函数/旧配置的默认行为没有改变。

本次统一接口对所有策略与32/64上限检查了与原入口的球数组一致性；
历史495个冻结文件不因本轮接口调整改写。测试/交接结果见源码交接文档。
