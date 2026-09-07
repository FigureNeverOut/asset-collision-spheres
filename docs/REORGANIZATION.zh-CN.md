# 碰撞球代码集中管理（0.1.1）

碰撞球算法和预览功能现在统一维护于 `asset-collision-spheres`。
本次只调整本地文件、导入和安装关系，没有创建 Git 提交或推送远端。

## 在哪里修改

| 工作内容 | 当前目录或文件 |
| --- | --- |
| Python 对外接口 | `src/asset_collision_spheres/api.py` |
| 自动几何分球 | `src/asset_collision_spheres/algorithms/auto_geometry.py` |
| 手动区域分球 | `src/asset_collision_spheres/algorithms/mesh_regions.py` |
| 解析形状、实心盒分区 | `src/asset_collision_spheres/algorithms/` |
| 材质距离、球合法性、数据类型 | `src/asset_collision_spheres/geometry/` |
| 资产加载与单位处理 | `src/asset_collision_spheres/loaders/` |
| 通用静态预览、Isaac 查看器 | `src/asset_collision_spheres/preview/` |
| FastSim 预计算产物与挂载适配 | `src/asset_collision_spheres/adapters/fastsim.py` |
| 1e/1g 场景诊断、原生 cuRobo 对照 | `src/asset_collision_spheres/adapters/workspace/` |
| 杯、碗、盒子参数 | `configs/mug/`、`configs/bowl/`、`configs/box/` |
| 本机资产路径 | `configs/local/`（Git、wheel、sdist 均排除） |
| 算法、通用预览、工作区测试 | `tests/`、`tests/preview/`、`tests/workspace/` |

原 FastSim-Plugins 中五个同名模块、task2sim 中七个预览/健康检查脚本，以及独立包的旧顶层模块，
都只保留轻量导入转发。这些文件没有独立算法实现，后续不要在转发文件中添加算法。
旧模块与新模块指向同一对象，保持类身份、缓存和正常导入后的 monkeypatch 一致。

## 配置与运行

通用可共享参数在 `configs/` 按资产分组。保留冻结 mug 配置的原始字节，避免改变历史 SHA256。
碗、盒子的可共享 USD preset 使用相对资产路径；本工作区原有绝对路径保存在 `configs/local/`。
`config_path()` 默认优先寻找本地覆盖，再使用随包安装的通用配置。
显式 `--preset` 始终读取所指定的文件，相对资产路径以该配置所在目录为基准。

新环境在本仓库执行：

```bash
python -m pip install -c constraints-tested.txt -e '.[dev,usd]'
python -m pytest -q
python examples/create_demo_assets.py
asset-spheres --mesh outputs/demo/cup.obj --unit-scale-m 1 \
  --config configs/demo/demo_auto.json --output outputs/cup_spheres_new.json
```

已有 FastSim 环境，在工作区根目录执行：

```bash
python -m pip install --no-deps --no-build-isolation -e ./asset-collision-spheres
python -m asset_collision_spheres.preview.mesh --build-only
```

本工作区的独立 `.venv` 和 `fastsim_vnext` 已完成安装。
后者使用现有依赖，没有升级 Isaac、cuRobo 或其他仿真包。
Mission 已声明 `asset-collision-spheres>=0.1.1,<0.2` 依赖；此版本尚未发布到 PyPI，
因此其他机器要先从此仓库安装本地包，再安装/运行修改后的 Mission。

可选工作区入口（在已有仿真环境运行）：

```bash
python -m asset_collision_spheres.adapters.workspace.mug --help
python -m asset_collision_spheres.adapters.workspace.mug_auto --help
python -m asset_collision_spheres.adapters.workspace.book --help
python -m asset_collision_spheres.adapters.workspace.book_comparison --help
python -m asset_collision_spheres.adapters.workspace.curobo_matched --help
```

它们依然需要原项目的场景、资产与依赖。源码目录会识别同级工作区；其他部署方式用
`ASSET_SPHERES_WORKSPACE` 指定包含 task2sim 和 FastSim-Demo 的工作区根目录。

## 输出和历史结果

| 内容 | 默认位置 |
| --- | --- |
| 通用单资产预览 | `outputs/<preset名称>/` |
| 杯子诊断 | `outputs/mug/` |
| 书本诊断 | `outputs/book/` |
| cuRobo 同预算对照 | `outputs/comparisons/` |

源码运行时输出根目录为本仓库 `outputs/`，与调用时的当前目录无关。
普通 wheel 安装默认使用当前目录的 `outputs/`。
`ASSET_SPHERES_OUTPUT_DIR` 统一覆盖输出根目录；已有 `--output` 选项继续支持单次覆盖。
`ASSET_SPHERES_CONFIG_DIR` 可以指定本地配置覆盖目录，内部沿用 `bowl/...` 等相对结构。

历史运行结果、已冻结的手动球与原生对照报告继续保留原位。
这些报告记录了绝对路径和文件哈希，移动会破坏其验证关系；本次没有重写或重新生成它们。
对照脚本继续读取这些历史输入，新结果写入新输出根目录。
旧脚本入口继续可用，但打开原来的预览需要传入 `--output` 指向原目录。

## 回归与版本控制

算法函数体与原版本保持一致，改动集中于模块依赖、入口和路径解析。
通用预览与 API 共用同一 USD loader，保留原有单位、变换、绕序和尺寸检查。
查看器使用延迟导入，避免在 SimulationApp 初始化之前加载独立 USD/数值库。
wheel 和 sdist 明确只包含可共享配置，排除本地路径覆盖。

FastSim 原项目保留后端集成测试；独立包维护算法、预览和场景诊断测试。
`tests/workspace/` 在缺少原工作区或 FastSim 环境时跳过，通用功能不依赖它。
本次静态回归不等于重新完成 GPU 原生挂载或完整物理任务。

迁移映射见 [reorganization_map.json](reorganization_map.json)。
首次提取的哈希清单 `extraction_manifest.json` 保留为 0.1.0 历史记录，
不能再拿其中旧路径的哈希校验当前转发文件。

本次修改涉及独立包、FastSim-Plugins 和 task2sim 三个 Git 工作树。
后续同步到其他机器时需一起同步相应调用改动；只推独立包不会自动更新另外两个仓库。


## 本次验证结果

- `fastsim_vnext` 运行独立仓库全部测试：**113 passed**。
- 独立 `.venv`：**100 passed, 9 skipped**；跳过的是缺少 FastSim/PyYAML 集成环境的场景测试（含模块级跳过）。
- Mission 挂载、配置、可视化回归：**51 passed, 2 skipped**；两个实际 G2/CUDA 挂载测试未运行。
- Mission 到独立包的模块身份集成检查：**5 passed**。
- 七个 task2sim 旧脚本入口的 `--help` 全部通过。
- 64 个算法/材质查询定义的 AST 与迁移前一致。
- 通用预览 CLI 已从另一个工作目录读取相对 USD preset 并生成 8 球 USD/JSON。
- wheel 安装副本可独立导入并找到配置；wheel/sdist 均排除了本机配置与路径。

本机迁移前备份位于工作区 `_project_docs/out/sphere_reorganization_backup_20260907_140514.tar.gz`。
该备份不属于待上传的碰撞球仓库。

## 迁移之后的算法增量

上述 113 项测试和 AST 一致性描述的是本次目录迁移时的状态。
后续新增的 box-like 特征、球大小／数量联合策略，以及真实 native 验证，见
[JOINT_V2.zh-CN.md](JOINT_V2.zh-CN.md)。新实现仍在本仓库；旧配置和冻结产物保留。
