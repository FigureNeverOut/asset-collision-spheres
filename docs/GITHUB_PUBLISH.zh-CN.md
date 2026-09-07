# 源码交接与后续GitHub准备

## 本轮范围

按用户最新要求，只整理本地源码接口，供已有环境的同事直接调用。
不制作安装包、不新增安装流程、不创建Git提交、不推送，不改变仓库可见性。
现有origin是 `FigureNeverOut/asset-collision-spheres`；远端仍是此前版本，
本轮源码和已有v2–v4增量尚未上传。不要向同事声称GitHub已经包含这些改动。

## 同事如何使用

交接本仓库源码（至少保留src完整目录；命令行还需要根入口generate_spheres.py）。
同事在自己的程序里设置：

```python
import sys
sys.path.insert(0, "/path/to/asset-collision-spheres/src")
from asset_collision_spheres import generate_from_file

result = generate_from_file("object.obj", unit_scale_m=0.001,
                            config={"max_spheres":32})
spheres = result.spheres  # [N,4]，米制
```

不用pip install本项目。同事的原有环境需能导入numpy、scipy、trimesh、rtree；
读取USD才需要pxr。机器人和Isaac不是核心算法的必要环境。
命令行：`python generate_spheres.py --help`。
完整配置及坐标说明见 [API.zh-CN.md](API.zh-CN.md)。

## 源码范围

- 对外：api.py、contracts.py、service.py、根generate_spheres.py。
- 内部：algorithms/、geometry/、loaders/。
- 可选：preview/、adapters/，仍依赖对应工作区/资产，不能声称通用环境也能跑原1e/1g。
- 历史示例与回归：原有examples/、tests/不动；本轮临时接口测试与演示脚本验证后清理。
- 不交接本机资产、outputs、configs/local、缓存、虚拟环境和本地状态笔记。

`.gitignore`排除这些文件；已有本地状态笔记保留原位，没有删除。
与本机路径绑定的参数在configs/local中；共享文档使用可替换路径。

## 后续公开前要确认

1. 选择合适且有权授予的许可证。当前LICENSE明确为pending，本轮未替你选择。
2. 确认准备上传独立仓库的哪些改动。该仓库还有此前迁移及v2–v4未提交增量。
3. 核查git diff和新增文件，不上传机器私有资产、日志、令牌或凭据。
4. 本机曾有独立包安装，因此要像本轮测试一样确认同事实际加载的是交接源码，而非环境里的旧包。
5. 再明确授权创建提交和推送。当前终端对该origin也缺少可用Git凭据；
   此事留待实际推送时处理，不把凭据写到脚本或remote URL。

公开仓库不自动代表已授予开源使用许可。当前阶段不创建Release，不发布PyPI。

## 验证方式

现有环境、仓库根目录：

```bash
PYTHONPATH=src python -m pytest -q
python generate_spheres.py --help
python examples/freeze_geometry_reference.py
```

无USD或无原工作区时，对应可选测试会跳过，核心API仍可使用。
运行原有弯管/弯杆合成样例还需要NetworkX（Trimesh修复法线所用）；核心接口不要求它。

本轮验证（2026-09-07）：

- 现有fastsim_vnext环境：原回归加临时接口测试共250项通过。
- 独立Python环境：38项临时接口测试通过，包括复制源码到独立目录、不安装本项目直接调用，
  并检查未导入FastSim、cuRobo、Isaac、Torch。
- 独立环境扩展回归：231项通过、12项可选集成跳过；3项依赖NetworkX的合成样例未选入，
  没有为测试额外安装依赖。这3项已在fastsim_vnext完整测试中通过。
- 16个已有样例分别检查32/64上限，共32组球数组与上一轮历史结果逐项完全一致；
  495个pre-v4冻结文件校验未变。
- 验证过程中补充Trimesh空射线结果兼容处理；分球策略和阈值未调整。
- 清理临时测试代码后，重新运行保留的原有回归：212项全部通过。

按用户要求，本轮临时测试代码、演示脚本在验证后删除，测试日志移入回收站，以上为当时执行结果，
不是保留测试集的数量。原有测试、历史算法实验与预览产物保留。
