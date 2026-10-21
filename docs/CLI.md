# 命令参考

安装后使用 `mealvla`，也可使用 `python -m mealvla`。所有命令支持 `--help`。

| 命令 | 功能 | 主要选项 |
| --- | --- | --- |
| `doctor` | 软件版本与来源信息 | `--cuda-check` 执行 CUDA 前向/反向检查 |
| `tasks` | 列出任务 ID 与语言指令 | `--suite libero_object` |
| `prepare` | 划分并导出示教数据 | `--config configs/data.json` |
| `train` | 启动微调 | `--config configs/train_5090.json`、`--print-only` |
| `evaluate` | 对 checkpoint 进行仿真评估 | `--config configs/eval_libero.json` |
| `pipeline` | 串联数据、训练、评估 | `--config`、`--output`、`--reuse-data`、`--print-plan` |
| `report` | 从原始记录重新导出 HTML/CSV/JSON | 位置参数：评估目录 |
| `demo` | 固定动作 mock 联调 | `--output`、`--record-video` |

## 流水线

```bash
mealvla pipeline --print-plan
mealvla pipeline
mealvla pipeline --reuse-data --output outputs/experiments/run02
```

`--print-plan` 只解析配置，不下载模型、不准备数据、不启动训练，也不创建运行目录。独立 `train --print-only` 会额外检查训练集文件，两者用途不同。

配置路径规则：

- `data_config`、`train_config`、`eval_config` 相对流水线配置文件所在目录解析。
- `project_root` 同样相对流水线配置文件解析，默认 `..`。
- 数据输出和默认实验输出相对 `project_root` 解析；命令行 `--output` 相对当前工作目录解析。
- 流水线重写训练数据路径、训练输出、评估 checkpoint 和评估输出，快照保存在运行目录的 `configs/` 下。
- 基础模型可以使用 Hub ID；显式的 `./`、`../` 或绝对路径按本地模型路径处理。

数据和评估配置必须使用相同 suite 与任务列表。每次运行使用新实验目录；已有数据只能通过 `--reuse-data` 显式复用，并检查完成状态、来源、任务、种子与子集文件。指定 revision 时必须等于导出 manifest 中的已解析 revision，建议使用完整 Hub commit SHA。revision 留空表示接受该本地导出的版本，不表示自动检查 Hub 最新版本。

各阶段日志位于 `logs/`；例如 `tail -f outputs/experiments/mealvla_seed42/logs/train.log`。失败后保留已有产物，`pipeline.json` 标记为 `aborted`。Linux 上中断流水线会终止其正在运行的子进程组。改用新实验目录即可重新运行；训练恢复由上游 trainer 处理。

## 视频与报告

在评估配置中设置 `"record_video": true`。视频依赖包含在 `.[robot]` 中，也可以独立安装 `python -m pip install -e '.[video]'`。

```bash
mealvla demo --output outputs/mock-video --record-video
mealvla report outputs/mock-video
```

每段 MP4 横向排列第三人称和腕部相机，以 `control_hz` 作为播放帧率。包含重置帧和每次动作后的帧；重试独立成片。视频播放时间表示控制步序列，不代表包含推理等待的墙钟时间。

报告无需服务器或互联网。`report` 会重写派生的 `summary.json`、`episodes.csv` 和 `report.html`，原始 `episodes.jsonl` 与 manifest 保留。空的中断运行也可以导出报告；损坏的 JSONL 记录会报出行号，不会静默跳过。

## 退出码

正常结束返回 0；配置、依赖或子进程错误返回 1；用户中断返回 130。源码分发包含配置和脚本；wheel 安装提供 Python 模块及 CLI，示例配置从仓库获取。
