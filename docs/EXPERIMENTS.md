# 评估协议

## 任务与初始状态

默认配置选择 `libero_object` 的 task IDs `[0,1,2]`，每个任务使用 20 个官方初始状态，随机种子为 `42 + initial_state`。总计 60 个场景，任务名和语言指令沿用 LIBERO 定义。

执行 horizon 为 10，每回合最多 280 步，默认不重试。这是自定义子集协议；任务范围和完整 LIBERO benchmark 不同，不能直接比较套件平均分。公开初始状态与训练示教可能重叠，因此该协议不用于宣称未见场景泛化。

## 对照实验

| 实验 | 变量 | 固定项 |
| --- | --- | --- |
| 动作执行步数 | horizon = 1、5、10 | checkpoint、任务、种子、最大步数 |
| 训练阶段 | 不同 step 的 checkpoint | 数据划分、评估配置 |
| 重置重试 | max_retries = 0、1 | checkpoint、初始场景、horizon |

这些实验可通过独立的配置文件运行，每次使用不同输出目录。语言改写与同场景目标切换属于额外的对照实验，不在默认协议中。

## 指标

| 字段 | 定义 |
| --- | --- |
| `first_attempt_success_rate` | 首次执行成功的场景数 / 已评估场景数 |
| `success_rate` | 在重试预算内成功的场景数 / 已评估场景数 |
| `success_rate_ci95` | 按场景计算的 Wilson 95% 置信区间 |
| `inference_ms` | 预处理、策略推理和结果回传的 p50/p95 |
| `control_step_ms` | 仿真 step 调用耗时的 p50/p95 |
| `mean_episode_s` | 包含重置重试的平均场景耗时 |
| `reset_retries` | 实际发生的场景重置次数 |

重试不增加成功率分母。任务、初始状态与种子相同的重复记录会被拒绝；不同 policy/environment 类型的记录不能合并。

`complete=false` 或 manifest 为 `aborted` 时，结果属于部分运行。模型加载时间不计入回合推理延迟，仿真 step 耗时也不代表真机伺服延迟。Mock 输出单独标记，不参与模型评估。

## 结果

结果由真实运行记录生成，不随源码预填。归档实验时需同时提供以下信息：

- checkpoint、代码版本和运行配置；
- 数据 revision 与 episode 划分 manifest；
- 成功数/总数、首次与最终成功率及置信区间；
- 推理耗时和逐回合失败记录；
- 是否启用重试，以及运行是否完整。

原始记录采用 `episodes.jsonl`，聚合结果为 `summary.json`；评估同时导出 `episodes.csv` 和 `report.html`。流水线统一保留配置、阶段日志和数据 manifest。建议将 checkpoint 和完整环境列表与实验记录共同归档。

启用录制时，每次尝试对应独立 MP4，`videos` 字段保存相对路径。`recording_s` 为逐帧写入及编码器关闭的累计时间；回合总耗时包含该开销，模型推理和仿真 step 的计时不包含视频写入。进行延迟对比时应使用一致的录制设置。
