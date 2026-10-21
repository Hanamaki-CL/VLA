# 开源来源与贡献边界

MealVLA 是围绕开源 VLA 构建的个人应用工程项目。MIT 许可证只覆盖本仓库新增代码，不重新许可第三方模型、数据、软件或资源。

| 上游 | 用途 | 来源 |
| --- | --- | --- |
| Hugging Face LeRobot 0.4.3 | 模型实现、训练器、数据格式、LIBERO 封装 | https://github.com/huggingface/lerobot/tree/v0.4.3 |
| SmolVLA | 预训练模型和动作生成策略 | https://huggingface.co/lerobot/smolvla_base |
| LIBERO / hf-libero | 机器人仿真任务、初始状态和资产 | https://github.com/Lifelong-Robot-Learning/LIBERO |
| HuggingFaceVLA/libero | 公开示教数据 | https://huggingface.co/datasets/HuggingFaceVLA/libero |
| PyTorch / torchvision | 训练、张量和图像计算 | https://pytorch.org/ |
| NumPy | 数据校验与统计 | https://numpy.org/ |
| ROS 2 | 可选 dry-run 消息接口 | https://docs.ros.org/en/humble/ |

依赖通过安装脚本获取，不复制整个上游仓库，不把权重包含在分发包中。LeRobot 的 Apache-2.0 许可和其他上游各自的 LICENSE/model card/data card 继续适用。发布衍生权重、数据或视频前，核对相应版本的许可和归属要求；本文件不替代上游许可文本。

本仓库新增部分：配置与命令封装、episode 划分和重新导出、输入输出契约、动作 TTL 与执行队列、回合重试记录、评估报告、ROS 2 dry-run 接口、中文使用与项目说明。

本仓库未贡献：基础 VLM/VLA 的预训练、SmolVLA 架构、LIBERO 数据集/仿真器、通用流匹配算法。相机旋转及状态表示遵循 LeRobot 的 LIBERO 处理约定。

接口核对使用的主要源码：

- [SmolVLA policy](https://github.com/huggingface/lerobot/blob/v0.4.3/src/lerobot/policies/smolvla/modeling_smolvla.py)
- [Policy processors](https://github.com/huggingface/lerobot/blob/v0.4.3/src/lerobot/policies/factory.py)
- [LIBERO environment](https://github.com/huggingface/lerobot/blob/v0.4.3/src/lerobot/envs/libero.py)
- [Observation conventions](https://github.com/huggingface/lerobot/blob/v0.4.3/src/lerobot/processor/env_processor.py)
- [Dataset implementation](https://github.com/huggingface/lerobot/blob/v0.4.3/src/lerobot/datasets/lerobot_dataset.py)

核对日期：2026-10-21。阅读源码不等于已经完成安装、训练和运行验证。
