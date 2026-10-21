# Third-party notices

MealVLA 使用以下开源软件、模型与数据。仓库中的 MIT 许可证适用于本仓库代码；第三方资源保留各自的许可证和使用条件。

| 项目 | 用途 | 来源 |
| --- | --- | --- |
| LeRobot 0.4.3 | SmolVLA 实现、训练器、数据格式与环境封装 | [源码](https://github.com/huggingface/lerobot/tree/v0.4.3) |
| SmolVLA | 预训练视觉—语言—动作模型 | [Model card](https://huggingface.co/lerobot/smolvla_base) |
| LIBERO / hf-libero | 仿真任务、初始状态与场景资产 | [源码](https://github.com/Lifelong-Robot-Learning/LIBERO) |
| HuggingFaceVLA/libero | 公开示教数据 | [Dataset card](https://huggingface.co/datasets/HuggingFaceVLA/libero) |
| PyTorch / torchvision | 张量计算、训练与图像处理 | [项目主页](https://pytorch.org/) |
| NumPy | 数组处理与统计 | [项目主页](https://numpy.org/) |
| ROS 2 | 可选消息接口 | [Humble 文档](https://docs.ros.org/en/humble/) |

LeRobot 采用 Apache-2.0 许可。模型权重、数据集和场景资产的具体许可以相应版本的 model card、dataset card 和 LICENSE 为准；本仓库不分发这些权重或数据。

## 集成约定

本仓库的新增实现包括数据划分与重新导出、模型/环境适配、动作执行约束、回合重试和评估报告。基础模型架构与预训练权重来自 SmolVLA，训练流程使用 LeRobot。

LIBERO 的图像旋转、状态表示和动作空间遵循上游约定。接口依据：

- [SmolVLA policy](https://github.com/huggingface/lerobot/blob/v0.4.3/src/lerobot/policies/smolvla/modeling_smolvla.py)
- [Policy processors](https://github.com/huggingface/lerobot/blob/v0.4.3/src/lerobot/policies/factory.py)
- [LIBERO environment](https://github.com/huggingface/lerobot/blob/v0.4.3/src/lerobot/envs/libero.py)
- [Observation processing](https://github.com/huggingface/lerobot/blob/v0.4.3/src/lerobot/processor/env_processor.py)
- [Dataset implementation](https://github.com/huggingface/lerobot/blob/v0.4.3/src/lerobot/datasets/lerobot_dataset.py)
