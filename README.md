# LatentSync-Realtime: Real-time 2D Digital Human Live Streaming Server

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="License">
  <img src="https://img.shields.io/github/stars/nxneo/LatentsyncRealtime?style=social" alt="GitHub Stars">
</p>

This project provides a production-ready, real-time inference server based on **LatentSync 1.0**, enabling high-quality, low-latency 2D digital human live streaming. We have built a sophisticated, production-grade streaming architecture to address the challenges of real-world live scenarios.

**[重要提示]** 本项目基于 **LatentSync 1.0** 版本构建。它不兼容 LatentSync 1.5 及 1.6 版本。LatentSync 1.0 的模型文件可以在 [Hugging Face](https://huggingface.co/ByteDance/LatentSync) 下载。我们欢迎社区成员尝试基于 1.5 版本进行实时化改造，但由于 1.6 版本推理耗时较长，我们认为它不适合用于实时交互场景。

[**[Important Note]** This project is based on **LatentSync version 1.0**. It is not compatible with versions 1.5 or 1.6. The model files for LatentSync 1.0 can be downloaded from [Hugging Face](https://huggingface.co/ByteDance/LatentSync). We encourage the community to adapt this real-time framework for version 1.5. However, we believe version 1.6 is not suitable for real-time interaction due to its longer inference time.]

---

## 效果演示 (Demo)

下面是本项目运行时的实际效果，展示了从**空闲待机流**到**接收任务**，再到**平滑过渡**并开始**实时推理**的全过程。
<center><video src=https://github.com/user-attachments/assets/9af5cc82-4da8-4cea-9b52-abc6c28ae69f controls preload></video></center>
---

## 主要功能 (Core Features)

*   **🚀 多GPU并行推理 (Multi-GPU Parallel Inference)**: 利用持久化的多 GPU 工作进程池 (`multiprocessing.Pool`)，将推理任务分发到所有可用 GPU，实现吞吐量最大化和延迟最小化。

*   **🎬 无缝码流切换 (Seamless Stream Switching)**: 实现了一个先进的推流控制器，能够在“空闲/占位视频”和“正式任务视频”之间进行无缝切换，保证直播流7x24小时不中断。

*   **✨ 平滑交叉淡入过渡 (Smooth Crossfade Transitions)**: 在空闲流与任务流切换时，通过可配置的平滑交叉淡-入淡出效果，消除了画面的生硬跳跃，确保了专业且视觉舒适的播出质量。

*   **🧠 智能队列与状态管理 (Intelligent Queue & State Management)**: 设计了多层、线程安全的队列系统和原子化的状态机，有效防止了在高并发下因“生产者（GPU）过快，消费者（RTMP流）过慢”而导致的丢帧和状态竞争问题。

*   **🔧 生产级服务架构 (Production-Ready Architecture)**: 基于 Flask 构建，为高可用性而设计，包含优雅停机、详细日志、以及在模型推理、流媒体管理和API层之间的清晰职责分离，可直接用于生产环境。

---

## 架构介绍 (Architecture Overview)

本项目的核心是一个解耦的、多阶段的异步处理流程：

1.  **API层 (Flask)**: 接收外部 HTTP 请求（如上传的音频文件），将任务打包后立即返回，实现非阻塞式响应。
2.  **任务分发层 (TaskProcessor)**: 为每个请求创建一个独立的任务处理器，它负责从多GPU工作池中收集乱序的推理结果，并按正确顺序重新排序。
3.  **推理层 (Multi-Process Pool)**: 一个常驻的多进程池，每个进程绑定一个 GPU 并预加载模型。它们是执行密集计算（唇形生成）的主力。
4.  **推流控制层 (StreamingController)**: 一个独立的线程，负责管理最终的 RTMP 推流。它从一个统一的媒体队列中消费数据，并智能地处理空闲流与正式流的切换逻辑，包括注入过渡动画。

这种分层解耦的设计保证了系统的高性能、高响应性和高稳定性。

---

## 核心代码文件 (Core Code Files)

相对于原始的 LatentSync 1.0 版本，我们的修改主要集中在以下两个文件：

1.  `infer_serv.py` (新增): 这是我们实现的服务入口和核心控制逻辑，包含了 Flask API、StreamingController 和 TaskProcessor。
2.  `latentsync/pipelines/lipsync_pipeline.py` (修改): 我们对原始的 Pipeline 进行了一些适配性修改，以便于在多进程环境中进行初始化和调用。

---

## 安装与运行 (Installation and Usage)

### 1. 环境与依赖安装
请首先参考官方 [LatentSync GitHub](https://github.com/bytedance/LatentSync) 项目的指引，完成环境的配置和所有依赖的安装。
或者参考如下步骤：
# Create a new conda environment
conda create -y -n latentsync python=3.10.13
conda activate latentsync

# Install ffmpeg
conda install -y -c conda-forge ffmpeg

# Python dependencies
pip install -r requirements.txt

# OpenCV dependencies
sudo apt -y install libgl1

# Download the checkpoints required for inference from HuggingFace
huggingface-cli download ByteDance/LatentSync-1.6 whisper/tiny.pt --local-dir checkpoints
huggingface-cli download ByteDance/LatentSync-1.6 latentsync_unet.pt --local-dir checkpoints

### 2. 模型下载与存放
从 [LatentSync Hugging Face](https://huggingface.co/ByteDance/LatentSync) 仓库下载模型文件。**请务必选择并下载 1.0 版本对应的模型**，并按照官方项目的目录结构要求存放它们。

通常，您需要将检查点文件放在 `checkpoints/` 目录下。

### 3. 启动服务
完成以上步骤后，直接运行 `infer_serv.py` 文件即可启动实时推理服务。

```bash
python infer_serv.py
```

服务启动后，它将默认在 `http://0.0.0.0:11001` 监听，并开始向您在代码中配置的 RTMP 地址推流（默认为空闲视频流）。

---

## 接口调用示例 (API Example)

您可以使用 `curl` 或任何 HTTP 客户端向 `/generate_video` 端点发送一个 `POST` 请求，并附上一个音频文件。

```bash
curl -X POST \
  -F "audio_file=@assets/audio/7s_shan_0801_01.wav" \
  http://127.0.0.1:11001/generate_video
```

服务器接收到请求后，将自动处理音频，生成口型动画，并平滑地将直播流切换到该任务的视频上。任务结束后，直播流会自动切换回空闲视频。

---

## 开源计划 (Roadmap)

我们计划在未来继续为该项目添加更多激动人心的功能，使其成为一个更完整的数字人解决方案：

*   **[ ] 2D数字人全身生成**: 对接 `majic-animate` `i2v` `emoh` 等服务，实现从口型驱动到全身动作的生成。
*   **[ ] 实时对话能力**: 对接大型语言模型 (LLM) API，如 GPT、ChatGLM 或本地模型，实现与数字人的实时语音对话。
*   **[ ] 移动端 Demo**: 开发一个手机 App，展示并与此后端服务进行交互，提供更直观的用户体验。

---
## 作者 (Author)

- **[Yingjie Zhang](https://github.com/nxneo)**
- nxoor2022@gmail.com
