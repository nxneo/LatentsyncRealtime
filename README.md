# LatentSync-Realtime: Real-time Inference Server for 2D Digital Human Live Streaming

<p align="center">
  <img src="YOUR_PROJECT_DEMO_GIF_URL_HERE" width="800">
</p>
<p align="center">
  <em>An example of seamless crossfade transition between idle and formal streams.</em>
</p>

This project provides a production-ready, high-performance real-time inference server based on the **LatentSync 1.0** framework. It is specifically designed for live streaming applications like 2D digital humans, virtual anchors, and AI-driven avatars, enabling high-quality, low-latency lipsync generation.

We have built a sophisticated, production-grade streaming architecture that addresses common challenges in real-world live scenarios, transforming the original offline inference pipeline into a robust real-time service.

**Important Note:** This real-time implementation is based on **LatentSync 1.0**. It is not compatible with LatentSync versions 1.5 or 1.6. We encourage the community to adapt this real-time architecture for version 1.5. However, due to the longer inference times of version 1.6, it is not well-suited for real-time interactive applications.

## Core Features

1.  **🚀 High-Performance Multi-GPU Inference**: Utilizes a persistent multi-GPU worker pool (`multiprocessing.Pool`) to achieve maximum throughput and minimal latency. Each request is broken down into chunks and processed in parallel across all available GPUs.

2.  **🎬 Seamless Stream Switching**: Implements an advanced streaming controller that can flawlessly switch between a pre-configured idle/placeholder video and the formal task-driven video stream.

3.  **✨ Smooth Crossfade Transitions**: Eliminates jarring cuts during stream switches by performing a smooth, configurable crossfade transition. This ensures a professional and visually pleasing broadcast quality, directly blending the last idle frame with the first task frame.

4.  **🧠 Intelligent Queue & State Management**: A multi-layered, thread-safe queue system combined with a robust state machine prevents frame drops and race conditions. It gracefully handles scenarios where the producer (GPU) is much faster than the consumer (RTMP stream), ensuring every generated frame is streamed in the correct order.

5.  **🔧 Production-Ready Architecture**: Built with Flask, the server is designed for high availability and robustness. It features an asynchronous, non-blocking API, detailed logging, and a clear separation of concerns between model inference, stream management, and the API layer.

## Architecture Overview

Our real-time architecture is designed for decoupling and high performance. The system is composed of several key components that work in concert:

<p align="center">
  <img src="YOUR_ARCHITECTURE_DIAGRAM_URL_HERE" width="900">
  <br>
  <em>(Optional: You can create a diagram to illustrate this flow)</em>
</p>

1.  **Flask API Server (Main Process)**: The entry point for all requests. It's a lightweight, non-blocking server that accepts audio files, validates them, and immediately dispatches the inference job.

2.  **Persistent GPU Worker Pool (`multiprocessing.Pool`)**: On startup, we create a pool of worker processes, one for each available GPU. Each worker pre-loads the LatentSync model, eliminating model loading overhead on a per-request basis.

3.  **Task Processor (`TaskProcessor` Thread)**: For each incoming request, a dedicated `TaskProcessor` thread is created. It receives the inference results (which may be out-of-order) from the GPU pool, re-orders them, and places the final, ordered video/audio chunks into the `FORMAL_QUEUE`.

4.  **Streaming Controller (`StreamingController` Thread)**: This is the heart of the live stream. It's a master controller that runs an endless loop with the following logic:
    *   It continuously plays a looping idle video from the `IDLE_QUEUE`.
    *   It constantly probes the `FORMAL_QUEUE` for new tasks.
    *   Upon detecting a new task, it executes a **crossfade transition** from the last idle frame to the first formal frame.
    *   It then streams all chunks from the `FORMAL_QUEUE` until it's empty and the task is complete.
    *   Finally, it seamlessly switches back to playing the idle stream.

5.  **Named Pipes (FIFO) & FFmpeg**: The `StreamingController` pushes the final raw video and audio data into dedicated named pipes. A separate, persistent FFmpeg process reads from these pipes, encodes the data in real-time, and pushes it to the specified RTMP server. This decouples the Python application from the complexities of video encoding and streaming.

## Demo

Here is a demonstration of the seamless transition between the idle stream and the lipsync task generated from an audio input.

*(Embed your GIF or video demo here)*

`<img src="YOUR_PROJECT_DEMO_GIF_URL_HERE" width="800">`

## Installation and Setup

### Prerequisites
- Python 3.8+
- CUDA-enabled GPU (NVIDIA)
- FFmpeg

### Steps

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/LatentSync-Realtime.git
    cd LatentSync-Realtime
    ```

2.  **Install Dependencies:**
    Please follow the official [LatentSync dependency installation guide](https://github.com/bytedance/LatentSync#installation).
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download Models:**
    Download the **LatentSync 1.0 models** from their official Hugging Face repository:
    [huggingface.co/ByteDance/LatentSync](https://huggingface.co/ByteDance/LatentSync)

    Organize the downloaded checkpoints into the `checkpoints/` directory according to the structure specified in the official LatentSync project.

4.  **Configure Your Streams:**
    Open `infer_serv.py` and modify the following paths and URLs at the bottom of the file:
    ```python
    # In the if __name__ == "__main__": block
    CONFIG_PATH = "configs/unet/second_stage.yaml"
    CHECKPOINT_PATH = "checkpoints/latentsync_unet.pt"
    VIDEO_PATH = "assets/video/your_formal_base_video.mp4" # Base video for lipsyncing
    IDLE_VIDEO_PATH = "assets/video/your_idle_loop_video.mp4" # Looping idle video

    # In the StreamingController class
    rtmp_url = "rtmp://your-rtmp-server/live/stream_key"
    ```

5.  **Run the Server:**
    ```bash
    python infer_serv.py
    ```
    The server will start, load the models into the GPU workers, and begin streaming the idle video to your RTMP URL.

## API Usage Example

You can send a request to generate a video using any audio file. The server will perform the transition and lipsync generation in real-time.

```bash
curl -X POST \
  -F "audio_file=@assets/audio/your_audio.wav" \
  http://127.0.0.1:11001/generate_video
