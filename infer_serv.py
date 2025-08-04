# infer_serv.py
import os
import tempfile
import logging
import time
import atexit
import multiprocessing
import queue
import torch
import torchvision
import tqdm
import numpy as np
import cv2
import subprocess
import threading
import uuid

from datetime import datetime
from flask import Flask, request, jsonify
from omegaconf import OmegaConf
from threading import Thread, Lock


# 导入您项目中的相关模块
from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import (
    LipsyncPipeline, 
    init_worker, 
    run_inference_on_worker,
    get_audio_chunk,
)
from latentsync.whisper.audio2feature import Audio2Feature
from diffusers import AutoencoderKL, DDIMScheduler
from latentsync.utils.image_processor import ImageProcessor
from latentsync.utils.util import read_audio, read_video

# =========================================================================
# 1. 日志 和 Flask App 初始化
# =========================================================================

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(processName)s - %(message)s", # 添加进程名以便调试
        handlers=[
            logging.FileHandler("latentsync_server.log"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger()

logger = setup_logger()
app = Flask(__name__)

# 全局变量，用于持有加载好的流水线和配置
pipeline_global = None
config_global = None
video_data_global = {}
pool_global = None
stream_controller_global = None # 全局推流控制器

# 关键：为流式推流引入的全局队列
IDLE_QUEUE = queue.Queue()      # 持久化的空闲流队列
FORMAL_QUEUE = queue.Queue()    # 实时任务流队列
STREAM_LOCK = Lock()            # 确保对 ffmpeg 进程操作的原子性

# --- 新增：用于管理并发任务的全局变量 ---
ACTIVE_TASKS = {}
TASK_LOCK = Lock()

# =========================================================================
# 2. 推流控制器 (核心新增部分)
# =========================================================================
class StreamingController(Thread):
    def __init__(self, idle_queue, formal_queue, rtmp_url, video_fps, audio_sample_rate, width, height):
        super().__init__()
        self.idle_queue = idle_queue
        self.formal_queue = formal_queue
        self.rtmp_url = rtmp_url
        self.video_fps = video_fps
        self.audio_sample_rate = audio_sample_rate
        self.width = width
        self.height = height

        self.fade_frames = 5  # 过渡效果的帧数，可以调整以平衡丝滑度和延迟
        # 预先创建一个黑场帧和一个静音音频块，以提高效率
        self.black_frame = np.zeros((height, width, 3), dtype=np.uint8)
        # 计算一个标准块的音频样本数
        audio_samples_per_chunk = int(audio_sample_rate * config_global.data.num_frames / video_fps)
        self.silent_audio_chunk = np.zeros(audio_samples_per_chunk, dtype=np.int16)

        # --- 用于跟踪流状态的变量 ---
        self.is_streaming_formal = False
        
        # 命名管道的路径
        self.video_fifo_path = "/tmp/latentsync_video_fifo"
        self.audio_fifo_path = "/tmp/latentsync_audio_fifo"

        self.ffmpeg_process = None
        self.stop_event = threading.Event()
        self.daemon = True

        # --- 改造点 1: 明确队列的角色 ---
        # 写入器队列，直接面向FFmpeg管道
        self.video_write_queue = queue.Queue(maxsize=4) 
        self.audio_write_queue = queue.Queue(maxsize=4)
        # 统一的媒体缓冲队列，用于切换时清空
        self.media_write_queue = queue.Queue(maxsize=5) # 容量可以稍大

    # --- 创建过渡视频块的辅助方法 ---
    def _create_fade_chunk(self, start_frame, end_frame):
        """根据开始帧和结束帧，生成一个平滑过渡的视频/音频块。"""
        fade_video_chunk = []
        for i in range(self.fade_frames):
            # alpha 从 0 (完全是start_frame) 渐变到 1 (完全是end_frame)
            alpha = i / (self.fade_frames - 1)
            # 使用cv2.addWeighted进行线性混合
            blended_frame = cv2.addWeighted(start_frame, 1 - alpha, end_frame, alpha, 0)
            fade_video_chunk.append(blended_frame)

        # 计算过渡期间的音频样本数
        fade_audio_samples = int(self.audio_sample_rate * self.fade_frames / self.video_fps)
        # 在过渡期间使用静音
        fade_audio_chunk = np.zeros(fade_audio_samples, dtype=np.int16)

        return {"video": np.array(fade_video_chunk), "audio": fade_audio_chunk}


    def _setup_pipes(self):
        # ... (此函数保持不变) ...
        """创建命名管道"""
        logger.info("Setting up named pipes (FIFOs)...")
        for path in [self.video_fifo_path, self.audio_fifo_path]:
            if os.path.exists(path):
                os.remove(path)
            os.mkfifo(path)

    def _cleanup_pipes(self):
        # ... (此函数保持不变) ...
        """删除命名管道"""
        logger.info("Cleaning up named pipes (FIFOs)...")
        for path in [self.video_fifo_path, self.audio_fifo_path]:
            if os.path.exists(path):
                os.remove(path)

    def _video_writer_loop(self):
        # ... (此函数保持不变, 非常重要) ...
        """一个专门向视频管道写入数据的线程循环"""
        try:
            with open(self.video_fifo_path, 'wb') as pipe:
                logger.info("Video pipe opened for writing.")
                while not self.stop_event.is_set():
                    try:
                        chunk = self.video_write_queue.get(timeout=1)
                    except queue.Empty:
                        continue
                    if chunk is None: # 停止信号
                        break
                    try:
                        for frame in chunk:
                            pipe.write(frame.tobytes())
                        pipe.flush() # 确保数据被写入
                    except IOError as e:
                        logger.error(f"Video pipe write error: {e}")
                        break
        except Exception as e:
            logger.error(f"Video writer thread crashed: {e}")
        logger.info("Video writer thread finished.")

    def _audio_writer_loop(self):
        # ... (此函数保持不变, 非常重要) ...
        """一个专门向音频管道写入数据的线程循环"""
        try:
            with open(self.audio_fifo_path, 'wb') as pipe:
                logger.info("Audio pipe opened for writing.")
                while not self.stop_event.is_set():
                    try:
                        chunk = self.audio_write_queue.get(timeout=1)
                    except queue.Empty:
                        continue
                    if chunk is None: # 停止信号
                        break
                    try:
                        pipe.write(chunk.tobytes())
                        pipe.flush()
                    except IOError as e:
                        logger.error(f"Audio pipe write error: {e}")
                        break
        except Exception as e:
            logger.error(f"Audio writer thread crashed: {e}")
        logger.info("Audio writer thread finished.")

    # --- 改造点 2: 新增一个媒体分发器线程 ---
    def _media_dispatcher_loop(self):
        """
        核心分发器：从统一的media_write_queue取数据，
        然后拆分并放入独立的音视频写入队列。
        """
        logger.info("Media dispatcher thread started.")
        while not self.stop_event.is_set():
            try:
                item = self.media_write_queue.get(timeout=1)
            except queue.Empty:
                continue

            if item is None: # 停止信号
                break
            
            # 拆分并分发到各自的写入队列
            self.video_write_queue.put(item['video'])
            self.audio_write_queue.put(item['audio'])
        logger.info("Media dispatcher thread finished.")

    # --- 改造点 3: 新增一个清空队列的函数 ---
    def _clear_buffer_queue(self):
        """清空媒体缓冲队列，用于流切换"""
        logger.info(f"Clearing media buffer queue (size: {self.media_write_queue.qsize()})...")
        # 这是清空队列的标准线程安全方法
        with self.media_write_queue.mutex:
            self.media_write_queue.queue.clear()
        logger.info("Media buffer queue cleared.")

    def run(self):
        """
        主循环（V3 - 带排空逻辑优化）：
        确保在切换回空闲模式前，完全排空正式任务队列。
        """
        self._setup_pipes()
        self.start_ffmpeg_process()

        # 启动所有工作线程
        video_writer = Thread(target=self._video_writer_loop, daemon=True)
        audio_writer = Thread(target=self._audio_writer_loop, daemon=True)
        media_dispatcher = Thread(target=self._media_dispatcher_loop, daemon=True)
        video_writer.start()
        audio_writer.start()
        media_dispatcher.start()

        # 关键：我们需要一个变量来持有即将被替换的空闲块
        last_idle_item = None

        try:
            while not self.stop_event.is_set():
                item_to_stream = None

                # -----------------------------------------------------------
                # 核心逻辑：检查 formal_queue 是否有新任务，这是最高优先级
                # -----------------------------------------------------------
                try:
                    # 使用 get_nowait() 探测新任务
                    first_formal_item = self.formal_queue.get_nowait()
                    
                    # 如果我们能取到数据，并且当前正在播放空闲流，说明切换点到了！
                    if not self.is_streaming_formal:
                        logger.info(">>>>>> Event: A formal task is starting. Initiating crossfade. <<<<<<")
                        
                        # 清空媒体缓冲，为过渡做准备，保证低延迟
                        self._clear_buffer_queue()
                        
                        if last_idle_item:
                            # 1. 提取关键帧
                            last_idle_frame = last_idle_item['video'][-1]
                            first_formal_frame = first_formal_item['video'][0]

                            # 2. 创建 Crossfade 过渡块
                            crossfade_chunk = self._create_fade_chunk(last_idle_frame, first_formal_frame)
                            self.media_write_queue.put(crossfade_chunk)
                        
                        self.is_streaming_formal = True

                    # 无论是否是切换点，都需要处理这个刚取出的正式任务块
                    item_to_stream = first_formal_item

                except queue.Empty:
                    # FORMAL_QUEUE 为空，检查宏观任务状态
                    with TASK_LOCK:
                        is_task_active = bool(ACTIVE_TASKS)

                    # 如果任务结束且我们仍在正式流模式，切换回空闲
                    if not is_task_active and self.is_streaming_formal:
                        logger.info(">>>>>> Event: All formal tasks finished and queue drained. Switching back to IDLE. <<<<<<")
                        # 这里我们选择直接切换，也可以设计从正式到空闲的淡出
                        self.is_streaming_formal = False

                    # -----------------------------------------------------------
                    # 如果当前应该播放空闲流
                    # -----------------------------------------------------------
                    if not self.is_streaming_formal:
                        try:
                            # 从空闲队列获取数据，并循环使用
                            item_to_stream = self.idle_queue.get(timeout=0.1)
                            self.idle_queue.put(item_to_stream)
                            
                            # 关键：持续更新“最后一个空闲块”，为随时可能到来的切换做准备
                            last_idle_item = item_to_stream
                            
                        except queue.Empty:
                            continue
                    
                    # 如果任务仍在运行但队列暂时为空，则等待
                    elif is_task_active:
                        time.sleep(0.01)
                        continue

                # -----------------------------------------------------------
                # 将获取到的数据块送入统一缓冲队列
                # -----------------------------------------------------------
                if item_to_stream:
                    # 预处理数据格式 (不变)
                    video_chunk = item_to_stream['video']
                    audio_chunk = item_to_stream['audio']
                    if video_chunk.dtype != np.uint8:
                         video_chunk = (video_chunk * 255).astype(np.uint8) if video_chunk.max() <= 1.0 else video_chunk.astype(np.uint8)
                    if audio_chunk.dtype != np.int16:
                        audio_chunk = (audio_chunk * 32767).astype(np.int16)
                    
                    self.media_write_queue.put({"video": video_chunk, "audio": audio_chunk})

        finally:
            # 清理逻辑保持不变
            logger.info("Streaming controller main loop is shutting down.")
            self.media_write_queue.put(None)
            media_dispatcher.join(timeout=2)
            self.video_write_queue.put(None)
            self.audio_write_queue.put(None)
            video_writer.join(timeout=2)
            audio_writer.join(timeout=2)
            if self.ffmpeg_process and self.ffmpeg_process.poll() is None:
                self.ffmpeg_process.terminate()
            self._cleanup_pipes()

    def start_ffmpeg_process(self):
        # ... (此函数保持不变) ...
        logger.info(f"Starting FFmpeg to read from named pipes...")
        command = [
            'ffmpeg', '-y', '-re',
            '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24',
            '-s', f'{self.width}x{self.height}', '-r', str(self.video_fps),
            '-i', self.video_fifo_path,
            '-f', 's16le', '-ar', str(self.audio_sample_rate), '-ac', '1',
            '-i', self.audio_fifo_path,
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'veryfast', '-crf', '22',
            '-c:a', 'aac', '-b:a', '128k',
            '-f', 'flv', self.rtmp_url,
        ]
        self.ffmpeg_process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    def stop(self):
        # ... (此函数保持不变) ...
        """外部调用的停止方法"""
        logger.info("Received stop signal for StreamingController.")
        self.stop_event.set()

# =========================================================================
# 3. [新] 结果处理器 TaskProcessor
# =========================================================================
class TaskProcessor(Thread):
    def __init__(self, task_id, results_iterator, audio_samples, num_frames, video_fps, audio_sample_rate, total_chunks):
        super().__init__()
        self.task_id = task_id
        self.results_iterator = results_iterator
        self.audio_samples = audio_samples
        self.num_frames = num_frames
        self.video_fps = video_fps
        self.audio_sample_rate = audio_sample_rate
        self.total_chunks = total_chunks

        self.reorder_buffer = {}
        self.expected_chunk_index = 0
        self.daemon = True

    def run(self):
        """处理乱序结果，并按顺序放入FORMAL_QUEUE"""
        logger.info(f"[Task-{self.task_id}] Started processing. Expecting {self.total_chunks} chunks.")
        
        try:
            for chunk_index, result_segment in self.results_iterator:
                if result_segment is None:
                    logger.warning(f"[Task-{self.task_id}] Received a failed chunk (None) for index {chunk_index}. Skipping.")
                    # 我们可以选择如何处理失败的块，这里选择跳过
                    self.total_chunks -= 1
                    continue

                # 将收到的结果放入缓冲区
                self.reorder_buffer[chunk_index] = result_segment
                
                # [核心逻辑] 检查是否可以释放缓冲区中的连续块
                while self.expected_chunk_index in self.reorder_buffer:
                    # 从缓冲区取出有序的块
                    video_chunk = self.reorder_buffer.pop(self.expected_chunk_index)
                    
                    # 提取对应的音频块
                    start_frame_index = self.expected_chunk_index * self.num_frames
                    audio_chunk = get_audio_chunk(
                        self.audio_samples, start_frame_index, self.num_frames, self.video_fps, self.audio_sample_rate
                    )
                    
                    # 将准备好的音视频块放入全局推流队列
                    FORMAL_QUEUE.put({"video": video_chunk, "audio": audio_chunk})
                    logger.info(f"[Task-{self.task_id}] Chunk {self.expected_chunk_index} sent to stream queue.")
                    
                    # 期待下一个块
                    self.expected_chunk_index += 1
            
            if self.expected_chunk_index == self.total_chunks:
                logger.info(f"[Task-{self.task_id}] All chunks processed and queued successfully.")
            else:
                logger.warning(f"[Task-{self.task_id}] Task finished, but some chunks are missing. "
                               f"Processed {self.expected_chunk_index}/{self.total_chunks}.")

        except Exception as e:
            logger.error(f"[Task-{self.task_id}] An error occurred during result processing: {e}", exc_info=True)
        finally:
            # 任务结束，从全局任务列表中移除自己
            with TASK_LOCK:
                if self.task_id in ACTIVE_TASKS:
                    del ACTIVE_TASKS[self.task_id]
            logger.info(f"[Task-{self.task_id}] Processor thread finished and cleaned up.")

# =========================================================================
# 3. 模型加载 和 空闲视频预处理 (重构)
# =========================================================================
def load_models_and_prepare_idle_stream(config_path, checkpoint_path, video_path, idle_video_path):
    global pipeline_global, config_global, video_data_global, pool_global, stream_controller_global

    # --- 模型加载和主视频预处理 (不变) ---
    logger.info("Loading models and creating worker pool...")
    load_models_and_pool(config_path, checkpoint_path, video_path)
    logger.info("Models and worker pool loaded successfully.")

    # --- [关键修正] 从已处理的主视频中获取最终的输出分辨率 ---
    try:
        first_frame = video_data_global["original_video_frames"][0]
        # shape 通常是 (height, width, channels)
        target_height, target_width, _ = first_frame.shape
        logger.info(f"Detected target stream resolution from source video: {target_width}x{target_height}")
    except (IndexError, KeyError):
        logger.error("Could not determine target resolution from preprocessed video data. Exiting.")
        raise ValueError("Failed to get video dimensions.")

    # --- 加载并正确地预处理空闲视频 ---
    logger.info(f"Loading and preprocessing idle video from: {idle_video_path}")
    try:
        idle_video_frames_raw = read_video(idle_video_path, use_decord=True)
        idle_audio_samples = read_audio(idle_video_path)

        if idle_video_frames_raw.size == 0:
             raise ValueError(f"Could not read any frames from idle video: {idle_video_path}")

        # 其他参数
        num_frames_per_chunk = config_global.data.num_frames
        video_fps = 25
        audio_sample_rate = 16000

        logger.info(f"Processing {len(idle_video_frames_raw)} idle frames to match target resolution {target_width}x{target_height}")

        processed_idle_frames = []
        for frame in idle_video_frames_raw:
            # 确保数据类型正确
            if frame.dtype != np.uint8:
                 frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)

            # [关键修正] 将空闲视频帧调整为最终的输出分辨率
            resized_frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
            
            # 转换为 FFmpeg 期望的 RGB 格式
            # rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            processed_idle_frames.append(resized_frame)

        # 创建队列
        for i in range(0, len(processed_idle_frames) - num_frames_per_chunk + 1, num_frames_per_chunk):
            video_chunk = np.array(processed_idle_frames[i : i + num_frames_per_chunk])
            if video_chunk.shape[0] < num_frames_per_chunk: continue

            audio_chunk = get_audio_chunk(idle_audio_samples, i, num_frames_per_chunk, video_fps, audio_sample_rate)
            if audio_chunk.dtype != np.int16:
                audio_chunk = (audio_chunk * 32767).astype(np.int16)

            IDLE_QUEUE.put({"video": video_chunk, "audio": audio_chunk})

        logger.info(f"Idle stream queue populated with {IDLE_QUEUE.qsize()} chunks.")
        if IDLE_QUEUE.empty():
            raise ValueError("Failed to process idle video into chunks.")

    except Exception as e:
        logger.error(f"Could not process idle video: {e}", exc_info=True)
        raise

    # --- [关键修正] 使用正确的高分辨率启动推流控制器 ---
    logger.info("Starting the Streaming Controller...")
    rtmp_url = "rtmp://117.74.66.188:1935/dash/latentsync_video"

    stream_controller_global = StreamingController(
        idle_queue=IDLE_QUEUE,
        formal_queue=FORMAL_QUEUE,
        rtmp_url=rtmp_url,
        video_fps=video_fps,
        audio_sample_rate=audio_sample_rate,
        width=target_width,      # <-- 使用检测到的宽度
        height=target_height,    # <-- 使用检测到的高度
    )
    stream_controller_global.start()
    logger.info("Streaming Controller started, pushing idle stream.")

# =========================================================================
# 2. 模型加载逻辑
# =========================================================================
def load_models_and_pool(config_path, checkpoint_path, video_path):
    global pipeline_global, config_global, video_data_global, pool_global

    logger.info("Loading configuration file...")
    config_global = OmegaConf.load(config_path)
    logger.info(f"Configuration loaded from: {config_path}")

    # 使用 config 来获取参数
    cross_attention_dim = config_global.model.cross_attention_dim
    resolution = config_global.data.resolution
    
    # 决定 Whisper 模型路径
    if cross_attention_dim == 768:
        WHISPER_MODEL_PATH = "checkpoints/whisper/small.pt"
    elif cross_attention_dim == 384:
        WHISPER_MODEL_PATH = "checkpoints/whisper/tiny.pt"
    else:
        raise NotImplementedError("cross_attention_dim must be 768 or 384")

    # 定义其他组件路径
    VAE_PATH = "stabilityai/sd-vae-ft-mse"
    SCHEDULER_PATH = "configs" # 假设调度器配置在此

    logger.info("Loading model components...")
    scheduler = DDIMScheduler.from_pretrained(SCHEDULER_PATH)
    audio_encoder = Audio2Feature(model_path=WHISPER_MODEL_PATH, device="cuda")
    vae = AutoencoderKL.from_pretrained(VAE_PATH, torch_dtype=torch.float16)
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0
    
    unet_model, _ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(config_global.model),
        checkpoint_path,
        device="cuda"
    )
    
    # 现在 unet_model 是一个纯粹的模型对象，可以安全地调用 .to()
    unet = unet_model.to(dtype=torch.float16)
    logger.info("All components loaded successfully.")

    # 创建并初始化流水线，传递所有必要的路径信息
    logger.info("Creating and initializing the main pipeline...")
    pipeline_global = LipsyncPipeline(
        vae=vae,
        audio_encoder=audio_encoder,
        unet=unet,
        scheduler=scheduler,
        config_path=config_path,
        vae_path=VAE_PATH,
        audio_encoder_path=WHISPER_MODEL_PATH,
        unet_path=checkpoint_path,
        scheduler_path=SCHEDULER_PATH,
        video_path=video_path,
    )
    
    main_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pipeline_global.to(main_device)
    logger.info(f"Pipeline initialized on {main_device}.")

    logger.info(f"Preprocessing video file: {video_path}...")
    resolution = config_global.data.resolution
    # 创建一个 ImageProcessor 实例专门用于预处理
    image_processor = ImageProcessor(resolution, mask="fix_mask", device=main_device)
    
    # 调用静态方法进行预处理
    faces, original_frames, boxes, affine_matrices = LipsyncPipeline.affine_transform_video(
        video_path, image_processor
    )
    
    # 将结果存储在全局变量中
    video_data_global = {
        "video_frames": faces,
        "original_video_frames": original_frames,
        "boxes": boxes,
        "affine_matrices": affine_matrices
    }
    logger.info(f"Video preprocessed successfully. Found {len(faces)} frames/faces.")

    # =========================================================================
    # --- C. 关键优化：创建并初始化常驻的工作进程池 ---
    # =========================================================================
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    max_workers = torch.cuda.device_count()
    if max_workers == 0:
        raise RuntimeError("No GPUs found!")

    logger.info(f"Creating and initializing a persistent pool of {max_workers} GPU workers...")
    
    # 准备传递给每个工作进程的初始化参数
    common_inference_params = {
        "height": config_global.data.resolution,
        "width": config_global.data.resolution,
        "num_frames": config_global.data.num_frames,
        "weight_dtype": torch.float16, # 假设是 float16
        "guidance_scale": 1.0, # 提供一个默认值
        "do_classifier_free_guidance": 1.0 > 1.0,
        "extra_step_kwargs": pipeline_global.prepare_extra_step_kwargs(None, 0.0),
        "num_inference_steps": config_global.run.inference_steps,
    }
    
    pool_global = multiprocessing.Pool(
        processes=max_workers,
        initializer=init_worker,
        initargs=(pipeline_global.model_init_args, common_inference_params)
    )
    
    logger.info("GPU worker pool has been successfully initialized and models are loaded.")

# =========================================================================
# 5. Flask API 端点 (改造为任务分发器)
# =========================================================================
@app.route("/generate_video", methods=["POST"])
def generate_video_api():
    """
    接收请求，创建并启动一个后台任务处理器，然后立即返回。
    """
    if pool_global is None or not stream_controller_global:
        return jsonify({"error": "Server is initializing, please wait."}), 503
    if "audio_file" not in request.files:
        return jsonify({"error": "No audio file provided."}), 400
    
    # [可选] 限制并发任务数量
    with TASK_LOCK:
        if len(ACTIVE_TASKS) >= pool_global._processes: # 例如，不允许任务数超过GPU数量
             return jsonify({"error": "Server is busy with other tasks, please try again later."}), 503

    audio_file = request.files["audio_file"]
    audio_path = ""
    try:
        # 1. 准备任务 (与之前类似，但不再等待结果)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            audio_file.save(tmp.name)
            audio_path = tmp.name
        
        logger.info(f"Received new request for audio: {audio_file.filename}. Preparing tasks...")
        num_frames = config_global.data.num_frames
        video_fps = 25
        audio_sample_rate = 16000

        audio_samples = read_audio(audio_path)
        whisper_feature = pipeline_global.audio_encoder.audio2feat(audio_path)
        whisper_chunks = pipeline_global.audio_encoder.feature2chunks(whisper_feature, fps=video_fps)
        num_inferences = min(len(video_data_global["video_frames"]), len(whisper_chunks)) // num_frames
        
        task_list = []
        guidance_scale = float(request.form.get("guidance_scale", 1.0))
        do_classifier_free_guidance = guidance_scale > 1.0

        for i in range(num_inferences):
            start_idx, end_idx = i * num_frames, (i + 1) * num_frames
            audio_embeds_chunk = torch.stack(whisper_chunks[start_idx:end_idx])
            if do_classifier_free_guidance:
                audio_embeds_chunk = torch.cat([torch.zeros_like(audio_embeds_chunk), audio_embeds_chunk])
            
            task_list.append({
                "chunk_index": i,
                "data_chunk": {
                    "audio_embeds": audio_embeds_chunk.cpu(),
                    "inference_video_frames": video_data_global["video_frames"][start_idx:end_idx],
                    "original_frames_chunk": video_data_global["original_video_frames"][start_idx:end_idx],
                    "boxes_chunk": video_data_global["boxes"][start_idx:end_idx],
                    "affine_matrices_chunk": video_data_global["affine_matrices"][start_idx:end_idx],
                }
            })
        
        if not task_list:
            return jsonify({"error": "No tasks could be created from the provided audio/video."}), 400

        # 2. 获取乱序的结果迭代器
        results_iterator = pool_global.imap_unordered(run_inference_on_worker, task_list)
        
        # 3. 创建并启动后台任务处理器
        task_id = str(uuid.uuid4())
        processor = TaskProcessor(
            task_id=task_id,
            results_iterator=results_iterator,
            audio_samples=audio_samples,
            num_frames=num_frames,
            video_fps=video_fps,
            audio_sample_rate=audio_sample_rate,
            total_chunks=len(task_list)
        )
        
        # 注册并启动任务
        with TASK_LOCK:
            ACTIVE_TASKS[task_id] = processor
        processor.start()
        
        logger.info(f"Task {task_id} has been started in the background.")
        
        # 4. 立即返回响应
        return jsonify({"status": "processing_started", "task_id": task_id})

    except Exception as e:
        logger.error(f"Critical error during task submission: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
    finally:
        # 注意：临时音频文件现在需要由TaskProcessor处理完后删除，或者使用其他机制
        # 为了简单起见，我们暂时保留它，但在生产中需要一个清理策略
        pass # if os.path.exists(audio_path): os.remove(audio_path)

# =========================================================================
# 5. 清理 与 主程序入口 (重构)
# =========================================================================
def cleanup():
    """在服务器退出时，关闭所有资源。"""
    global pool_global, stream_controller_global
    logger.info("Initiating server cleanup...")
    if stream_controller_global:
        logger.info("Stopping the Streaming Controller...")
        stream_controller_global.stop()
        stream_controller_global.join()
        logger.info("Streaming Controller stopped.")
    if pool_global:
        logger.info("Closing the GPU worker pool...")
        pool_global.close()
        pool_global.join()
        logger.info("GPU worker pool closed.")

# =========================================================================
# 4. 主程序入口
# =========================================================================
if __name__ == "__main__":
    # --- 关键：必须在任何CUDA操作之前设置多进程启动方法 ---
    torch.multiprocessing.set_start_method("spawn", force=True)
    
    logger.info("Starting Flask service...")

    # 定义所有需要的路径
    CONFIG_PATH = "configs/unet/second_stage.yaml"
    CHECKPOINT_PATH = "checkpoints/latentsync_unet.pt"
    VIDEO_PATH = "assets/video/10s_girl_0729_00_formal.mp4"
    IDLE_VIDEO_PATH = "assets/video/18s_GirlMuseum_0730.mp4"

    # 启动时加载模型并创建常驻进程池
    load_models_and_prepare_idle_stream(
        config_path=CONFIG_PATH, 
        checkpoint_path=CHECKPOINT_PATH, 
        video_path=VIDEO_PATH,
        idle_video_path=IDLE_VIDEO_PATH
    )
    # 注册退出时要执行的清理函数
    # atexit.register(cleanup)

    # 启动 Flask 服务
    logger.info("Flask service started. Listening on http://0.0.0.0:11001")
    app.run(host="0.0.0.0", port=11001, debug=False)