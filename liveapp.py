# liveapp.py
import os
from datetime import datetime
from flask import Flask, request, jsonify
from omegaconf import OmegaConf
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from accelerate.utils import set_seed
import logging
import threading

# 导入我们重构的类
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from live_streaming_manager import LiveStreamingManager
# 假设您的 whisper 模块路径正确
from latentsync.whisper.audio2feature import Audio2Feature
from latentsync.models.unet import UNet3DConditionModel

# --- 全局变量 ---
app = Flask(__name__)
# 全局的直播管理器实例
live_manager: LiveStreamingManager = None
# 用于确保模型只加载一次的锁
model_load_lock = threading.Lock()
# 存储一些全局配置
config = None
# 视频资产路径
VIDEO_ASSET_PATH = "assets/WeChat_20250106160900.mp4" # 默认的背景视频

# --- 日志设置 ---
def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("live_service.log"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)

logger = setup_logger()

# --- 模型加载 ---
def load_models_and_init_pipeline(config_path="configs/unet/second_stage.yaml", checkpoint_path="checkpoints/latentsync_unet.pt"):
    """
    加载所有必要的模型并返回一个配置好的 LipsyncPipeline 实例。
    这个函数只应该被调用一次。
    """
    global config
    
    with model_load_lock:
        logger.info("Acquired model lock. Loading models...")
        
        # 加载配置
        config = OmegaConf.load(config_path)
        logger.info(f"Configuration loaded from: {config_path}")

        # 加载调度器
        scheduler = DDIMScheduler.from_pretrained("configs")

        # 加载 Whisper 模型
        whisper_model_path = "checkpoints/whisper/tiny.pt" # 简化
        audio_encoder = Audio2Feature(
            model_path=whisper_model_path, device="cuda", num_frames=config.data.num_frames
        )

        # 加载 VAE
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)
        vae.config.scaling_factor = 0.18215
        vae.config.shift_factor = 0

        # 加载 UNet
        unet, _ = UNet3DConditionModel.from_pretrained(
            OmegaConf.to_container(config.model),
            checkpoint_path,
            device="cuda",
        )
        unet = unet.to(dtype=torch.float16)

        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
            logger.info("xformers enabled.")

        # 创建基础 pipeline 实例
        pipeline = LipsyncPipeline(
            vae=vae,
            audio_encoder=audio_encoder,
            unet=unet,
            scheduler=scheduler,
        ).to("cuda")
        
        logger.info("All models loaded and pipeline created successfully.")
        return pipeline

# --- Flask API Endpoints ---

@app.route('/start_stream', methods=['POST'])
def start_stream_api():
    """启动直播推流服务"""
    global live_manager
    if live_manager and live_manager.is_running:
        return jsonify({"status": "error", "message": "Stream is already running."}), 400

    data = request.get_json()
    if not data or 'rtmp_url' not in data:
        return jsonify({"status": "error", "message": "rtmp_url is required."}), 400

    rtmp_url = data['rtmp_url']
    logger.info(f"Received request to start streaming to: {rtmp_url}")

    try:
        # 在第一次启动时加载模型
        if live_manager is None:
            base_pipeline = load_models_and_init_pipeline()
            live_manager = LiveStreamingManager(
                pipeline=base_pipeline,
                rtmp_url=rtmp_url,
                max_workers=2,
                video_fps=config.data.video_fps
            )
        else:
            # 如果已存在但已停止，则更新 rtmp url 并重启
            live_manager.rtmp_url = rtmp_url

        live_manager.start()
        logger.info("Live streaming manager started successfully.")
        return jsonify({"status": "success", "message": "Streaming service started."})
    except Exception as e:
        logger.error(f"Failed to start streaming service: {e}", exc_info=True)
        return jsonify({"status": "error", "message": f"Failed to start service: {e}"}), 500

@app.route('/stop_stream', methods=['POST'])
def stop_stream_api():
    """停止直播推流服务"""
    global live_manager
    if not live_manager or not live_manager.is_running:
        return jsonify({"status": "error", "message": "Stream is not running."}), 400

    logger.info("Received request to stop streaming.")
    try:
        live_manager.stop()
        logger.info("Live streaming manager stopped successfully.")
        return jsonify({"status": "success", "message": "Streaming service stopped."})
    except Exception as e:
        logger.error(f"Failed to stop streaming service: {e}", exc_info=True)
        return jsonify({"status": "error", "message": f"Failed to stop service: {e}"}), 500


@app.route('/submit_task', methods=['POST'])
def submit_task_api():
    """向正在运行的直播流提交一个新的口型同步任务"""
    global live_manager
    if not live_manager or not live_manager.is_running:
        return jsonify({"status": "error", "message": "Stream is not running. Please start the stream first."}), 400

    # 检查文件
    if 'audio_file' not in request.files:
        return jsonify({"status": "error", "message": "No audio_file provided."}), 400

    audio_file = request.files['audio_file']
    
    # 保存上传的音频文件
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{audio_file.filename}"
    audio_path = os.path.join(upload_dir, filename)
    audio_file.save(audio_path)
    logger.info(f"Audio file saved to: {audio_path}")
    
    # 提交任务给管理器
    try:
        # 使用全局配置的背景视频
        live_manager.submit_task(audio_path=audio_path, video_path=VIDEO_ASSET_PATH)
        logger.info(f"Task submitted for audio: {audio_path}")
        # 立即返回，不等待处理完成
        return jsonify({"status": "success", "message": "Task submitted successfully."})
    except Exception as e:
        logger.error(f"Failed to submit task: {e}", exc_info=True)
        return jsonify({"status": "error", "message": f"Failed to submit task: {e}"}), 500

@app.route('/stream_status', methods=['GET'])
def stream_status_api():
    """查询当前直播流的状态"""
    global live_manager
    if not live_manager:
        return jsonify({"status": "inactive", "message": "Streaming service has not been initialized."})

    if live_manager.is_running:
        return jsonify({
            "status": "active",
            "rtmp_url": live_manager.rtmp_url,
            "tasks_in_queue": live_manager.task_queue.qsize()
        })
    else:
        return jsonify({"status": "inactive", "message": "Streaming service is stopped."})

if __name__ == "__main__":
    # 注意：我们不在主线程中加载模型，而是在第一次调用 /start_stream 时延迟加载。
    # 这可以加快服务启动速度。
    logger.info("Flask service starting...")
    logger.info("Models will be loaded on the first '/start_stream' request.")
    app.run(host="0.0.0.0", port=5000)