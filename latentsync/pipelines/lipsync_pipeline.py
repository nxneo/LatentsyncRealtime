# lipsync_pipeline.py
import inspect
import os
import shutil
from typing import Callable, List, Optional, Union
import subprocess
import time
import threading
import concurrent.futures
import multiprocessing

import numpy as np
import torch
import torchvision
import cv2
import tqdm
import soundfile as sf

from omegaconf import OmegaConf
from diffusers.models import AutoencoderKL
from ..models.unet import UNet3DConditionModel
from ..whisper.audio2feature import Audio2Feature

from diffusers.utils import is_accelerate_available
from packaging import version
from diffusers.configuration_utils import FrozenDict
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
from einops import rearrange

from ..utils.image_processor import ImageProcessor
from ..utils.util import read_video, read_audio, write_video

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def run_inference_on_gpu(worker_args: dict):
    """
    在一个独立的进程中，为单个视频块在指定的GPU上运行推理。
    """
    chunk_index = worker_args["chunk_index"]
    gpu_id = worker_args["gpu_id"]
    device = f"cuda:{gpu_id}"
    
    model_init_args = worker_args["model_init_args"]
    inference_params = worker_args["inference_params"]
    data_chunk = worker_args["data_chunk"]

    try:
        print(f"[进程 {os.getpid()}] [GPU {gpu_id}] 正在为数据块 {chunk_index} 初始化模型...")
        
        config = OmegaConf.load(model_init_args["config_path"])
        
        vae = AutoencoderKL.from_pretrained(model_init_args["vae_path"], torch_dtype=torch.float16)
        vae.config.scaling_factor = 0.18215
        vae.config.shift_factor = 0
        print(f"[进程 {os.getpid()}] [GPU {gpu_id}] VAE config patched.")

        audio_encoder = Audio2Feature(model_path=model_init_args["audio_encoder_path"])
        unet_model, _ = UNet3DConditionModel.from_pretrained(
            OmegaConf.to_container(config.model),
            model_init_args["unet_path"],
            device=device
        )
        unet = unet_model.to(dtype=inference_params["weight_dtype"])
        
        scheduler_class = model_init_args["scheduler_class"]
        scheduler = scheduler_class.from_pretrained(model_init_args["scheduler_path"])

        # =========================================================================
        # --- 这是之前出错的地方，现在已被完全修正 ---
        # 我们必须提供 __init__ 方法所需的所有必需参数。
        # =========================================================================
        pipeline = LipsyncPipeline(
            vae=vae,
            audio_encoder=audio_encoder,
            unet=unet,
            scheduler=scheduler,
            video_path=model_init_args["video_path"],
            config_path=model_init_args["config_path"],
            vae_path=model_init_args["vae_path"],
            audio_encoder_path=model_init_args["audio_encoder_path"],
            unet_path=model_init_args["unet_path"],
            scheduler_path=model_init_args["scheduler_path"],
        )
        pipeline.to(device)
        pipeline.unet.eval()
        
        print(f"[进程 {os.getpid()}] [GPU {gpu_id}] 模型初始化完成。开始处理数据块 {chunk_index}...")

        with torch.inference_mode():
            current_latents = pipeline.prepare_latents(
                batch_size=1, num_frames=inference_params["num_frames"],
                num_channels_latents=pipeline.vae.config.latent_channels,
                height=inference_params["height"], width=inference_params["width"],
                dtype=inference_params["weight_dtype"], device=device, generator=None
            )
            
            if data_chunk["audio_embeds"] is not None:
                audio_embeds = data_chunk["audio_embeds"].to(device, dtype=inference_params["weight_dtype"])
            else:
                audio_embeds = None

            restored_segment = pipeline._process_chunk(
                inference_video_frames=data_chunk["inference_video_frames"],
                audio_embeds=audio_embeds,
                latents=current_latents,
                original_frames_chunk=data_chunk["original_frames_chunk"],
                boxes_chunk=data_chunk["boxes_chunk"],
                affine_matrices_chunk=data_chunk["affine_matrices_chunk"],
                # timesteps=inference_params["timesteps"].to(device),
                do_classifier_free_guidance=inference_params["do_classifier_free_guidance"],
                guidance_scale=inference_params["guidance_scale"],
                extra_step_kwargs=inference_params["extra_step_kwargs"],
                height=inference_params["height"],
                width=inference_params["width"],
                weight_dtype=inference_params["weight_dtype"],
                generator=None,
                callback=None,
                callback_steps=1,
                num_inference_steps=inference_params["num_inference_steps"]
            )

        torch.cuda.empty_cache()
        
        print(f"[进程 {os.getpid()}] [GPU {gpu_id}] 数据块 {chunk_index} 处理完成。")
        
        return chunk_index, restored_segment

    except Exception as e:
        print(f"!!! [进程 {os.getpid()}] [GPU {gpu_id}] 在处理数据块 {chunk_index} 时发生严重错误: {e}")
        import traceback
        traceback.print_exc()
        return chunk_index, None
# =========================================================================
# 1. 初始化函数：在每个工作进程启动时，只执行一次
# =========================================================================
def init_worker(model_init_args, inference_params_base):
    """
    这个函数由 multiprocessing.Pool 在每个工作进程启动时调用一次。
    它负责加载模型，并将模型实例存储在工作进程的全局变量中。
    """
    global worker_pipeline, worker_inference_params
    
    gpu_id = multiprocessing.current_process()._identity[0] - 1
    device = f"cuda:{gpu_id}"
    
    print(f"[进程 {os.getpid()}] [GPU {gpu_id}] 正在初始化... (模型只加载一次)")
    
    config = OmegaConf.load(model_init_args["config_path"])
    vae = AutoencoderKL.from_pretrained(model_init_args["vae_path"], torch_dtype=torch.float16)
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0
    print(f"[进程 {os.getpid()}] [GPU {gpu_id}] VAE config patched.")

    audio_encoder = Audio2Feature(model_path=model_init_args["audio_encoder_path"])
    unet_model, _ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(config.model),
        model_init_args["unet_path"], device=device
    )
    # 注意：UNet 在这里就转换为 float16
    unet = unet_model.to(dtype=inference_params_base["weight_dtype"])
    
    scheduler_class = model_init_args["scheduler_class"]
    scheduler = scheduler_class.from_pretrained(model_init_args["scheduler_path"])

    worker_pipeline = LipsyncPipeline(
        vae=vae, audio_encoder=audio_encoder, unet=unet, scheduler=scheduler,
        video_path=model_init_args["video_path"],
        config_path=model_init_args["config_path"], vae_path=model_init_args["vae_path"],
        audio_encoder_path=model_init_args["audio_encoder_path"],
        unet_path=model_init_args["unet_path"],
        scheduler_path=model_init_args["scheduler_path"],
    )
    worker_pipeline.to(device)
    worker_pipeline.unet.eval()
    
    # 存储基础的推理参数
    worker_inference_params = inference_params_base
    
    print(f"[进程 {os.getpid()}] [GPU {gpu_id}] 初始化完成。")

# =========================================================================
# 2. 工作函数：现在变得非常轻量，只负责推理
# =========================================================================
def run_inference_on_worker(data_chunk_with_index: dict):
    """
    这个函数由工作进程为每个任务调用。
    它直接使用已经加载在全局变量中的模型进行推理。
    """
    global worker_pipeline, worker_inference_params
    
    chunk_index = data_chunk_with_index["chunk_index"]
    data_chunk = data_chunk_with_index["data_chunk"]
    
    gpu_id = multiprocessing.current_process()._identity[0] - 1
    device = f"cuda:{gpu_id}"

    try:
        with torch.inference_mode():
            current_latents = worker_pipeline.prepare_latents(
                batch_size=1, num_frames=worker_inference_params["num_frames"],
                num_channels_latents=worker_pipeline.vae.config.latent_channels,
                height=worker_inference_params["height"], width=worker_inference_params["width"],
                dtype=worker_inference_params["weight_dtype"], device=device, generator=None
            )
            
            if data_chunk["audio_embeds"] is not None:
                audio_embeds = data_chunk["audio_embeds"].to(device, dtype=worker_inference_params["weight_dtype"])
            else:
                audio_embeds = None

            restored_segment = worker_pipeline._process_chunk(
                inference_video_frames=data_chunk["inference_video_frames"],
                audio_embeds=audio_embeds,
                latents=current_latents,
                original_frames_chunk=data_chunk["original_frames_chunk"],
                boxes_chunk=data_chunk["boxes_chunk"],
                affine_matrices_chunk=data_chunk["affine_matrices_chunk"],
                # 传递所有共享参数
                do_classifier_free_guidance=worker_inference_params["do_classifier_free_guidance"],
                guidance_scale=worker_inference_params["guidance_scale"],
                extra_step_kwargs=worker_inference_params["extra_step_kwargs"],
                height=worker_inference_params["height"],
                width=worker_inference_params["width"],
                weight_dtype=worker_inference_params["weight_dtype"],
                generator=None,
                callback=None,
                callback_steps=1,
                num_inference_steps=worker_inference_params["num_inference_steps"],
                progress_bar_pos=gpu_id,             # 将GPU ID作为进度条的行号
                chunk_index_for_desc=chunk_index     # 将块索引用于描述
            )

        return chunk_index, restored_segment

    except Exception as e:
        print(f"!!! [进程 {os.getpid()}] [GPU {gpu_id}] 在处理数据块 {chunk_index} 时发生严重错误: {e}")
        import traceback
        traceback.print_exc()
        return chunk_index, None

class LipsyncPipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae: AutoencoderKL,
        audio_encoder: Audio2Feature,
        unet: UNet3DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        video_path: str,
        config_path: str, 
        vae_path: str,
        audio_encoder_path: str,
        unet_path: str,
        scheduler_path: str,
    ):
        super().__init__()

        # 保存用于多进程初始化的路径和类信息 ---
        # 子进程将使用这些信息来创建自己的模型副本
        self.model_init_args = {
            "config_path": config_path,
            "vae_path": vae_path,
            "unet_path": unet_path,
            "scheduler_path": scheduler_path,
            "audio_encoder_path": audio_encoder_path,
            "scheduler_class": scheduler.__class__,
            "video_path": video_path,
        }

        # self.image_processor = ImageProcessor(height, mask=mask, device="cuda")

        # 加载并预处理视频
        # self.video_path = video_path
        # self.video_frames, self.original_video_frames, self.boxes, self.affine_matrices = self.affine_transform_video(video_path)
        # logger.info(f"Preprocessed {len(self.video_frames)} frames from video: {video_path}")

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            audio_encoder=audio_encoder,
            unet=unet,
            scheduler=scheduler,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        self.set_progress_bar_config(desc="Steps")

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def decode_latents(self, latents):
        latents = latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        image = self.vae.decode(latents).sample
        return image

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, height, width, callback_steps):
        assert height == width, "Height and width must be equal"

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def prepare_latents(self, batch_size, num_frames, num_channels_latents, height, width, dtype, device, generator):
        if isinstance(device, str):
            device = torch.device(device)
        shape = (
            batch_size,
            num_channels_latents,
            1,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        rand_device = "cpu" if device.type == "mps" else device
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        latents = latents.repeat(1, 1, num_frames, 1, 1)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_mask_latents(
        self, mask, masked_image, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )
        masked_image = masked_image.to(device=device, dtype=dtype)

        # encode the mask image into latents space so we can concatenate it to the latents
        masked_image_latents = self.vae.encode(masked_image).latent_dist.sample(generator=generator)
        masked_image_latents = (masked_image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        # aligning device to prevent device errors when concating it with the latent model input
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        mask = mask.to(device=device, dtype=dtype)

        # assume batch size = 1
        mask = rearrange(mask, "f c h w -> 1 c f h w")
        masked_image_latents = rearrange(masked_image_latents, "f c h w -> 1 c f h w")

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = (
            torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
        )
        return mask, masked_image_latents

    def prepare_image_latents(self, images, device, dtype, generator, do_classifier_free_guidance):
        images = images.to(device=device, dtype=dtype)
        
        image_latents = self.vae.encode(images).latent_dist.sample(generator=generator)
        # 使用正确的缩放和偏移
        image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        
        # 转换为目标 dtype
        image_latents = image_latents.to(device=device, dtype=dtype)

        image_latents = rearrange(image_latents, "f c h w -> 1 c f h w")
        if do_classifier_free_guidance:
            image_latents = torch.cat([image_latents] * 2)

        return image_latents

    def set_progress_bar_config(self, **kwargs):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        self._progress_bar_config.update(kwargs)

    @staticmethod
    def paste_surrounding_pixels_back(decoded_latents, pixel_values, masks, device, weight_dtype):
        # Paste the surrounding pixels back, because we only want to change the mouth region
        pixel_values = pixel_values.to(device=device, dtype=weight_dtype)
        masks = masks.to(device=device, dtype=weight_dtype)
        combined_pixel_values = decoded_latents * masks + pixel_values * (1 - masks)
        return combined_pixel_values

    @staticmethod
    def pixel_values_to_images(pixel_values: torch.Tensor):
        pixel_values = rearrange(pixel_values, "f c h w -> f h w c")
        pixel_values = (pixel_values / 2 + 0.5).clamp(0, 1)
        images = (pixel_values * 255).to(torch.uint8)
        images = images.cpu().numpy()
        return images

    @staticmethod
    def affine_transform_video(video_path: str, image_processor: ImageProcessor):
        video_frames = read_video(video_path, use_decord=False)
        if video_frames.size == 0: 
            logger.error(f"Could not read any frames from video: {video_path}. Please check the path and file integrity.")
            raise ValueError(f"Video file at {video_path} could not be opened or is empty.")

        faces = []
        boxes = []
        affine_matrices = []
        print(f"Affine transforming {len(video_frames)} faces...")
        for frame in tqdm.tqdm(video_frames):
            face, box, affine_matrix = image_processor.affine_transform(frame)
            faces.append(face)
            boxes.append(box)
            affine_matrices.append(affine_matrix)
        
        if not faces: # 再次检查，如果视频中未检测到人脸
            raise ValueError(f"No faces were detected in the video at {video_path}.")

        faces = torch.stack(faces)
        return faces, video_frames, boxes, affine_matrices

    def restore_video(self, faces, video_frames, boxes, affine_matrices, image_processor):
        video_frames = video_frames[: faces.shape[0]]
        out_frames = []
        for index, face in enumerate(faces):
            x1, y1, x2, y2 = boxes[index]
            height = int(y2 - y1)
            width = int(x2 - x1)
            face = torchvision.transforms.functional.resize(face, size=(height, width), antialias=True)
            face = rearrange(face, "c h w -> h w c")
            face = (face / 2 + 0.5).clamp(0, 1)
            face = (face * 255).to(torch.uint8).cpu().numpy()
            out_frame = image_processor.restorer.restore_img(video_frames[index], face, affine_matrices[index])
            out_frames.append(out_frame)
        return np.stack(out_frames, axis=0)
    
    def restore_segment(self, decoded_latents, original_frames, boxes, affine_matrices, image_processor):
        # Ensure the inputs are limited to the current segment
        restored_segment = self.restore_video(
            faces=decoded_latents,  # Decoded latents for the current segment
            video_frames=original_frames,  # Corresponding original frames for the segment
            boxes=boxes,  # Bounding boxes for the segment
            affine_matrices=affine_matrices,  # Affine matrices for the segment
            image_processor=image_processor,
        )

        # 1. 确保数值范围正确 [0,1] -> [0,255]
        if restored_segment.max() <= 1.0:
            restored_segment = (restored_segment * 255).astype(np.uint8)
        else:
            restored_segment = restored_segment.astype(np.uint8)
        
        restored_segment = np.clip(restored_segment, 0, 255).astype(np.uint8)

        # 提取第一帧
        first_frame = restored_segment[0]  # 得到形状为 (1024, 576, 3)

        # 2. 检查颜色通道顺序（RGB -> BGR）
        if first_frame.shape[-1] == 3:  # 如果有 3 个通道,如Shape: (16, 1024, 576, 3)通表示每个像素有 3 个颜色通道（RGB 或 BGR）
            first_frame = cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR)

        # 3. 保存第一帧图像
        cv2.imwrite("debug_frame_fixed.png", first_frame)
        
        return restored_segment
    
    def _process_chunk(
        self,
        inference_video_frames: torch.Tensor,
        audio_embeds: Optional[torch.Tensor],
        latents: torch.Tensor,
        original_frames_chunk: List[np.ndarray],
        boxes_chunk: List,
        affine_matrices_chunk: List,
        # --- 以下是共享参数 ---
        do_classifier_free_guidance: bool,
        guidance_scale: float,
        extra_step_kwargs: dict,
        height: int,
        width: int,
        num_inference_steps: int,
        weight_dtype: torch.dtype,
        generator: Optional[torch.Generator],
        callback: Optional[Callable] = None,
        callback_steps: Optional[int] = 1,
        progress_bar_pos: Optional[int] = None,
        chunk_index_for_desc: Optional[int] = None,
    ) -> np.ndarray:
        """
        处理单个视频/音频块的核心推理逻辑。
        这个方法将由每个工作进程在它自己的GPU上调用。
        """
        device = self._execution_device
        image_processor = ImageProcessor(
            height, 
            mask="fix_mask", 
            device=str(device) # <--- 将 torch.device("cuda:0") 转换为 "cuda:0"
        )
        
        pixel_values, masked_pixel_values, masks = image_processor.prepare_masks_and_masked_images(
            inference_video_frames, affine_transform=False
        )
        
        mask_latents, masked_image_latents = self.prepare_mask_latents(
            masks, masked_pixel_values, height, width, weight_dtype, device, generator, do_classifier_free_guidance,
        )

        image_latents = self.prepare_image_latents(
            pixel_values, device, weight_dtype, generator, do_classifier_free_guidance,
        )
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        # timesteps 参数现在由子进程的调度器生成，我们不再需要从主进程传递它
        timesteps = self.scheduler.timesteps

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        iterable = timesteps
        if progress_bar_pos is not None:
            iterable = tqdm.tqdm(
                timesteps,
                desc=f"GPU-{progress_bar_pos} Chunk-{chunk_index_for_desc}",
                position=progress_bar_pos, # 关键：为每个进程分配固定行
                leave=True               # 关键：完成后消失，保持终端整洁
            )
        for j, t in enumerate(iterable):
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            latent_model_input = torch.cat(
                [latent_model_input, mask_latents, masked_image_latents, image_latents], dim=1
            )
            
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=audio_embeds).sample

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_audio = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_audio - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

        decoded_latents = self.decode_latents(latents)
        decoded_latents = self.paste_surrounding_pixels_back(
            decoded_latents, pixel_values, 1 - masks, device, weight_dtype
        )
        restored_segment = self.restore_segment(
            decoded_latents=decoded_latents,
            original_frames=original_frames_chunk,
            boxes=boxes_chunk,
            affine_matrices=affine_matrices_chunk,
            image_processor=image_processor,
        )
        
        return restored_segment

    
def get_audio_chunk(audio_samples, frame_index, num_frames, video_fps, audio_sample_rate):
    """
    根据视频帧的索引和数量，提取对应的音频样本块。
    """
    start_sample = int(frame_index / video_fps * audio_sample_rate)
    end_sample = int((frame_index + num_frames) / video_fps * audio_sample_rate)
    chunk = audio_samples[start_sample:end_sample]
    return chunk.cpu().numpy()

def push_video_to_rtmp(synced_video_frames, video_fps, audio_samples, audio_sample_rate, rtmp_url):
    # 获取视频帧的形状
    num_frames, height, width, channels = synced_video_frames.shape

    # 创建两个命名管道
    video_fifo = "/tmp/video_fifo"
    audio_fifo = "/tmp/audio_fifo"
    # 检查并删除已存在的 FIFO 文件
    if os.path.exists(video_fifo):
        os.remove(video_fifo)
    if os.path.exists(audio_fifo):
        os.remove(audio_fifo)
    # 创建新的 FIFO 文件
    os.mkfifo(video_fifo)
    os.mkfifo(audio_fifo)

    try:
        # 定义 FFmpeg 命令
        ffmpeg_command = [
            'ffmpeg',
            '-y',  # 覆盖输出文件（如果有）
            "-re",  # 实时模式
            # 视频输入
            '-f', 'rawvideo',  # 输入格式为原始视频帧
            '-pix_fmt', 'rgb24',  # 像素格式为 BGR（OpenCV 默认格式）
            '-s', f'{width}x{height}',  # 视频分辨率
            '-r', str(video_fps),  # 帧率
            '-i', video_fifo,  # 视频管道
            # 音频输入
            '-f', 's16le',  # 输入格式为 PCM 音频
            '-ar', str(audio_sample_rate),  # 音频采样率
            '-ac', '1',  # 单声道
            '-thread_queue_size', '1024',
            '-i', audio_fifo,  # 音频管道
            # 显式映射音视频流
            '-map', '0:v',  # 第一个输入的视频流
            '-map', '1:a',  # 第二个输入的音频流
            # 视频输出
            '-c:v', 'libx264',  # 视频编码器
            '-pix_fmt', 'yuv420p',  # 输出像素格式
            '-preset', 'fast',  # 编码速度与质量平衡
            "-crf", "18",
            # 音频输出
            '-c:a', 'aac',  # 音频编码器
            '-b:a', '96k',  # 音频比特率
            '-use_wallclock_as_timestamps', '1',
            '-f', 'flv',  # 输出格式为 FLV（RTMP 流）
            rtmp_url  # RTMP 服务器地址
        ]

        # 启动 FFmpeg 进程
        ffmpeg_process = subprocess.Popen(ffmpeg_command)

        # 定义写入视频帧的函数
        def write_video():
            with open(video_fifo, 'wb') as v_pipe:
                while True:
                    for frame in synced_video_frames:
                        # 转换帧为 bytes 并写入
                        if frame.max() <= 1.0:
                            frame = (frame * 255).astype(np.uint8)
                        else:
                            frame = frame.astype(np.uint8)
                        v_pipe.write(frame.tobytes())
                        time.sleep(1 / video_fps)  # 控制推流速度

        # 定义写入音频样本的函数
        def write_audio():
            with open(audio_fifo, 'wb') as a_pipe:
                chunk_size = int(audio_sample_rate / video_fps)  # 每帧对应的音频样本数
                while True:
                    for i in range(0, len(audio_samples), chunk_size):
                        chunk = audio_samples[i:i + chunk_size]
                        if len(chunk) < chunk_size:
                            chunk = np.append(chunk, np.zeros(chunk_size - len(chunk), dtype=np.int16))

                        # 将浮点数转换为 int16
                        chunk = (chunk * 32767).astype(np.int16)  # 缩放并转换
                        a_pipe.write(chunk.tobytes())

        # 启动线程
        video_thread = threading.Thread(target=write_video, daemon=True)
        audio_thread = threading.Thread(target=write_audio, daemon=True)
        video_thread.start()
        audio_thread.start()

        # 等待 FFmpeg 进程结束
        ffmpeg_process.wait()

    except KeyboardInterrupt:
        print("推流已手动停止")
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        # 删除 FIFO 文件
        if os.path.exists(video_fifo):
            os.remove(video_fifo)
        if os.path.exists(audio_fifo):
            os.remove(audio_fifo)
        print("视频推流完成！")


