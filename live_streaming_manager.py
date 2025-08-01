# live_streaming_manager.py

import queue
import threading
from concurrent.futures import ThreadPoolExecutor
import time
import numpy as np
import cv2
import subprocess
import os
import torch
import logging

from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from latentsync.utils.util import read_audio

class Signal:
    def __init__(self, name): self.name = name
    def __repr__(self): return f"Signal({self.name})"

TRANSITION_TO_ACTIVE = Signal("TRANSITION_TO_ACTIVE")
TASK_COMPLETE = Signal("TASK_COMPLETE")

class LiveStreamingManager:
    # ... __init__, start, stop, submit_task, _task_processor_loop, _process_and_queue_chunk, _create_transition_frames ...
    # (以上方法保持不变，直接从上一个回答中复制即可)

    def __init__(self, pipeline: LipsyncPipeline, rtmp_url: str, max_workers: int = 2, video_fps: int = 25, audio_sample_rate: int = 16000):
        self.pipeline = pipeline
        self.rtmp_url = rtmp_url
        self.max_workers = max_workers
        self.video_fps = video_fps
        self.audio_sample_rate = audio_sample_rate
        self.frame_duration = 1.0 / video_fps
        self.height = self.pipeline.unet.config.sample_size * self.pipeline.vae_scale_factor
        self.width = self.height
        self.task_queue = queue.Queue()
        self.result_queue = queue.PriorityQueue()
        self.task_processor_thread = None
        self.streamer_thread = None
        self._stop_event = threading.Event()
        self.is_running = False
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("LiveStreamingManager instance created.")

    def start(self):
        if self.is_running:
            self.logger.warning("Manager is already running.")
            return
        self.logger.info("Starting Live Streaming Manager...")
        self._stop_event.clear()
        self.task_processor_thread = threading.Thread(target=self._task_processor_loop, name="TaskProcessorThread", daemon=True)
        self.streamer_thread = threading.Thread(target=self._streamer_loop, name="StreamerThread", daemon=True)
        self.task_processor_thread.start()
        self.streamer_thread.start()
        self.is_running = True
        self.logger.info(f"Manager started. Streaming to {self.rtmp_url}. Ready for tasks.")

    def stop(self):
        if not self.is_running:
            self.logger.warning("Manager is not running.")
            return
        self.logger.info("Stopping Live Streaming Manager...")
        self._stop_event.set()
        try:
            self.task_queue.put_nowait(None)
        except queue.Full:
            self.logger.warning("Task queue is full, could not place stop signal.")
        try:
            self.result_queue.put_nowait((-1, None))
        except queue.Full:
            self.logger.warning("Result queue is full, could not place stop signal.")
        if self.task_processor_thread and self.task_processor_thread.is_alive():
            self.logger.info("Waiting for task processor thread to finish...")
            self.task_processor_thread.join(timeout=5)
        if self.streamer_thread and self.streamer_thread.is_alive():
            self.logger.info("Waiting for streamer thread to finish...")
            self.streamer_thread.join(timeout=5)
        self.is_running = False
        self.logger.info("Manager stopped.")

    def submit_task(self, audio_path: str, video_path: str):
        if not self.is_running:
            self.logger.error("Error: Manager is not running. Please call start() first.")
            raise RuntimeError("Manager is not running.")
        task = {"audio_path": audio_path, "video_path": video_path}
        self.logger.info(f"Submitting new task: {task}")
        self.task_queue.put(task)

    def _task_processor_loop(self):
        self.logger.info("Task processor loop started.")
        with ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="ChunkProcessor") as executor:
            while not self._stop_event.is_set():
                try:
                    task = self.task_queue.get(timeout=1)
                    if task is None: continue
                    self.logger.info(f"Processing task: {task['audio_path']}")
                    p = self.pipeline
                    audio_path = task['audio_path']
                    video_path = task['video_path']
                    video_frames, orig_v_frames, boxes, matrices = p.affine_transform_video(video_path, self.height, self.width, "fix_mask")
                    whisper_feature = p.audio_encoder.audio2feat(audio_path)
                    audio_embed_chunks = p.audio_encoder.feature2chunks(feature_array=whisper_feature, fps=self.video_fps)
                    raw_audio_samples = read_audio(audio_path, sample_rate=self.audio_sample_rate).cpu().numpy()
                    chunk_size = 16
                    samples_per_chunk = int(chunk_size / self.video_fps * self.audio_sample_rate)
                    num_chunks = min(len(video_frames) // chunk_size, len(audio_embed_chunks) // chunk_size, len(raw_audio_samples) // samples_per_chunk)
                    if num_chunks == 0:
                        self.logger.warning(f"Warning: Not enough data in task {task} to process. Skipping.")
                        continue
                    self.result_queue.put((-2, TRANSITION_TO_ACTIVE))
                    for i in range(num_chunks):
                        v_start, v_end = i * chunk_size, (i + 1) * chunk_size
                        a_start, a_end = i * samples_per_chunk, (i + 1) * samples_per_chunk
                        audio_embeds = torch.stack(audio_embed_chunks[v_start:v_end]).to(p.device, dtype=torch.float16)
                        if p.unet.config.in_channels > p.unet.config.in_channels // 2 * 2 :
                           audio_embeds = torch.cat([torch.zeros_like(audio_embeds), audio_embeds])
                        executor.submit(
                            self._process_and_queue_chunk,
                            chunk_index=i,
                            raw_audio_chunk=raw_audio_samples[a_start:a_end],
                            inference_video_frames=video_frames[v_start:v_end],
                            audio_embeds=audio_embeds,
                            original_frames_chunk=orig_v_frames[v_start:v_end],
                            boxes_chunk=boxes[v_start:v_end],
                            affine_matrices_chunk=matrices[v_start:v_end]
                        )
                    self.result_queue.put((num_chunks, TASK_COMPLETE))
                except queue.Empty:
                    continue
                except Exception as e:
                    self.logger.error(f"Error in task processor loop: {e}", exc_info=True)
        self.logger.info("Task processor loop finished.")

    def _process_and_queue_chunk(self, chunk_index, raw_audio_chunk, **kwargs):
        try:
            processed_video_chunk = self.pipeline.process_chunk(
                height=self.height, width=self.width, 
                num_inference_steps=4, guidance_scale=2.5, weight_dtype=torch.float16,
                **kwargs
            )
            self.result_queue.put((chunk_index, (processed_video_chunk, raw_audio_chunk)))
            self.logger.info(f"✅ Chunk {chunk_index} processed and queued.")
        except Exception as e:
            self.logger.error(f"Error processing chunk {chunk_index}: {e}", exc_info=True)

    @staticmethod
    def _create_transition_frames(start_frame, end_frame, num_frames=12):
        frames = []
        for i in range(num_frames):
            alpha = (i + 1) / (num_frames + 1)
            blended = cv2.addWeighted(end_frame.astype(np.float32), alpha, start_frame.astype(np.float32), 1 - alpha, 0)
            frames.append(blended.astype(np.uint8))
        return frames

    # ===============================================
    #  `_streamer_loop` 的最终版
    # ===============================================
    def _streamer_loop(self):
        self.logger.info("Streamer loop started.")
        ffmpeg_process = None
        pid = os.getpid()
        video_fifo = f"/tmp/video_fifo_{pid}"
        audio_fifo = f"/tmp/audio_fifo_{pid}"
        
        try:
            for fifo in [video_fifo, audio_fifo]:
                if os.path.exists(fifo): os.remove(fifo)
                os.mkfifo(fifo)

            command = [
                'ffmpeg', '-y', '-re',
                '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{self.width}x{self.height}', '-r', str(self.video_fps), '-i', video_fifo,
                '-f', 's16le', '-ar', str(self.audio_sample_rate), '-ac', '1', '-i', audio_fifo,
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'veryfast', '-tune', 'zerolatency',
                '-c:a', 'aac', '-b:a', '96k',
                '-f', 'flv', self.rtmp_url
            ]
            
            self.logger.info(f"Starting FFmpeg with command: {' '.join(command)}")
            ffmpeg_process = subprocess.Popen(command, stderr=subprocess.PIPE, text=True)

            def log_ffmpeg_errors(proc):
                for line in iter(proc.stderr.readline, ''):
                    self.logger.error(f"[FFMPEG]: {line.strip()}")
                self.logger.info("[FFMPEG] stderr stream closed.")
            ffmpeg_log_thread = threading.Thread(target=log_ffmpeg_errors, args=(ffmpeg_process,), daemon=True)
            ffmpeg_log_thread.start()
            
            state = "idle"
            expected_chunk_index = 0
            idle_video_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            last_frame_cache = idle_video_frame.copy()
            samples_per_frame = int(self.audio_sample_rate / self.video_fps)
            silent_audio_chunk = np.zeros(samples_per_frame, dtype=np.int16)
        
            with open(video_fifo, 'wb') as v_pipe, open(audio_fifo, 'wb') as a_pipe:
                while not self._stop_event.is_set():
                    if ffmpeg_process.poll() is not None:
                        self.logger.error(f"FFmpeg process terminated unexpectedly with code {ffmpeg_process.returncode}.")
                        break

                    try:
                        if state == "idle":
                            # 在 idle 状态，我们不需要严格的 sleep，因为 get 会超时
                            v_pipe.write(last_frame_cache.tobytes())
                            a_pipe.write(silent_audio_chunk.tobytes())
                            v_pipe.flush(); a_pipe.flush()
                            
                            try:
                                priority, signal = self.result_queue.get(timeout=self.frame_duration)
                                if signal is None: break
                                if signal is TRANSITION_TO_ACTIVE:
                                    state = "transition_to_active"
                                    self.logger.info("State changed: idle -> transition_to_active")
                                self.result_queue.task_done()
                            except queue.Empty:
                                continue

                        else: # active, transition_to_active, transition_to_idle 状态
                            # ===============================================
                            #  FIX: 逐帧写入并手动控制速率
                            # ===============================================
                            if state == "transition_to_active":
                                try:
                                    priority, data = self.result_queue.get(timeout=5)
                                    if data is None: break
                                    if isinstance(data, tuple) and priority == 0:
                                        video_chunk, audio_chunk = data
                                        
                                        # --- 推送过渡帧 ---
                                        first_frame = video_chunk[0]
                                        transition_frames = self._create_transition_frames(last_frame_cache, first_frame)
                                        for frame in transition_frames:
                                            if self._stop_event.is_set(): break
                                            v_pipe.write(frame.tobytes())
                                            a_pipe.write(silent_audio_chunk.tobytes())
                                            v_pipe.flush(); a_pipe.flush()
                                            time.sleep(self.frame_duration)
                                        if self._stop_event.is_set(): break
                                        
                                        # --- 推送第一个真实块 (逐帧) ---
                                        audio_pcm = (audio_chunk * 32767).astype(np.int16)
                                        for i in range(len(video_chunk)):
                                            if self._stop_event.is_set(): break
                                            v_pipe.write(video_chunk[i].tobytes())
                                            audio_offset = i * samples_per_frame
                                            a_pipe.write(audio_pcm[audio_offset:audio_offset + samples_per_frame].tobytes())
                                            v_pipe.flush(); a_pipe.flush()
                                            time.sleep(self.frame_duration)
                                        if self._stop_event.is_set(): break

                                        last_frame_cache = video_chunk[-1]
                                        state = "active"
                                        expected_chunk_index = 1
                                        self.logger.info("State changed: transition_to_active -> active")
                                    self.result_queue.task_done()
                                except queue.Empty:
                                    self.logger.warning("Warning: Timed out waiting for first chunk. Reverting to idle.")
                                    state = "idle"

                            elif state == "active":
                                try:
                                    priority, data = self.result_queue.get(timeout=self.frame_duration * 17) # 等待下一个块
                                    if data is None: break
                                    if data is TASK_COMPLETE:
                                        state = "transition_to_idle"
                                        self.logger.info("State changed: active -> transition_to_idle")
                                    elif isinstance(data, tuple) and priority == expected_chunk_index:
                                        video_chunk, audio_chunk = data
                                        
                                        # --- 推送后续真实块 (逐帧) ---
                                        audio_pcm = (audio_chunk * 32767).astype(np.int16)
                                        for i in range(len(video_chunk)):
                                            if self._stop_event.is_set(): break
                                            v_pipe.write(video_chunk[i].tobytes())
                                            audio_offset = i * samples_per_frame
                                            a_pipe.write(audio_pcm[audio_offset:audio_offset + samples_per_frame].tobytes())
                                            v_pipe.flush(); a_pipe.flush()
                                            time.sleep(self.frame_duration)
                                        if self._stop_event.is_set(): break
                                        
                                        last_frame_cache = video_chunk[-1]
                                        expected_chunk_index += 1
                                    self.result_queue.task_done()
                                except queue.Empty:
                                    self.logger.warning("Warning: Timed out waiting for next chunk. Ending active session.")
                                    state = "transition_to_idle"

                            elif state == "transition_to_idle":
                                idle_frame = np.zeros_like(last_frame_cache)
                                transition_frames = self._create_transition_frames(last_frame_cache, idle_frame)
                                for frame in transition_frames:
                                    if self._stop_event.is_set(): break
                                    v_pipe.write(frame.tobytes())
                                    a_pipe.write(silent_audio_chunk.tobytes())
                                    v_pipe.flush(); a_pipe.flush()
                                    time.sleep(self.frame_duration)
                                if self._stop_event.is_set(): break

                                last_frame_cache = idle_frame
                                state = "idle"
                                expected_chunk_index = 0
                                self.logger.info("State changed: transition_to_idle -> idle")

                    except BrokenPipeError:
                        self.logger.error("FFmpeg process pipe closed unexpectedly. Stopping streamer.")
                        break
            
        except Exception as e:
            self.logger.error(f"Critical error in streamer loop: {e}", exc_info=True)
        finally:
            self.logger.info("Stopping FFmpeg...")
            if ffmpeg_process and ffmpeg_process.poll() is None:
                try:
                    ffmpeg_process.terminate()
                    ffmpeg_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.logger.warning("FFmpeg did not terminate in time, killing.")
                    ffmpeg_process.kill()
            for fifo in [video_fifo, audio_fifo]:
                if os.path.exists(fifo):
                    os.remove(fifo)
            self.logger.info("Streamer loop finished.")