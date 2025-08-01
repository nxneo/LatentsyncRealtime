import requests
import time
import logging
from datetime import datetime

# 配置日志
def setup_logger():
    # 创建 logger 对象
    logger = logging.getLogger("API_Caller")
    logger.setLevel(logging.DEBUG)  # 设置全局日志级别为 DEBUG

    # 创建文件日志处理器
    file_handler = logging.FileHandler("api_caller.log")
    file_handler.setLevel(logging.INFO)  # 文件日志记录 INFO 及以上级别的日志

    # 创建控制台日志处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)  # 控制台日志记录 DEBUG 及以上级别的日志

    # 定义日志格式
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 将处理器添加到 logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# 初始化日志
logger = setup_logger()

# 定义接口 URL 和音频文件路径
API_URL = "http://127.0.0.1:5000/generate_video"
AUDIO_FILE_PATH = "/data0/projects/LatentSync/assets/Wavfile_1719704064_11.wav"

# 每 10 秒调用一次接口
def call_generate_video_api():
    num = 0
    while True:
        try:
            # 打开音频文件并发送 POST 请求
            with open(AUDIO_FILE_PATH, "rb") as audio_file:
                files = {"audio_file": audio_file}
                start_time = time.time()  # 记录请求开始时间
                response = requests.post(API_URL, files=files)
                elapsed_time = time.time() - start_time  # 计算请求耗时

            # 检查响应状态码
            if response.status_code == 200:
                logger.info(f"API call successful at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                logger.debug(f"Response: {response.json()}")
                logger.info(f"Request completed in {elapsed_time:.2f} seconds")
            else:
                logger.error(f"API call failed with status code {response.status_code} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                logger.error(f"Response: {response.text}")

        except FileNotFoundError as e:
            logger.error(f"File not found: {AUDIO_FILE_PATH}. Error: {e}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error occurred: {e}")
        except Exception as e:
            logger.error(f"Unexpected error occurred: {e}")

        # 记录调用次数
        num += 1
        logger.info(f"持续调用次数: {num}")

        # 等待 10 秒
        time.sleep(10)

if __name__ == "__main__":
    logger.info("Starting API caller...")
    call_generate_video_api()