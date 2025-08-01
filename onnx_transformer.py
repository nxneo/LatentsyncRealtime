import torch.onnx
from latentsync.whisper.audio2feature import Audio2Feature
from omegaconf import OmegaConf
config_path="configs/unet/second_stage.yaml"
config = OmegaConf.load(config_path)
if config.model.cross_attention_dim == 768:
    whisper_model_path = "checkpoints/whisper/small.pt"
elif config.model.cross_attention_dim == 384:
    whisper_model_path = "checkpoints/whisper/tiny.pt"
else:
    raise NotImplementedError("cross_attention_dim must be 768 or 384")

print(f"Loading whisper model from {whisper_model_path}")
# 加载 Whisper 模型
model = Audio2Feature(model_path=whisper_model_path, device="cuda", num_frames=config.data.num_frames)

# 模拟输入（这里你需要提供一个符合模型输入要求的张量）
dummy_input = torch.randn(1, config.data.num_frames).to("cuda")

# 导出为 ONNX 格式
onnx_path = "./onnx_model/whisper_model.onnx"
torch.onnx.export(model, dummy_input, onnx_path, export_params=True, opset_version=11)