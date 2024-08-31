import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers import BitsAndBytesConfig

# 모델을 다운로드 받을 경로 설정
os.environ["TRANSFORMERS_CACHE"] = "./prevenotics/cache/"
os.environ["HF_HOME"] = "./prevenotics/cache/"

# 사용 가능한 GPU 개수 확인
num_gpus = torch.cuda.device_count()
print(f"사용 가능한 GPU 개수: {num_gpus}")

# 모델 ID 설정
model_id = "m42-health/Llama3-Med42-70B"
save_path = "./local_model/med42_bitsandbytes_4bit"

# LLM 모델 및 토크나이저 다운로드
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 8-bit 양자화 설정으로 변경
quantization_config = BitsAndBytesConfig(load_in_4bit_fp32_cpu_offload=True)

# 모델 설정 로드
config = AutoConfig.from_pretrained(model_id)

# 모델 로드 (디스크 오프로딩 적용)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    config=config,
    device_map='auto',
    quantization_config=quantization_config,
    torch_dtype=torch.float16,
    offload_folder="offload",
)

# 모델과 토크나이저 저장
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)

print("모델, 토크나이저, 벡터 저장소가 저장되었습니다.")