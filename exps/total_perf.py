from torch.profiler import profile, record_function, ProfilerActivity
from qwen3_from_scratch.factory import ComponentFactory, ModelConfig
from qwen3_from_scratch.models.qwen3 import Qwen3
from qwen3_from_scratch.inference.context import ModelContext
import torch
import os
from qwen3_from_scratch.utils.env import load_env_file
from qwen3_from_scratch.factory.config import load_from_file
load_env_file()

def main():
    model_path = os.environ.get("MODEL_PATH")
    config = load_from_file(model_path + "/config.json")
    # config.decoder_layer.name = 'my_op'
    model = Qwen3(config=config).cuda().to(torch.bfloat16).eval()

    x = torch.randint(0, 1000, (1, 1024)).cuda()
    torch.cuda.reset_peak_memory_stats()
    with profile(
        activities=[ProfilerActivity.CUDA], record_shapes=True, with_stack=True
    ) as prof:
        with record_function("model_infer"):
            context = ModelContext()
            model(x, context)
            mem = torch.cuda.max_memory_allocated() / 1024**2
            print(f"显存占用 {mem}")

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

if __name__ == "__main__":
    # 打印 TOP 20 最慢的算子
    main()
