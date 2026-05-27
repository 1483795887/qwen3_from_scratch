from transformers import AutoModel, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
import torch
import json
import jinja2
from typing import Sequence, Type
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv
import os

load_dotenv()


def w8_a16_lienar(weight, x, scales, bias=None):
    casted_weights = weight.to(x.dtype)
    output = F.linear(x, casted_weights, bias) * scales
    return output


class W8A16Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, dtype=torch.float32):
        super().__init__()
        self.register_buffer(
            "int8_weights", torch.randint(-128, 127, (out_features, in_features), dtype=torch.int8)
        )
        self.register_buffer("scales", torch.randn(out_features, dtype=dtype))
        if bias:
            self.register_buffer("bias", torch.randn(out_features, dtype=dtype))
        else:
            self.bias = None

    def forward(self, x):
        return w8_a16_lienar(self.int8_weights, x, self.scales, self.bias)

    def quantize(self, weights):
        w_fp32 = weights.clone().to(torch.float32)
        scales = w_fp32.abs().max(dim=-1).values / 127
        scales = scales.to(weights.dtype)
        int8_weights = torch.round(weights / scales.unsqueeze(-1)).to(torch.int8)
        self.int8_weights = int8_weights
        self.scales = scales


def replace_linear_with_target(
    module: nn.Module, target_class: Type[nn.Module], module_name_to_exclude: Sequence[str]
):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and not any([x == name for x in module_name_to_exclude]):
            old_bias = child.bias
            old_weight = child.weight
            new_module = target_class(
                child.in_features,
                child.out_features,
                bias=old_bias is not None,
                dtype=old_weight.dtype,
            )
            new_module.quantize(old_weight)
            setattr(module, name, new_module)
            if old_bias is not None:
                getattr(module, name).bias.data = old_bias.data
            # 显式删除旧的权重和偏置
            del old_weight
            if old_bias is not None:
                del old_bias
            del child
        else:
            replace_linear_with_target(child, target_class, module_name_to_exclude)

def main():
    local_path = os.environ.get("MODEL_PATH")
    tokenizer = AutoTokenizer.from_pretrained(local_path)
    model = Qwen3ForCausalLM.from_pretrained(local_path)
    model.eval()
    replace_linear_with_target(model, W8A16Linear, ["lm_head"])
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(model)
    # 加载 chat template
    with open(local_path + "/tokenizer_config.json") as f:
        data = json.load(f)
        template = jinja2.Template(data["chat_template"])
        prompt = template.render(
            messages=[{"role": "user", "content": "介绍一下你自己"}]
        )
    
    # 使用 tokenizer 返回张量格式，并移动到设备
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 生成回复，设置必要的参数
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
        )
    
    # 解码并打印结果
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(response)


if __name__ == "__main__":
    main()
