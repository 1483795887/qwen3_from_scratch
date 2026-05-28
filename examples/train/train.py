from qwen3_from_scratch.factory import ComponentFactory, ModelConfig
from qwen3_from_scratch.models.qwen3 import Qwen3
import torch
from qwen3_from_scratch.inference.context import ModelContext
import torch.nn as nn
from torch.optim import AdamW
from dataset import PretrainDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def count_model_params(model: nn.Module):
    """
    统计模型总参数 & 可训练参数
    返回: 总参数量, 可训练参数量, 参数量说明(字符串)
    """
    total_params = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        total_params += param.numel()  # 每个张量的总元素数 = 参数数量
        if param.requires_grad:
            trainable_params += param.numel()

    # 转成易读格式（M / B）
    def fmt(n):
        if n >= 1e9:
            return f"{n / 1e9:.2f}B"
        elif n >= 1e6:
            return f"{n / 1e6:.2f}M"
        else:
            return f"{n:,}"

    info = (
        f"✅ 总参数量: {fmt(total_params)} ({total_params:,})\n"
        f"✅ 可训练参数量: {fmt(trainable_params)} ({trainable_params:,})"
    )
    return total_params, trainable_params, info
def build_model():
    config = ModelConfig(
        vocab_size=50258,  # GPT2分词器 + 1 (PAD token)
        hidden_size=128,  # 朝着50M来的
        num_hidden_layers=12,
        eos_token_id=50256,  # 虽然用不着
        num_attention_heads=8,
        num_key_value_heads=4,
        intermediate_size=128 * 3,
    )
    # 所有组件使用 base ，因为非 base 都没写反向传播
    return Qwen3(config=config)


def train(data_path: str):
    model = build_model()
    info = count_model_params(model)
    print(info)

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_pretrained("gpt2")
    dataset = PretrainDataset(data_path, tokenizer)

    BATCH_SIZE = 16
    EPOCHS = 10

    data_loader = DataLoader(dataset, batch_size=16)
    optimizer = AdamW(model.parameters())
    pad_id = 50257  # GPT2 vocab_size，即新增的 PAD token id
    loss_func = nn.CrossEntropyLoss(ignore_index=pad_id)

    all_losses = []
    for e in range(EPOCHS):
        model.train()
        epoch_losses = []
        pbar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f'Epoch {e+1}/{EPOCHS}', leave=False)
        for step_idx, (x, y) in pbar:
            optimizer.zero_grad()
            context = ModelContext()
            pred = model(x, context)  # [B, L, V]
            B, L, V = pred.shape
            loss = loss_func(pred.view(B * L, V), y.view(B * L))
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
            pbar.set_postfix(loss=f'{loss.item():.4f}', step=step_idx)
        all_losses.extend(epoch_losses)

        checkpoint_path = f'checkpoint_epoch_{e+1}.pt'
        torch.save(model.state_dict(), checkpoint_path)
        logger.info(f'Saved checkpoint: {checkpoint_path}')

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(all_losses)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig('training_loss.png')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to the training data JSONL file')
    args = parser.parse_args()
    train(args.data_path)
