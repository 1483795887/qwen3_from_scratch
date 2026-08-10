from qwen3_from_scratch.inference.sync_engine import SyncEngine


def get_config_path():
    from pathlib import Path

    return Path(__file__).parent / "configs" / "batch2.yaml"


def main():
    engine = SyncEngine(get_config_path(), "qwen3-0.6b")
    engine.warmup()

    print("=== 单请求流式（带指标） ===")
    for chunk in engine.generate_stream([{"role": "user", "content": "介绍一下自己"}]):
        print(chunk.delta, end="")
    print(f"\nTTFT: {chunk.metrics.ttft:.4f}s | "
          f"tokens: {chunk.metrics.token_count} | "
          f"TPS: {chunk.metrics.tps:.2f} | "
          f"total: {chunk.metrics.total_elapsed:.4f}s")

    print("\n=== 单请求非流式 ===")
    text = engine.generate("你好")
    print(text)

    print("\n=== 整批非流式（连续批处理 + 汇总指标） ===")
    prompts = [
        [{"role": "user", "content": "介绍一下自己"}],
        [{"role": "user", "content": "1+1等于几"}],
        [{"role": "user", "content": "什么是深度学习"}],
    ]
    texts, batch = engine.batch_generate(prompts)
    for t in texts:
        print(f"- {t}")
    print(f"\n请求数: {batch.num_requests} | "
          f"总 token: {batch.total_tokens} | "
          f"总耗时: {batch.total_elapsed:.4f}s | "
          f"整批吞吐: {batch.aggregate_tps:.2f} tokens/s")
    for i, m in enumerate(batch.per_request):
        print(f"  req{i}: TTFT {m.ttft:.4f}s, TPS {m.tps:.2f}, tokens {m.token_count}")

    # 无 close()：进程结束即自然退出


if __name__ == "__main__":
    main()
