import json
import tqdm


def parse_to_jsonl(input_path: str, output_path: str) -> None:
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()

    segments = content.split("<|endoftext|>")
    segments = [s for s in segments if s.strip()]

    with open(output_path, "w", encoding="utf-8") as f:
        for seg in tqdm.tqdm(segments):
            record = {"Text": seg.strip() + "<|endoftext|>"}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parse_to_jsonl(
        input_path=r"/data/tinystories_sample_5M.txt",
        output_path=r"/data/tinystories_sample_5M.jsonl",
    )
