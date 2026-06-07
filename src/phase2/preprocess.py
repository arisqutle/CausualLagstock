import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from pathlib import Path

def process(
    json_path,
    save_path,
    limit=None,
    model_name="transformers/FinBERT",
    device=None,
    batch_size=32,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    data = []
    with open(json_path, encoding="utf-8") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if limit is not None and len(data) >= limit:
                break

    embeddings = []
    meta = []

    with torch.no_grad():
        for start in tqdm(range(0, len(data), batch_size)):
            batch = data[start:start + batch_size]
            texts = [x["headline"] for x in batch]

            enc = tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=128,
                return_tensors="pt"
            ).to(device)

            out = model(**enc)
            cls = out.last_hidden_state[:, 0].cpu()  # (B, 768)

            embeddings.extend(cls)
            for x in batch:
                meta.append({
                    "event_type": x["event_type"],
                    "timestamp": x["source_date"],
                    "ticker": x["source_ticker"]
                })

    embeddings = torch.stack(embeddings)  # (N, 768)

    save_path.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, save_path / "emb.pt")

    with open(save_path / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Embed Phase 2/3 labeled event headlines with FinBERT")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--model-name",
        default="transformers/FinBERT",
        help="Hugging Face model name or local FinBERT directory",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device, for example cuda, cuda:0, or cpu",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    print(torch.version.cuda)
    process(
        Path(args.labels),
        Path(args.output_dir),
        limit=args.limit,
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
    )
