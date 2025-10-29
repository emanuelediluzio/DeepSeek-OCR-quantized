"""Utility script to quantize DeepSeek-OCR weights to 4-bit format."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict

import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig


def str_to_dtype(name: str) -> torch.dtype:
    try:
        return getattr(torch, name)
    except AttributeError as exc:
        raise argparse.ArgumentTypeError(f"Unsupported torch dtype: {name}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a 4-bit quantized copy of the DeepSeek-OCR model."
    )
    parser.add_argument(
        "--model-name",
        default="deepseek-ai/DeepSeek-OCR",
        help="Model identifier or local path registered on Hugging Face.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory that will contain the quantized checkpoint.",
    )
    parser.add_argument(
        "--bnb-compute-dtype",
        default="float16",
        choices=("float16", "bfloat16"),
        help="bitsandbytes compute dtype used during quantization.",
    )
    parser.add_argument(
        "--bnb-quant-type",
        default="nf4",
        choices=("nf4", "fp4"),
        help="Data type of the 4-bit quantized weights.",
    )
    parser.add_argument(
        "--no-bnb-double-quant",
        dest="bnb_double_quant",
        action="store_false",
        help="Disable bitsandbytes double quantization.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Device map passed to `from_pretrained`. Defaults to auto placement.",
    )
    parser.add_argument(
        "--no-trust-remote-code",
        dest="trust_remote_code",
        action="store_false",
        help="Disable trusting custom model code when loading checkpoints.",
    )
    parser.set_defaults(bnb_double_quant=True, trust_remote_code=True)
    return parser.parse_args()


def quantize(args: argparse.Namespace) -> Dict[str, object]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    compute_dtype = str_to_dtype(args.bnb_compute_dtype)

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=args.bnb_double_quant,
        bnb_4bit_quant_type=args.bnb_quant_type,
    )

    model_kwargs: Dict[str, object] = {
        "quantization_config": quant_config,
        "use_safetensors": True,
        "device_map": args.device_map,
        "trust_remote_code": args.trust_remote_code,
    }

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=args.trust_remote_code
    )
    model = AutoModel.from_pretrained(args.model_name, **model_kwargs)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    metadata = {
        "model_name": args.model_name,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "bnb_compute_dtype": args.bnb_compute_dtype,
        "bnb_quant_type": args.bnb_quant_type,
        "bnb_double_quant": args.bnb_double_quant,
        "device_map": args.device_map,
        "pytorch_version": torch.__version__,
    }

    summary_path = output_dir / "quantization_summary.json"
    summary_path.write_text(json.dumps(metadata, indent=2))

    return metadata


def main() -> None:
    args = parse_args()
    metadata = quantize(args)
    print("Quantized model saved to", args.output_dir)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
