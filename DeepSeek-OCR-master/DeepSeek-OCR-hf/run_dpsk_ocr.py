"""Command line helper for running DeepSeek-OCR inference.

The script exposes a small CLI that mirrors the original example while also
supporting on-the-fly 4-bit quantization using bitsandbytes.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from transformers import AutoModel, AutoTokenizer


DEFAULT_PROMPT = "<image>\n<|grounding|>Convert the document to markdown."


def str_to_torch_dtype(name: str) -> torch.dtype:
    try:
        return getattr(torch, name)
    except AttributeError as exc:
        raise argparse.ArgumentTypeError(f"Unsupported torch dtype: {name}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run DeepSeek-OCR inference with optional 4-bit quantization."
    )
    parser.add_argument(
        "--model-name",
        default="deepseek-ai/DeepSeek-OCR",
        help="Model identifier or local path registered on Hugging Face.",
    )
    parser.add_argument(
        "--image-file",
        required=True,
        help="Path to the image that should be processed.",
    )
    parser.add_argument(
        "--output-path",
        default="outputs",
        help="Directory where inference artefacts will be written.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt passed to the multimodal model.",
    )
    parser.add_argument(
        "--base-size",
        type=int,
        default=1024,
        help="Base resolution for the global image view.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=640,
        help="Resolution for local cropped views.",
    )
    parser.add_argument(
        "--no-crop",
        dest="crop_mode",
        action="store_false",
        help="Disable cropping strategy (enabled by default).",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device used for non-quantized inference (default: cuda).",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        type=str,
        help="Torch dtype applied to the non-quantized model (default: bfloat16).",
    )
    parser.add_argument(
        "--quantization",
        choices=("none", "4bit"),
        default="none",
        help="Enable 4-bit quantization using bitsandbytes.",
    )
    parser.add_argument(
        "--device-map",
        default=None,
        help="Device map passed to `from_pretrained`. Useful for large models.",
    )
    parser.add_argument(
        "--bnb-compute-dtype",
        default="float16",
        type=str,
        help="bitsandbytes compute dtype used when quantization=4bit.",
    )
    parser.add_argument(
        "--bnb-quant-type",
        default="nf4",
        choices=("nf4", "fp4"),
        help="bitsandbytes quantization data type.",
    )
    parser.add_argument(
        "--no-bnb-double-quant",
        dest="bnb_double_quant",
        action="store_false",
        help="Disable bitsandbytes double quantization (enabled by default).",
    )
    parser.add_argument(
        "--attn-implementation",
        default="flash_attention_2",
        help="Attention implementation passed to Transformers when not quantized.",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES override.",
    )
    parser.add_argument(
        "--no-trust-remote-code",
        dest="trust_remote_code",
        action="store_false",
        help="Disable trusting custom model code from the checkpoint.",
    )
    parser.add_argument(
        "--save-results",
        dest="save_results",
        action="store_true",
        help="Persist OCR outputs (default: enabled).",
    )
    parser.add_argument(
        "--no-save-results",
        dest="save_results",
        action="store_false",
        help="Do not persist OCR outputs.",
    )
    parser.add_argument(
        "--test-compress",
        dest="test_compress",
        action="store_true",
        help="Use compression-aware decoding (default: enabled).",
    )
    parser.add_argument(
        "--no-test-compress",
        dest="test_compress",
        action="store_false",
        help="Disable compression-aware decoding.",
    )
    parser.set_defaults(
        crop_mode=True,
        save_results=True,
        test_compress=True,
        bnb_double_quant=True,
        trust_remote_code=True,
    )
    return parser


def configure_environment(cuda_visible_devices: Optional[str]) -> None:
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices


def build_quantization_config(args: argparse.Namespace) -> Dict[str, object]:
    from transformers import BitsAndBytesConfig

    compute_dtype = str_to_torch_dtype(args.bnb_compute_dtype)
    if compute_dtype not in {torch.float16, torch.bfloat16}:
        raise ValueError(
            "Only float16 and bfloat16 compute dtypes are supported for 4-bit quantization."
        )

    return {
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.bnb_double_quant,
            bnb_4bit_quant_type=args.bnb_quant_type,
        )
    }


def load_model(args: argparse.Namespace) -> Tuple[AutoTokenizer, torch.nn.Module]:
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=args.trust_remote_code
    )

    model_kwargs: Dict[str, object] = {
        "trust_remote_code": args.trust_remote_code,
        "use_safetensors": True,
    }

    if args.quantization == "4bit":
        try:
            model_kwargs.update(build_quantization_config(args))
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "bitsandbytes is required for 4-bit quantization. Install it with `pip install bitsandbytes`."
            ) from exc
        if args.device_map is not None:
            model_kwargs["device_map"] = args.device_map
        model = AutoModel.from_pretrained(args.model_name, **model_kwargs)
    else:
        target_dtype = str_to_torch_dtype(args.dtype)
        model_kwargs["_attn_implementation"] = args.attn_implementation
        if args.device_map is not None:
            model_kwargs["device_map"] = args.device_map
        model = AutoModel.from_pretrained(args.model_name, **model_kwargs)
        model = model.to(device=args.device, dtype=target_dtype)

    model = model.eval()
    return tokenizer, model


def run_inference(args: argparse.Namespace) -> None:
    configure_environment(args.cuda_visible_devices)

    tokenizer, model = load_model(args)

    image_path = Path(args.image_file)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    _ = model.infer(
        tokenizer,
        prompt=args.prompt,
        image_file=str(image_path),
        output_path=str(output_dir),
        base_size=args.base_size,
        image_size=args.image_size,
        crop_mode=args.crop_mode,
        save_results=args.save_results,
        test_compress=args.test_compress,
    )


def main() -> None:
    args = build_parser().parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
