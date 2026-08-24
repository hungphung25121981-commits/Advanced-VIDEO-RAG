"""Qwen2.5-VL-8B-Instruct runner (4-bit NF4 bitsandbytes quantization).

This is the SINGLE model powering the entire pipeline:
  - Phase 2: visual scene captioning + structured OCR/table description
  - Phase 4: sub-query decomposition, RAG relevance evaluation, and
             final answer generation (text-only calls, same engine)

The class is intentionally a lazy singleton loader (`get_qwen_engine()`)
so that Phase 4 modules can import and reuse the exact same in-memory
model/tokenizer without reloading (~16GB VRAM in 4-bit) a second time.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Optional

from src.utils_common import format_dynamic_metadata_block, free_gpu_memory, get_logger, load_config

logger = get_logger(__name__)

_ENGINE_SINGLETON: Optional["QwenVLEngine"] = None


def _load_images(image_paths: list[str | Path]):
    """Load local keyframe images as PIL Images for the Qwen2.5-VL processor.

    Minimal, dependency-free stand-in for `qwen_vl_utils.process_vision_info`'s
    image path -- see the comment in `QwenVLEngine.generate` for why we don't
    import that package. Only local file paths are needed here (Phase 2 always
    passes on-disk keyframe paths), so no URL/base64 handling is required.
    """
    from PIL import Image

    return [Image.open(str(p)).convert("RGB") for p in image_paths]


VISUAL_CAPTION_SYSTEM_PROMPT = (
    "You are an expert video visual analyst. Describe the provided sequential "
    "keyframes representing a 15-30 second video segment. Since there is no "
    "audio transcript, your description must be precise, concise, and focused "
    "on technical actions, text overlays, structured tables, charts, and "
    "news/UI elements.\n\n"
    "FOCUS AREAS:\n"
    "1. User Actions & Interactions: Mouse clicks, menu navigation, button "
    "presses, typing, window switching, presenter gestures.\n"
    "2. UI, Code & Tables: Extract structured table values, chart data, "
    "specific active windows, visible code snippets, news banners, file "
    "names, URLs, terminal commands.\n"
    "3. Visual Changes: Significant UI or scene state changes across the "
    "sequence.\n\n"
    "OUTPUT FORMAT:\n"
    "Provide a single, dense paragraph (maximum {word_limit} words). Do not "
    "use introductory fluff (e.g., \"In these images...\"). State facts "
    "directly."
)


class QwenVLEngine:
    """Thin wrapper around Qwen2.5-VL-8B-Instruct loaded in 4-bit NF4."""

    def __init__(self, config: Optional[dict] = None):
        self.cfg = config or load_config()
        p2 = self.cfg["phase2"]
        self.model_id: str = p2["vlm_model_id"]
        self.max_new_tokens: int = p2["vlm_max_new_tokens"]
        self.temperature: float = p2["vlm_temperature"]
        self.top_p: float = p2["vlm_top_p"]
        self.model = None
        self.processor = None
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return

        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

        compute_dtype = getattr(torch, self.cfg["phase2"]["vlm_compute_dtype"], torch.bfloat16)

        logger.info("Loading %s in 4-bit NF4 (compute_dtype=%s)...", self.model_id, compute_dtype)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map=self.cfg["runtime"]["device_map"],
            torch_dtype=compute_dtype,
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model.eval()
        self._loaded = True
        logger.info("Qwen2.5-VL-8B-Instruct loaded and ready.")

    def unload(self) -> None:
        """Explicitly release the model from VRAM (e.g. between Kaggle sessions)."""
        self.model = None
        self.processor = None
        self._loaded = False
        gc.collect()
        free_gpu_memory()
        logger.info("Qwen2.5-VL-8B-Instruct unloaded.")

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------
    def generate(
        self,
        text_prompt: str,
        image_paths: Optional[list[str | Path]] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Run one Qwen2.5-VL generation call, with 0..N images + a text prompt."""
        self.load()
        import torch

        image_paths = image_paths or []
        content = [{"type": "image", "image": str(p)} for p in image_paths]
        content.append({"type": "text", "text": text_prompt})

        messages = [{"role": "user", "content": content}]
        chat_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # NOTE: this pipeline only ever feeds Qwen2.5-VL still keyframes (never raw
        # video), so we load images directly with PIL instead of pulling in the
        # `qwen-vl-utils` package's `process_vision_info`. That package's video
        # code path unconditionally imports decord/av at module import time, and
        # those wheels are compiled against an older numpy C-ABI -- on Kaggle's
        # numpy 2.0.2 base image that raises
        # `ValueError: numpy.dtype size changed, may indicate binary incompatibility`
        # before we ever get to use the (unused) video functionality. The
        # HF Qwen2.5-VL image processor already does its own min/max-pixel smart
        # resize internally, so nothing is lost by skipping qwen-vl-utils here.
        image_inputs = _load_images(image_paths)

        inputs = self.processor(
            text=[chat_text],
            images=image_inputs,
            videos=None,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        effective_temperature = temperature if temperature is not None else self.temperature
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens if max_new_tokens is not None else self.max_new_tokens,
            do_sample=effective_temperature > 0,
        )
        # transformers warns/errors on sampling-only kwargs (temperature, top_p) being set
        # while do_sample=False (greedy decoding ignores them) -- only include them when
        # actually sampling, and always resolve 0.0 to greedy instead of silently falling
        # back to the config default (the original bug this fixes).
        if gen_kwargs["do_sample"]:
            gen_kwargs["temperature"] = effective_temperature
            gen_kwargs["top_p"] = self.top_p

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)

        trimmed_ids = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]

        del inputs, generated_ids, trimmed_ids
        return output_text.strip()


def get_qwen_engine(config: Optional[dict] = None) -> QwenVLEngine:
    """Return the process-wide singleton QwenVLEngine (lazy-loaded on first `.generate`)."""
    global _ENGINE_SINGLETON
    if _ENGINE_SINGLETON is None:
        _ENGINE_SINGLETON = QwenVLEngine(config=config)
    return _ENGINE_SINGLETON


# ----------------------------------------------------------------------
# Phase 2 specific: segment visual captioning
# ----------------------------------------------------------------------
def build_caption_prompt(video_info: dict, word_limit: int) -> str:
    """Build the [GLOBAL VIDEO CONTEXT] + [SYSTEM INSTRUCTION] prompt block."""
    metadata_block = format_dynamic_metadata_block(video_info)
    system_instruction = VISUAL_CAPTION_SYSTEM_PROMPT.format(word_limit=word_limit)
    return f"[GLOBAL VIDEO CONTEXT]\n{metadata_block}\n\n[SYSTEM INSTRUCTION: VISUAL SCENE CAPTIONER]\n{system_instruction}"


def caption_segment(
    keyframe_paths: list[str | Path],
    video_info: dict,
    engine: Optional[QwenVLEngine] = None,
) -> str:
    """Generate the dense visual_caption paragraph for one segment's keyframes."""
    cfg = load_config()
    p2 = cfg["phase2"]
    engine = engine or get_qwen_engine(cfg)

    sampled_paths = keyframe_paths[: p2["vlm_images_per_call"]]
    prompt = build_caption_prompt(video_info, p2["vlm_caption_word_limit"])

    caption = engine.generate(
        text_prompt=prompt,
        image_paths=sampled_paths,
        max_new_tokens=p2["vlm_max_new_tokens"],
        temperature=p2["vlm_temperature"],
    )
    return caption
