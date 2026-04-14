"""
Adapter to run the zoom consistency router with the official ScreenSpot-Pro eval script.

Place this file in ScreenSpot-Pro-GUI-Grounding/models/zoom_consistency_router.py
and add to model_factory.py.

Usage:
  cd ScreenSpot-Pro-GUI-Grounding
  python eval_screenspot_pro.py \
    --model_type zoom_consistency_router \
    --screenspot_imgs /path/to/ScreenSpot-Pro/images \
    --screenspot_test /path/to/ScreenSpot-Pro/annotations \
    --task all --inst_style instruction --language en --gt_type positive \
    --log_path results/zoom_consistency_router.json
"""
import json
import math
import os
import re
import torch
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Default HuggingFace model identifiers for the two models in the ensemble.
# Override via environment variables to use local paths instead:
#   KV_GROUND_PATH=/local/path/to/kv-ground-8b
#   QWEN_PATH=/local/path/to/qwen3.5-27b-awq
#
# Models:
#   - KV-Ground-8B: GUI grounding specialist fine-tuned from GUI-Owl-1.5
#     https://huggingface.co/vocaela/KV-Ground-8B-BaseGuiOwl1.5-0315
#   - Qwen3.5-27B-AWQ: General-purpose VLM (4-bit AWQ quantized)
#     https://huggingface.co/cyankiwi/Qwen3.5-27B-AWQ-4bit
DEFAULT_KV_PATH = "vocaela/KV-Ground-8B-BaseGuiOwl1.5-0315"
DEFAULT_QWEN_PATH = "cyankiwi/Qwen3.5-27B-AWQ-4bit"

# Zoom pipeline configuration:
#   CROP_RATIO: fraction of image to crop around step-1 prediction (0.5 = 50%)
#   Both models use the same 2-step zoom pipeline with greedy decoding.
#   The router selects the model with lower zoom consistency (step-2 prediction
#   closer to crop center = higher confidence).
CROP_RATIO = 0.5

SYSTEM_PROMPT = (
    "You are a helpful assistant. The user will give you an instruction, "
    "and you MUST left click on the corresponding UI element via tool call. "
    "If you are not sure about where to click, guess a most likely one.\n\n"
    "# Tools\n\n"
    "You may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>\n"
    '{"type": "function", "function": {"name": "computer_use", '
    '"description": "Use a mouse to interact with a computer.\\n'
    "* The screen's resolution is 1000x1000.\\n"
    "* Make sure to click any buttons, links, icons, etc with the cursor tip "
    "in the center of the element. \\n"
    '* You can only use the left_click action to interact with the computer.", '
    '"parameters": {"properties": {"action": {"description": '
    '"The action to perform. The available actions are:\\n'
    '* `left_click`: Click the left mouse button with coordinate (x, y).", '
    '"enum": ["left_click"], "type": "string"}, '
    '"coordinate": {"description": "(x, y): The x (pixels from the left edge) '
    "and y (pixels from the top edge) coordinates to move the mouse to. "
    'Required only by `action=left_click`.", "type": "array"}, '
    '"required": ["action"], "type": "object"}}}}\n'
    "</tools>\n\n"
    "For each function call, return a json object with function name and "
    "arguments within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n'
    "</tool_call>"
)


class ZoomConsistencyRouterModel:
    def __init__(self):
        self.models = {}
        self.processors = {}

    def load_model(self, model_name_or_path=None,
                   kv_path=None, qwen_path=None):
        """Load both models for the ensemble.

        Args:
            model_name_or_path: Unused (kept for compatibility with eval script).
            kv_path: Path or HF ID for the specialist model.
                     Defaults to KV_GROUND_PATH env var or vocaela/KV-Ground-8B-BaseGuiOwl1.5-0315.
            qwen_path: Path or HF ID for the generalist model.
                       Defaults to QWEN_PATH env var or cyankiwi/Qwen3.5-27B-AWQ-4bit.
        """
        from transformers import AutoModelForImageTextToText, AutoProcessor

        kv_path = kv_path or os.environ.get("KV_GROUND_PATH", DEFAULT_KV_PATH)
        qwen_path = qwen_path or os.environ.get("QWEN_PATH", DEFAULT_QWEN_PATH)

        for name, path in [("kv", kv_path), ("qwen", qwen_path)]:
            print(f"Loading {name} from {path}...", flush=True)
            processor = AutoProcessor.from_pretrained(
                path, min_pixels=65536, max_pixels=99_999_999,
            )
            model = AutoModelForImageTextToText.from_pretrained(
                path, device_map=DEVICE, torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
            model.eval()
            self.models[name] = model
            self.processors[name] = processor
            print(f"{name} loaded.", flush=True)

    def set_generation_config(self, **kwargs):
        self.gen_config = kwargs

    def _run_vlm(self, name, image, instruction):
        """Run a model. Returns (x, y) in 1000x1000 space or None."""
        model = self.models[name]
        processor = self.processors[name]

        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": instruction},
            ]},
        ]

        inputs = processor.apply_chat_template(
            messages, tokenize=True, return_tensors="pt",
            return_dict=True, add_generation_prompt=True,
        ).to(DEVICE)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=512, do_sample=False,
            )

        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        response = processor.tokenizer.decode(generated, skip_special_tokens=True)
        return self._parse_tool_call(response), response

    def _parse_tool_call(self, response):
        if not response:
            return None
        tc_match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", response, re.DOTALL)
        if tc_match:
            try:
                data = json.loads(tc_match.group(1))
                coord = data.get("arguments", {}).get("coordinate", [])
                if len(coord) >= 2:
                    return (float(coord[0]), float(coord[1]))
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
        coord_match = re.search(
            r'"coordinate"\s*:\s*\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]', response)
        if coord_match:
            return (float(coord_match.group(1)), float(coord_match.group(2)))
        return None

    def _compute_crop_box(self, abs_x, abs_y, img_w, img_h):
        crop_w = int(img_w * CROP_RATIO)
        crop_h = int(img_h * CROP_RATIO)
        cx, cy = int(abs_x), int(abs_y)
        x1 = max(0, cx - crop_w // 2)
        y1 = max(0, cy - crop_h // 2)
        x2 = min(img_w, x1 + crop_w)
        y2 = min(img_h, y1 + crop_h)
        if x2 - x1 < crop_w:
            x1 = max(0, x2 - crop_w)
        if y2 - y1 < crop_h:
            y1 = max(0, y2 - crop_h)
        return x1, y1, x2, y2

    def _zoom_consistency(self, step2):
        if step2 is None:
            return None
        dx = step2[0] - 500.0
        dy = step2[1] - 500.0
        return math.sqrt(dx * dx + dy * dy)

    def _predict_2step(self, name, image, instruction):
        """Run full 2-step zoom. Returns (final_point_normalized, consistency, raw_response)."""
        orig_w, orig_h = image.size
        step1, raw1 = self._run_vlm(name, image, instruction)
        if step1 is None:
            return None, None, raw1 or ""

        abs_x = step1[0] / 1000.0 * orig_w
        abs_y = step1[1] / 1000.0 * orig_h
        x1, y1, x2, y2 = self._compute_crop_box(abs_x, abs_y, orig_w, orig_h)
        cropped = image.crop((x1, y1, x2, y2)).resize((orig_w, orig_h), Image.LANCZOS)

        step2, raw2 = self._run_vlm(name, cropped, instruction)
        cropped.close()

        if step2 is None:
            final = [abs_x / orig_w, abs_y / orig_h]
            return [max(0, min(1, final[0])), max(0, min(1, final[1]))], None, raw1

        cons = self._zoom_consistency(step2)
        abs2_x = x1 + (step2[0] / 1000.0) * (x2 - x1)
        abs2_y = y1 + (step2[1] / 1000.0) * (y2 - y1)
        final = [max(0, min(1, abs2_x / orig_w)), max(0, min(1, abs2_y / orig_h))]
        return final, cons, raw2

    def ground_only_positive(self, instruction, image):
        """Main entry point for the official eval script."""
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        # Run both models
        kv_point, kv_cons, kv_raw = self._predict_2step("kv", image, instruction)
        qw_point, qw_cons, qw_raw = self._predict_2step("qwen", image, instruction)

        # Route by lower consistency
        if kv_cons is None and qw_cons is None:
            chosen_point = kv_point
            chosen_raw = "router:kv (both None cons) | " + (kv_raw or "")
        elif kv_cons is None:
            chosen_point = qw_point
            chosen_raw = "router:qwen (kv None cons) | " + (qw_raw or "")
        elif qw_cons is None:
            chosen_point = kv_point
            chosen_raw = "router:kv (qw None cons) | " + (kv_raw or "")
        elif kv_cons <= qw_cons:
            chosen_point = kv_point
            chosen_raw = "router:kv (cons %.1f vs %.1f) | %s" % (kv_cons, qw_cons, kv_raw or "")
        else:
            chosen_point = qw_point
            chosen_raw = "router:qwen (cons %.1f vs %.1f) | %s" % (qw_cons, kv_cons, qw_raw or "")

        return {
            "point": chosen_point,
            "raw_response": chosen_raw,
        }

    def ground_allow_negative(self, instruction, image):
        """For negative samples. We always predict positive (we don't handle negatives)."""
        result = self.ground_only_positive(instruction, image)
        return {
            "point": result["point"],
            "result": "positive",
            "raw_response": result["raw_response"],
        }
