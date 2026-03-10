"""Drawing interpretation using Llama-3.2-11B-Vision.

Extracts structured geometry (DrawingSpec) from mechanical drawing images
via multi-stage prompting: scene → dimensions → features → reconciliation.
"""

import json
import re
import logging

import numpy as np
import torch
from PIL import Image

from .drawing_spec import DrawingSpec, Dimension, Feature, ViewSpec

logger = logging.getLogger(__name__)

# Default model
DEFAULT_MODEL = "meta-llama/Llama-3.2-11B-Vision-Instruct"


class DrawingInterpreter:
    """Extract structured geometry from mechanical drawings using a vision LLM."""

    def __init__(self, model_path=None, device="auto", quantize="auto"):
        """Load the vision model.

        Args:
            model_path: HuggingFace model ID or local path.
            device: "auto", "cuda", "cpu".
            quantize: "auto" (picks based on VRAM), "4bit", "8bit", "none".
        """
        self.model_path = model_path or DEFAULT_MODEL
        self.device = self._resolve_device(device)
        self.quantize = self._resolve_quantize(quantize)
        self.model = None
        self.processor = None
        self._load_model()

    def _resolve_device(self, device):
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _resolve_quantize(self, quantize):
        if quantize != "auto":
            return quantize
        if not torch.cuda.is_available():
            return "4bit"
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if mem_gb >= 24:
            return "none"
        elif mem_gb >= 12:
            return "8bit"
        return "4bit"

    def _load_model(self):
        """Load model and processor with appropriate quantisation."""
        from transformers import MllamaForConditionalGeneration, AutoProcessor

        logger.info("Loading %s (quantize=%s, device=%s)",
                     self.model_path, self.quantize, self.device)

        load_kwargs = {}
        if self.quantize == "4bit":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        elif self.quantize == "8bit":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            load_kwargs["torch_dtype"] = torch.float16

        if self.device == "cuda" and self.quantize == "none":
            load_kwargs["device_map"] = "auto"
        elif self.device == "cuda":
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["device_map"] = "cpu"

        self.model = MllamaForConditionalGeneration.from_pretrained(
            self.model_path, **load_kwargs
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        logger.info("Model loaded successfully")

    def _generate(self, image, prompt, max_new_tokens=2048) -> str:
        """Run a single prompt against the vision model."""
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ]}
        ]
        input_text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self.processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        # Decode only new tokens
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        return self.processor.decode(new_tokens, skip_special_tokens=True)

    def interpret_drawing(self, image_path, views_hint=None) -> DrawingSpec:
        """Full pipeline: image → structured DrawingSpec.

        Multi-stage prompting:
        1. Scene understanding — what object, what views
        2. Dimension extraction — measurements with units
        3. Feature extraction — per-view geometry as JSON
        4. Cross-view reconciliation — deterministic consistency check

        Args:
            image_path: path to drawing image (PNG, JPG).
            views_hint: optional list of known view types.

        Returns: DrawingSpec
        """
        image = Image.open(image_path).convert("RGB")

        # Stage 1: Scene understanding
        scene = self._prompt_scene(image)
        logger.info("Scene: %s", scene)

        if views_hint:
            scene["views"] = views_hint

        # Stage 2: Dimension extraction
        dimensions = self._prompt_dimensions(image, scene)
        logger.info("Dimensions: %d extracted", len(dimensions))

        # Stage 3: Feature extraction
        features_by_view = self._prompt_features(image, scene, dimensions)
        logger.info("Features: %s", {v: len(fs) for v, fs in features_by_view.items()})

        # Stage 4: Reconciliation (deterministic)
        return self._reconcile(scene, dimensions, features_by_view)

    def _prompt_scene(self, image) -> dict:
        """Stage 1: What is depicted and which views are present?"""
        prompt = (
            "This is a mechanical engineering drawing showing orthographic views "
            "of a part. Describe:\n"
            "1. What type of part is shown (e.g., shaft, flange, bracket, gear)?\n"
            "2. What views are present (front, side, top, section, isometric)?\n"
            "3. Is the part axially symmetric, bilaterally symmetric, or asymmetric?\n"
            "4. Approximate overall dimensions if visible.\n\n"
            'Reply ONLY as JSON: {"object_type": "...", "views": ["front", ...], '
            '"symmetry": "axial|bilateral|none", "description": "...", '
            '"overall_size": [width, height, depth]}'
        )
        response = self._generate(image, prompt)
        result = self._parse_json_response(response)
        # Defaults
        result.setdefault("object_type", "unknown")
        result.setdefault("views", ["front"])
        result.setdefault("symmetry", "none")
        result.setdefault("description", "")
        result.setdefault("overall_size", [1, 1, 1])
        return result

    def _prompt_dimensions(self, image, scene) -> list:
        """Stage 2: Extract all dimension annotations."""
        prompt = (
            f"This is a mechanical drawing of a {scene['object_type']}.\n"
            f"Views present: {', '.join(scene['views'])}.\n\n"
            "Extract EVERY dimension annotation shown in the drawing. "
            "For each dimension, provide:\n"
            "- value (number)\n"
            "- unit (mm or in)\n"
            "- measurement type (diameter, height, width, radius, depth, angle)\n"
            "- which feature it belongs to (body, bore, hole_1, flange_od, etc.)\n"
            "- which view it appears in\n\n"
            'Reply ONLY as JSON array: [{"value": ..., "unit": "mm", '
            '"measurement": "...", "feature": "...", "view": "..."}, ...]'
        )
        response = self._generate(image, prompt)
        raw = self._parse_json_response(response)
        if isinstance(raw, list):
            dims = raw
        elif isinstance(raw, dict) and "dimensions" in raw:
            dims = raw["dimensions"]
        else:
            dims = []

        result = []
        for d in dims:
            try:
                result.append(Dimension(
                    value=float(d.get("value", 0)),
                    unit=str(d.get("unit", "mm")),
                    measurement=str(d.get("measurement", "")),
                    feature=str(d.get("feature", "")),
                    view=str(d.get("view", "")),
                ))
            except (TypeError, ValueError):
                continue
        return result

    def _prompt_features(self, image, scene, dimensions) -> dict:
        """Stage 3: Extract geometric features per view."""
        dims_summary = "; ".join(
            f"{d.measurement}={d.value}{d.unit} ({d.feature})"
            for d in dimensions[:20]  # cap to avoid prompt overflow
        )
        view_list = ", ".join(scene["views"])

        prompt = (
            f"This is a mechanical drawing of a {scene['object_type']}.\n"
            f"Known dimensions: {dims_summary}\n"
            f"Views present: {view_list}\n\n"
            "For each view, describe every geometric feature visible:\n"
            "- feature_type: cylinder, hole, fillet, chamfer, flat, thread, slot, "
            "sphere, torus, counterbore\n"
            "- center_2d: [x, y] position as fraction 0-1 of view width/height\n"
            "- extent_2d: [w, h] bounding box as fraction of view\n"
            "- through: true/false (for holes)\n"
            "- dimensions: which dimension values apply to this feature\n\n"
            'Reply ONLY as JSON: {"views": [{"view_type": "front", '
            '"features": [{"feature_type": "...", "center_2d": [x,y], '
            '"extent_2d": [w,h], "through": false, '
            '"dimensions": [{"value": ..., "measurement": "..."}]}]}]}'
        )
        response = self._generate(image, prompt)
        raw = self._parse_json_response(response)

        result = {}
        views_data = []
        if isinstance(raw, dict):
            views_data = raw.get("views", [])
        elif isinstance(raw, list):
            views_data = raw

        for vd in views_data:
            if not isinstance(vd, dict):
                continue
            vtype = vd.get("view_type", "front")
            features = []
            for fd in vd.get("features", []):
                if not isinstance(fd, dict):
                    continue
                feat_dims = []
                for dd in fd.get("dimensions", []):
                    if isinstance(dd, dict):
                        try:
                            feat_dims.append(Dimension(
                                value=float(dd.get("value", 0)),
                                measurement=str(dd.get("measurement", "")),
                            ))
                        except (TypeError, ValueError):
                            continue

                features.append(Feature(
                    feature_type=str(fd.get("feature_type", "unknown")),
                    view=vtype,
                    center_2d=tuple(fd.get("center_2d", (0.5, 0.5))),
                    extent_2d=tuple(fd.get("extent_2d", (1.0, 1.0))),
                    dimensions=feat_dims,
                    through=bool(fd.get("through", False)),
                ))
            result[vtype] = features
        return result

    def _reconcile(self, scene, dimensions, features_by_view) -> DrawingSpec:
        """Stage 4: Cross-reference views, resolve conflicts, build DrawingSpec."""
        views = []
        for vtype, feats in features_by_view.items():
            views.append(ViewSpec(
                view_type=vtype,
                features=feats,
            ))
        # If no views from features, create from scene info
        if not views:
            for v in scene.get("views", ["front"]):
                views.append(ViewSpec(view_type=v))

        overall = scene.get("overall_size", [1, 1, 1])
        if len(overall) < 3:
            overall = list(overall) + [1] * (3 - len(overall))

        # Cross-reference: average close dimension values across views
        dims_merged = self._merge_dimensions(dimensions)

        return DrawingSpec(
            description=scene.get("description", ""),
            object_type=scene.get("object_type", "unknown"),
            views=views,
            dimensions=dims_merged,
            symmetry=scene.get("symmetry", "none"),
            overall_size=tuple(float(x) for x in overall[:3]),
        )

    def _merge_dimensions(self, dimensions: list) -> list:
        """Merge dimensions across views — average close values."""
        by_key = {}
        for d in dimensions:
            key = (d.measurement, d.feature)
            by_key.setdefault(key, []).append(d)

        merged = []
        for (meas, feat), dims in by_key.items():
            values = [d.value for d in dims]
            avg = float(np.mean(values))
            # Use first dim's metadata
            base = dims[0]
            merged.append(Dimension(
                value=avg,
                unit=base.unit,
                measurement=meas,
                feature=feat,
                view=base.view if len(dims) == 1 else "",
            ))
        return merged

    def _parse_json_response(self, response_text: str) -> dict | list:
        """Robustly parse JSON from LLM output."""
        # Strip markdown fences
        text = response_text.strip()
        text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'^```\s*$', '', text, flags=re.MULTILINE)
        text = text.strip()

        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object or array in the text
        for pattern in [r'\{[\s\S]*\}', r'\[[\s\S]*\]']:
            match = re.search(pattern, text)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    continue

        logger.warning("Failed to parse JSON from response: %s...", text[:200])
        return {}
