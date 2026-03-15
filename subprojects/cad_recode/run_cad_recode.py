#!/usr/bin/env python3
"""Local CAD-Recode inference from mesh input."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cadquery as cq
import numpy as np
import torch
import trimesh
from torch import nn
from transformers import AutoTokenizer, PreTrainedModel, Qwen2ForCausalLM, Qwen2Model

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from subprojects.common.mesh_preprocess import (
    load_mesh,
    normalize_mesh,
    sample_point_cloud,
    save_metadata,
    save_mesh,
    save_point_cloud_xyz,
)


class FourierPointEncoder(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        frequencies = 2.0 ** torch.arange(8, dtype=torch.float32)
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.projection = nn.Linear(51, hidden_size)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        encoded = (points.unsqueeze(-1) * self.frequencies).view(*points.shape[:-1], -1)
        encoded = torch.cat((points, encoded.sin(), encoded.cos()), dim=-1)
        return self.projection(encoded)


class CADRecode(Qwen2ForCausalLM):
    def __init__(self, config) -> None:
        PreTrainedModel.__init__(self, config)
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        torch.set_default_dtype(torch.float32)
        self.point_encoder = FourierPointEncoder(config.hidden_size)
        torch.set_default_dtype(torch.bfloat16)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        point_cloud=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if past_key_values is None or past_key_values.get_seq_length() == 0:
            assert inputs_embeds is None
            inputs_embeds = self.model.embed_tokens(input_ids)
            point_embeds = self.point_encoder(point_cloud).bfloat16()
            inputs_embeds[attention_mask == -1] = point_embeds.reshape(-1, point_embeds.shape[2])
            attention_mask[attention_mask == -1] = 1
            input_ids = None
            position_ids = None

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states).float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.CrossEntropyLoss()(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        from transformers.modeling_outputs import CausalLMOutputWithPast

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def load_model(model_id: str, device: str) -> tuple[AutoTokenizer, CADRecode]:
    attn_implementation = "flash_attention_2" if device.startswith("cuda") else None
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2-1.5B",
        pad_token="<|im_end|>",
        padding_side="left",
    )
    model = CADRecode.from_pretrained(
        model_id,
        torch_dtype="auto",
        attn_implementation=attn_implementation,
    ).eval().to(device)
    return tokenizer, model


def generate_code(
    model: CADRecode,
    tokenizer: AutoTokenizer,
    point_cloud: np.ndarray,
    max_new_tokens: int,
) -> str:
    input_ids = [tokenizer.pad_token_id] * len(point_cloud) + [tokenizer("<|im_start|>")["input_ids"][0]]
    attention_mask = [-1] * len(point_cloud) + [1]
    with torch.no_grad():
        batch_ids = model.generate(
            input_ids=torch.tensor(input_ids, device=model.device).unsqueeze(0),
            attention_mask=torch.tensor(attention_mask, device=model.device).unsqueeze(0),
            point_cloud=torch.tensor(point_cloud.astype(np.float32), device=model.device).unsqueeze(0),
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )
    decoded = tokenizer.batch_decode(batch_ids)[0]
    begin_raw = decoded.find("<|im_start|>")
    end = decoded.find("<|endoftext|>")
    if begin_raw < 0 or end < 0:
        raise RuntimeError("Failed to parse generated CadQuery code from model output")
    begin = begin_raw + len("<|im_start|>")
    return decoded[begin:end].strip()


def export_program(program: str, out_dir: Path) -> tuple[Path, Path]:
    namespace = {"cq": cq}
    exec(program, namespace)
    if "r" not in namespace:
        raise RuntimeError("Generated program did not define the expected CadQuery result variable 'r'")
    compound = namespace["r"].val()
    step_path = out_dir / "model.step"
    stl_path = out_dir / "model.stl"
    cq.exporters.export(compound, step_path)
    vertices, faces = compound.tessellate(0.001, 0.1)
    mesh = trimesh.Trimesh([(v.x, v.y, v.z) for v in vertices], faces)
    save_mesh(mesh, stl_path)
    return step_path, stl_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CAD-Recode from a mesh input")
    parser.add_argument("--mesh", required=True, help="Input mesh path")
    parser.add_argument("--out-dir", required=True, help="Directory for outputs")
    parser.add_argument("--model-id", default="filapro/cad-recode-v1.5")
    parser.add_argument("--n-points", type=int, default=256)
    parser.add_argument("--n-pre-points", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh = load_mesh(args.mesh)
    mesh, normalization = normalize_mesh(mesh, target="unit_symmetric")
    save_mesh(mesh, out_dir / "normalized_input.stl")

    points, _ = sample_point_cloud(mesh, n_points=args.n_points, n_pre_points=args.n_pre_points, seed=args.seed)
    save_point_cloud_xyz(points, out_dir / "point_cloud.xyz")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model = load_model(args.model_id, device)
    program = generate_code(model, tokenizer, points, args.max_new_tokens)

    program_path = out_dir / "program.py"
    program_path.write_text(program + "\n")
    step_path, stl_path = export_program(program, out_dir)

    save_metadata(
        out_dir / "run.json",
        method="cad-recode",
        model_id=args.model_id,
        device=device,
        input_mesh=args.mesh,
        normalization=normalization,
        outputs={
            "program": program_path,
            "step": step_path,
            "stl": stl_path,
        },
    )


if __name__ == "__main__":
    main()
