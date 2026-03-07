#!/usr/bin/env python3
"""Generate synthetic hourglass meshes and render them for visual inspection."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from meshxcad.hourglass_synthetic import (
    make_simple_hourglass_mesh,
    make_ornate_hourglass_mesh,
)
from meshxcad.render import render_mesh, render_comparison

output_dir = os.path.join(os.path.dirname(__file__), "..", "hourglass", "renders")
os.makedirs(output_dir, exist_ok=True)

print("Generating simple hourglass mesh...")
simple_v, simple_f = make_simple_hourglass_mesh(n_angular=48)
print(f"  Vertices: {len(simple_v)}, Faces: {len(simple_f)}")

print("Generating ornate hourglass mesh...")
ornate_v, ornate_f = make_ornate_hourglass_mesh(n_angular=48)
print(f"  Vertices: {len(ornate_v)}, Faces: {len(ornate_f)}")

print("\nRendering individual models...")
render_mesh(simple_v, simple_f,
            os.path.join(output_dir, "simple_hourglass.png"),
            title="Simple Hourglass (Plain)")

render_mesh(ornate_v, ornate_f,
            os.path.join(output_dir, "ornate_hourglass.png"),
            title="Ornate Hourglass (Featured)")

print("\nRendering comparison...")
render_comparison(
    [(simple_v, simple_f), (ornate_v, ornate_f)],
    ["Simple (Plain)", "Ornate (Featured)"],
    os.path.join(output_dir, "comparison_before_transfer.png"),
    title="Simple vs Ornate Hourglass",
)

print("\nDone! Check renders in:", output_dir)
