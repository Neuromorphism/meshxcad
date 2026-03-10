"""Iterative detail transfer with convergence detection.

Repeatedly transfers detail from a target mesh onto a CAD approximation,
feeding each output back as the new input until the mean distance to the
target stops improving.

Optionally uses a vision LLM to provide guidance between iterations.
"""

import os
import numpy as np

from .alignment import find_correspondences
from .detail_transfer import transfer_mesh_detail_to_mesh
from .render import render_comparison, HAS_MPL
from . import vision_guide


def _measure_distance(result_verts, target_verts):
    """Mean nearest-neighbour distance from result to target."""
    _, _, dists = find_correspondences(result_verts, target_verts)
    return float(np.mean(dists))


def _render_iteration(plain_verts, plain_faces,
                       detail_verts, detail_faces,
                       result_verts, result_faces,
                       output_dir, iteration):
    """Render a comparison image for this iteration."""
    if not HAS_MPL:
        return None

    os.makedirs(output_dir, exist_ok=True)
    image_path = os.path.join(output_dir, f"iteration_{iteration:03d}.png")

    render_comparison(
        [(plain_verts, plain_faces),
         (detail_verts, detail_faces),
         (result_verts, result_faces)],
        [f"Input (iter {iteration})", "Target Mesh", f"Result (iter {iteration})"],
        image_path,
        title=f"Iteration {iteration}",
    )
    return image_path


def iterative_transfer(plain_verts, plain_faces,
                        detail_verts, detail_faces,
                        max_iterations=20,
                        min_improvement=0.001,
                        patience=3,
                        use_vision=False,
                        vision_model=None,
                        output_dir=None,
                        render=True,
                        callback=None):
    """Iteratively refine a CAD mesh toward a detail mesh.

    Each iteration runs the detail transfer, measures the mean distance
    to the target, and feeds the result back in.  Stops when:
      - improvement drops below *min_improvement* (fractional)
      - no improvement for *patience* consecutive iterations
      - *max_iterations* is reached
      - vision LLM confidence drops below 0.15

    Args:
        plain_verts:     (N,3) starting CAD vertices
        plain_faces:     (M,3) starting CAD faces
        detail_verts:    (P,3) target detail mesh vertices
        detail_faces:    (Q,3) target detail mesh faces
        max_iterations:  hard cap on iterations
        min_improvement: minimum fractional improvement to continue
        patience:        stop after this many iterations without improvement
        use_vision:      whether to query the vision LLM between iterations
        vision_model:    model name override for vision LLM
        output_dir:      directory for per-iteration comparison images
        render:          whether to render comparison images
        callback:        optional fn(iteration, distance, improved) called each step

    Returns:
        dict with:
            result_verts  — final displaced vertices
            result_faces  — face array (unchanged topology)
            iterations    — number of iterations run
            distances     — list of mean distances per iteration
            converged     — True if stopped by convergence, not max_iterations
            vision_log    — list of vision LLM responses (if used)
    """
    current_verts = np.array(plain_verts, dtype=np.float64)
    current_faces = np.array(plain_faces)
    original_verts = current_verts.copy()

    distances = []
    vision_log = []
    stale_count = 0
    best_distance = float("inf")
    best_verts = current_verts.copy()

    if output_dir is None and render:
        output_dir = "meshxcad_iterations"

    for iteration in range(1, max_iterations + 1):
        # --- Run transfer ---
        result_verts = transfer_mesh_detail_to_mesh(
            current_verts, current_faces, detail_verts, detail_faces
        )

        # --- Measure quality ---
        dist = _measure_distance(result_verts, detail_verts)
        distances.append(dist)

        # Track best
        if dist < best_distance:
            improvement = (best_distance - dist) / best_distance if best_distance < float("inf") else 1.0
            best_distance = dist
            best_verts = result_verts.copy()
            stale_count = 0
            improved = True
        else:
            improvement = 0.0
            stale_count += 1
            improved = False

        print(f"  Iteration {iteration}: mean distance = {dist:.6f}"
              f"  (best = {best_distance:.6f}, stale = {stale_count})")

        if callback:
            callback(iteration, dist, improved)

        # --- Render comparison ---
        image_path = None
        if render and output_dir:
            image_path = _render_iteration(
                current_verts, current_faces,
                detail_verts, detail_faces,
                result_verts, current_faces,
                output_dir, iteration,
            )

        # --- Vision LLM guidance ---
        vision_result = None
        if use_vision and image_path and vision_guide.is_available():
            prev_dist = distances[-2] if len(distances) > 1 else None
            vision_result = vision_guide.analyze_comparison(
                image_path, iteration, prev_dist, dist,
                model=vision_model,
            )
            if vision_result:
                vision_log.append(vision_result)
                print(f"  Vision LLM confidence: {vision_result['confidence']:.2f}")
                if vision_result.get("suggestions"):
                    for s in vision_result["suggestions"][:3]:
                        print(f"    - {s}")

                # If vision LLM is very confident no more improvement is possible
                if vision_result["confidence"] < 0.15:
                    print("  Vision LLM suggests convergence reached.")
                    break

        # --- Convergence checks ---
        if stale_count >= patience:
            print(f"  Converged: no improvement for {patience} iterations.")
            break

        if improved and improvement < min_improvement and iteration > 1:
            print(f"  Converged: improvement {improvement:.6f} < threshold {min_improvement}")
            break

        # Feed result back as input for next iteration
        current_verts = result_verts

    converged = iteration < max_iterations

    # Render final comparison against original
    if render and output_dir:
        final_path = os.path.join(output_dir, "final_comparison.png")
        if HAS_MPL:
            render_comparison(
                [(original_verts, current_faces),
                 (detail_verts, detail_faces),
                 (best_verts, current_faces)],
                ["Original Input", "Target Mesh", "Final Result"],
                final_path,
                title="Final Comparison",
            )

    return {
        "result_verts": best_verts,
        "result_faces": current_faces,
        "iterations": iteration,
        "distances": distances,
        "converged": converged,
        "vision_log": vision_log,
    }


def scratch_transfer(detail_verts, detail_faces,
                      max_iterations=20,
                      min_improvement=0.001,
                      patience=3,
                      use_vision=False,
                      vision_model=None,
                      output_dir=None,
                      render=True,
                      callback=None,
                      use_tracing=True,
                      template_name=None):
    """Start from scratch — reconstruct an initial CAD shape from the detail
    mesh, then iteratively refine it.

    Uses the tracing pipeline (segment → trace → reconstruct) by default,
    falling back to reconstruct_cad if tracing is disabled or fails.

    Args:
        detail_verts:    (P,3) target detail mesh vertices
        detail_faces:    (Q,3) target detail mesh faces
        use_tracing:     use tracing-based reconstruction (default True)
        template_name:   optional template name to guide tracing
        (remaining args same as iterative_transfer)

    Returns:
        Same as iterative_transfer, plus:
            initial_shape_type — the auto-detected shape type
            initial_quality    — quality of the initial reconstruction
            template_name      — template used (if any)
            n_segments         — number of segments (if tracing)
    """
    shape_type = "unknown"
    quality = 0.0
    cad_verts = None
    cad_faces = None
    tpl_name = None
    n_segments = 0

    # Try tracing first
    if use_tracing:
        try:
            from .tracing import trace_reconstruct, trace_reconstruct_with_template_search
            from .object_templates import get_template

            print("No input CAD provided — tracing reconstruction from mesh...")

            if template_name:
                tpl = get_template(template_name)
                if tpl:
                    recon = trace_reconstruct(detail_verts, detail_faces, template=tpl)
                    recon["template_name"] = template_name
                else:
                    print(f"  Warning: template '{template_name}' not found, auto-detecting...")
                    recon = trace_reconstruct_with_template_search(detail_verts, detail_faces)
            else:
                recon = trace_reconstruct_with_template_search(detail_verts, detail_faces)

            cad_verts = recon["cad_vertices"]
            cad_faces = recon["cad_faces"]
            quality = recon["quality"]
            n_segments = recon["n_segments"]
            tpl_name = recon.get("template_name")
            shape_type = f"traced({n_segments} segments)"

            print(f"  Traced: {n_segments} segments, "
                  f"{len(cad_verts)} verts, {len(cad_faces)} faces, "
                  f"quality={quality:.4f}")
            if tpl_name:
                print(f"  Template: {tpl_name}")
        except Exception as e:
            print(f"  Tracing failed ({e}), falling back to basic reconstruction...")
            cad_verts = None

    # Fallback to basic reconstruction
    if cad_verts is None:
        from .reconstruct import reconstruct_cad

        print("Reconstructing initial shape from mesh (basic mode)...")
        recon = reconstruct_cad(detail_verts, detail_faces)

        shape_type = recon["shape_type"]
        quality = recon["quality"]
        cad_verts = recon["cad_vertices"]
        cad_faces = recon["cad_faces"]

        print(f"  Reconstructed as '{shape_type}' "
              f"({len(cad_verts)} verts, {len(cad_faces)} faces, "
              f"quality={quality:.4f})")

    # Now run iterative refinement from the reconstructed shape
    result = iterative_transfer(
        cad_verts, cad_faces,
        detail_verts, detail_faces,
        max_iterations=max_iterations,
        min_improvement=min_improvement,
        patience=patience,
        use_vision=use_vision,
        vision_model=vision_model,
        output_dir=output_dir,
        render=render,
        callback=callback,
    )

    result["initial_shape_type"] = shape_type
    result["initial_quality"] = quality
    result["template_name"] = tpl_name
    result["n_segments"] = n_segments
    return result
