"""Differentiable strategy selection via learned scoring functions.

Casts the strategy selection problem as continuous optimization:
instead of hard-coded heuristic thresholds, we learn soft scoring
weights that map mesh features to strategy probabilities.

Three optimizable selectors:

1. **SegmentationStrategySelector** — learns which segmentation strategy
   (skeleton, sdf, convexity, projection, normal_cluster) works best
   for a mesh with given geometric features (elongation, circularity,
   face count, etc.).

2. **FixerPrioritySelector** — learns priority weights over fixer
   functions, replacing the hand-tuned priority formula in the
   adversarial loop.

3. **DifferentiableRefiner** — replaces coordinate descent with
   gradient-based parameter refinement using autodiff through the
   mesh distance computation.

All selectors use PyTorch autograd when available, with a pure-numpy
fallback that uses finite-difference gradients.

Usage:
    from meshxcad.optim import (
        SegmentationStrategySelector,
        FixerPrioritySelector,
        DifferentiableRefiner,
    )

    # Segmentation
    selector = SegmentationStrategySelector()
    strategy = selector.select(vertices, faces)
    # After evaluating quality, update:
    selector.update(features, chosen_strategy, quality_score)

    # Fixer selection
    fixer_sel = FixerPrioritySelector(fixer_names)
    priorities = fixer_sel.score(diff_scores, tried_set, history)

    # Gradient-based refinement
    refiner = DifferentiableRefiner()
    new_params = refiner.refine(program, op_index, target_v, target_f)
"""

import copy
import math
import numpy as np

# ---------------------------------------------------------------------------
# Backend detection: prefer PyTorch autograd, fall back to numpy + finite diff
# ---------------------------------------------------------------------------

_HAS_TORCH = False
try:
    import torch
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    pass


def has_autograd():
    """Return True if PyTorch autograd is available."""
    return _HAS_TORCH


# ===========================================================================
# Feature extraction: mesh → feature vector
# ===========================================================================

def mesh_features(vertices, faces):
    """Extract geometric features from a mesh for strategy selection.

    Returns a dict of float features (all normalized to roughly [0, 1] or
    small positive range).
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces)
    n_verts = len(v)
    n_faces = len(f)

    center = v.mean(axis=0)
    centered = v - center

    # PCA eigenvalues
    cov = np.cov(centered.T) if n_verts > 3 else np.eye(3)
    eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    eigvals = np.maximum(eigvals, 1e-12)

    # Elongation: ratio of largest to middle eigenvalue
    elongation = float(np.sqrt(eigvals[0] / eigvals[1]))

    # Circularity: ratio of smallest to middle eigenvalue
    circularity = float(np.sqrt(eigvals[2] / eigvals[1]))

    # Log face count (normalized)
    log_faces = float(np.log1p(n_faces) / 12.0)  # ~1.0 at 160k faces

    # Face normal variance (measures surface complexity)
    if n_faces > 0:
        v0 = v[f[:, 0]]
        v1 = v[f[:, 1]]
        v2 = v[f[:, 2]]
        fn = np.cross(v1 - v0, v2 - v0)
        fn_norms = np.linalg.norm(fn, axis=1, keepdims=True)
        fn_norms = np.maximum(fn_norms, 1e-12)
        fn = fn / fn_norms
        normal_variance = float(fn.var())
    else:
        normal_variance = 0.0

    # Bounding box aspect ratio
    bbox = v.max(axis=0) - v.min(axis=0)
    bbox = np.maximum(bbox, 1e-12)
    bbox_sorted = np.sort(bbox)[::-1]
    bbox_aspect = float(bbox_sorted[0] / bbox_sorted[-1])

    # Compactness: volume / surface_area^(3/2) proxy
    bbox_vol = float(np.prod(bbox))
    bbox_sa = float(2 * (bbox[0]*bbox[1] + bbox[1]*bbox[2] + bbox[0]*bbox[2]))
    compactness = bbox_vol / max(bbox_sa ** 1.5, 1e-12)

    return {
        "elongation": elongation,
        "circularity": circularity,
        "log_faces": log_faces,
        "normal_variance": normal_variance,
        "bbox_aspect": min(bbox_aspect / 10.0, 1.0),  # cap at 1
        "compactness": min(compactness * 100, 1.0),
    }


FEATURE_NAMES = [
    "elongation", "circularity", "log_faces",
    "normal_variance", "bbox_aspect", "compactness",
]

STRATEGY_NAMES = ["skeleton", "sdf", "convexity", "projection", "normal_cluster"]


def _features_to_vector(feat_dict):
    """Convert feature dict to numpy array in canonical order."""
    return np.array([feat_dict[k] for k in FEATURE_NAMES], dtype=np.float64)


# ===========================================================================
# 1. SegmentationStrategySelector
# ===========================================================================

class SegmentationStrategySelector:
    """Differentiable selector: maps mesh features → strategy probabilities.

    Learns a weight matrix W (n_strategies × n_features) and bias b
    such that:
        logits = W @ features + b
        probs  = softmax(logits)
        strategy = argmax(probs)

    The weights are updated via gradient descent on a quality-based loss:
        loss = -quality * log(prob_of_chosen_strategy)
    i.e., REINFORCE-style policy gradient.
    """

    def __init__(self, lr=0.05):
        n_s = len(STRATEGY_NAMES)
        n_f = len(FEATURE_NAMES)
        self.lr = lr

        if _HAS_TORCH:
            self._W = torch.zeros(n_s, n_f, dtype=torch.float64,
                                  requires_grad=True)
            self._b = torch.zeros(n_s, dtype=torch.float64,
                                  requires_grad=True)
            # Initialize with heuristic prior
            self._init_heuristic_prior()
        else:
            self._W = np.zeros((n_s, n_f), dtype=np.float64)
            self._b = np.zeros(n_s, dtype=np.float64)
            self._init_heuristic_prior_np()

        self._history = []  # list of (features, strategy_idx, quality)
        self._record_experiences = True  # auto-record for federation

    def _init_heuristic_prior(self):
        """Warm-start weights to approximate the existing heuristic rules."""
        # skeleton: high elongation (>3) → elongation feature idx=0
        # sdf: high circularity (>0.85) → circularity feature idx=1
        # normal_cluster: many faces → log_faces feature idx=2
        # convexity: default
        with torch.no_grad():
            self._W[0, 0] = 2.0   # skeleton ← elongation
            self._W[1, 1] = 2.0   # sdf ← circularity
            self._W[4, 2] = 2.0   # normal_cluster ← log_faces
            self._b[2] = 0.5      # slight convexity bias (default)

    def _init_heuristic_prior_np(self):
        self._W[0, 0] = 2.0
        self._W[1, 1] = 2.0
        self._W[4, 2] = 2.0
        self._b[2] = 0.5

    def _compute_probs(self, feat_vec):
        """Compute strategy probabilities from feature vector."""
        if _HAS_TORCH:
            x = torch.tensor(feat_vec, dtype=torch.float64)
            logits = self._W @ x + self._b
            probs = F.softmax(logits, dim=0)
            return probs
        else:
            logits = self._W @ feat_vec + self._b
            # Stable softmax
            logits = logits - logits.max()
            exp_l = np.exp(logits)
            return exp_l / exp_l.sum()

    def select(self, vertices, faces):
        """Select the best segmentation strategy for the given mesh.

        Returns:
            str: strategy name
        """
        feats = mesh_features(vertices, faces)
        feat_vec = _features_to_vector(feats)
        probs = self._compute_probs(feat_vec)

        if _HAS_TORCH:
            idx = int(probs.argmax().item())
        else:
            idx = int(np.argmax(probs))

        return STRATEGY_NAMES[idx]

    def select_with_probs(self, vertices, faces):
        """Select strategy and return probability distribution.

        Returns:
            (str, dict): strategy name and {strategy: probability} dict
        """
        feats = mesh_features(vertices, faces)
        feat_vec = _features_to_vector(feats)
        probs = self._compute_probs(feat_vec)

        if _HAS_TORCH:
            probs_np = probs.detach().numpy()
        else:
            probs_np = probs

        prob_dict = {name: float(probs_np[i])
                     for i, name in enumerate(STRATEGY_NAMES)}
        idx = int(np.argmax(probs_np))
        return STRATEGY_NAMES[idx], prob_dict

    def update(self, vertices, faces, chosen_strategy, quality):
        """Update weights given observed quality of a strategy choice.

        Uses REINFORCE-style gradient: ∇loss = -quality * ∇log(π(a|s))

        Args:
            vertices, faces: mesh that was segmented
            chosen_strategy: which strategy was used
            quality: 0-1 quality score (higher = better)
        """
        if chosen_strategy not in STRATEGY_NAMES:
            return

        feats = mesh_features(vertices, faces)
        feat_vec = _features_to_vector(feats)
        strategy_idx = STRATEGY_NAMES.index(chosen_strategy)

        self._history.append((feat_vec.copy(), strategy_idx, quality))

        # Record for federated learning export
        if self._record_experiences:
            try:
                from .federation import record_segmentation_experience
                record_segmentation_experience(
                    feat_vec, chosen_strategy, quality)
            except ImportError:
                pass

        if _HAS_TORCH:
            self._update_torch(feat_vec, strategy_idx, quality)
        else:
            self._update_numpy(feat_vec, strategy_idx, quality)

    def _update_torch(self, feat_vec, strategy_idx, quality):
        """PyTorch autograd update."""
        x = torch.tensor(feat_vec, dtype=torch.float64)
        logits = self._W @ x + self._b
        log_probs = F.log_softmax(logits, dim=0)
        # REINFORCE loss: -quality * log(prob_of_chosen)
        loss = -quality * log_probs[strategy_idx]

        loss.backward()

        with torch.no_grad():
            if self._W.grad is not None:
                self._W -= self.lr * self._W.grad
                self._W.grad.zero_()
            if self._b.grad is not None:
                self._b -= self.lr * self._b.grad
                self._b.grad.zero_()

    def _update_numpy(self, feat_vec, strategy_idx, quality):
        """Numpy fallback: compute softmax gradient analytically."""
        logits = self._W @ feat_vec + self._b
        logits = logits - logits.max()
        exp_l = np.exp(logits)
        probs = exp_l / exp_l.sum()

        # Gradient of -log(probs[strategy_idx]) w.r.t. logits:
        # d/d_logit_j = probs[j]  for j != strategy_idx
        # d/d_logit_j = probs[j] - 1  for j == strategy_idx
        grad_logits = probs.copy()
        grad_logits[strategy_idx] -= 1.0
        grad_logits *= quality  # scale by reward

        # Chain rule: d_logit/d_W = outer(grad_logits, feat_vec)
        grad_W = np.outer(grad_logits, feat_vec)
        grad_b = grad_logits

        self._W -= self.lr * grad_W
        self._b -= self.lr * grad_b

    def get_weights(self):
        """Return current weight matrix and bias as numpy arrays."""
        if _HAS_TORCH:
            return self._W.detach().numpy().copy(), self._b.detach().numpy().copy()
        return self._W.copy(), self._b.copy()

    def set_weights(self, W, b):
        """Set weight matrix and bias from numpy arrays."""
        if _HAS_TORCH:
            with torch.no_grad():
                self._W.copy_(torch.tensor(W, dtype=torch.float64))
                self._b.copy_(torch.tensor(b, dtype=torch.float64))
        else:
            self._W = np.array(W, dtype=np.float64)
            self._b = np.array(b, dtype=np.float64)


# ===========================================================================
# 2. FixerPrioritySelector
# ===========================================================================

class FixerPrioritySelector:
    """Differentiable priority scoring for adversarial loop fixer selection.

    Learns per-fixer weights that combine with runtime signals
    (current diff score, freshness, historical success rate) to produce
    a priority score.  The weights are updated after each round based
    on whether the selected fixer actually improved the composite score.

    The scoring function is:
        priority_j = σ(w_score) * score_j
                   + σ(w_fresh) * freshness_j
                   + σ(w_hist)  * hist_rate_j
                   + bias_j

    where σ is sigmoid (keeps weights positive-bounded) and the w_*
    are learned via autograd.
    """

    def __init__(self, fixer_names, lr=0.02):
        self.fixer_names = list(fixer_names)
        n = len(self.fixer_names)
        self.lr = lr
        self._record_experiences = True  # auto-record for federation

        if _HAS_TORCH:
            # 3 global mixing weights + per-fixer bias
            self._w = torch.zeros(3, dtype=torch.float64, requires_grad=True)
            self._bias = torch.zeros(n, dtype=torch.float64, requires_grad=True)
            # Initialize: score weight high, freshness moderate, history moderate
            with torch.no_grad():
                self._w[0] = 1.0   # score weight
                self._w[1] = 0.5   # freshness weight
                self._w[2] = 0.5   # history weight
        else:
            self._w = np.array([1.0, 0.5, 0.5], dtype=np.float64)
            self._bias = np.zeros(n, dtype=np.float64)

    def score(self, diff_scores, tried_set, fixer_history):
        """Compute priority scores for all fixers.

        Args:
            diff_scores: dict {fixer_name: current_diff_score}
            tried_set: set of already-tried fixer names
            fixer_history: dict {fixer_name: {"avg_improvement": float}}

        Returns:
            list of (priority, fixer_name) sorted descending
        """
        priorities = []

        for i, name in enumerate(self.fixer_names):
            score = diff_scores.get(name, 0.0)
            freshness = 2.0 if name not in tried_set else 0.5
            hist_rate = fixer_history.get(name, {}).get(
                "avg_improvement", 50.0) / 100.0

            if _HAS_TORCH:
                sw = torch.sigmoid(self._w)
                p = (sw[0] * score
                     + sw[1] * freshness
                     + sw[2] * hist_rate
                     + self._bias[i])
                priorities.append((float(p.item()), name))
            else:
                sw = 1.0 / (1.0 + np.exp(-self._w))
                p = (sw[0] * score
                     + sw[1] * freshness
                     + sw[2] * hist_rate
                     + self._bias[i])
                priorities.append((float(p), name))

        priorities.sort(key=lambda x: x[0], reverse=True)
        return priorities

    def update(self, chosen_name, diff_scores, tried_set, fixer_history,
               improved, improvement_amount=0.0):
        """Update weights based on whether the chosen fixer helped.

        Uses a simple reward signal:
            reward = improvement_amount if improved, else -0.1
        Gradient pushes the chosen fixer's priority up (if rewarded)
        or down (if penalized).
        """
        if chosen_name not in self.fixer_names:
            return

        # Record for federated learning export
        if self._record_experiences:
            try:
                from .federation import record_fixer_experience
                record_fixer_experience(
                    self.fixer_names, chosen_name, diff_scores,
                    tried_set, fixer_history, improved,
                    improvement_amount)
            except ImportError:
                pass

        reward = improvement_amount if improved else -0.1
        idx = self.fixer_names.index(chosen_name)

        score = diff_scores.get(chosen_name, 0.0)
        freshness = 2.0 if chosen_name not in tried_set else 0.5
        hist_rate = fixer_history.get(chosen_name, {}).get(
            "avg_improvement", 50.0) / 100.0

        if _HAS_TORCH:
            self._update_torch(idx, score, freshness, hist_rate, reward)
        else:
            self._update_numpy(idx, score, freshness, hist_rate, reward)

    def _update_torch(self, idx, score, freshness, hist_rate, reward):
        sw = torch.sigmoid(self._w)
        priority = (sw[0] * score + sw[1] * freshness
                    + sw[2] * hist_rate + self._bias[idx])
        # Loss: -reward * priority (maximize reward-weighted priority)
        loss = -reward * priority
        loss.backward()

        with torch.no_grad():
            if self._w.grad is not None:
                self._w -= self.lr * self._w.grad
                self._w.grad.zero_()
            if self._bias.grad is not None:
                self._bias -= self.lr * self._bias.grad
                self._bias.grad.zero_()

    def _update_numpy(self, idx, score, freshness, hist_rate, reward):
        # Sigmoid and its derivative
        sw = 1.0 / (1.0 + np.exp(-self._w))
        dsw = sw * (1.0 - sw)

        # Gradient of priority w.r.t. w
        grad_w = np.array([
            dsw[0] * score,
            dsw[1] * freshness,
            dsw[2] * hist_rate,
        ])
        grad_w *= -reward

        grad_bias = np.zeros(len(self.fixer_names))
        grad_bias[idx] = -reward

        self._w -= self.lr * grad_w
        self._bias -= self.lr * grad_bias


# ===========================================================================
# 3. DifferentiableRefiner
# ===========================================================================

class DifferentiableRefiner:
    """Gradient-based parameter refinement using autodiff.

    Replaces coordinate descent in refine_operation() with gradient
    descent through a differentiable approximation of the mesh distance.

    The key challenge: mesh generation (boolean ops, tessellation) is
    not differentiable.  We solve this by:
    1. Computing the mesh at current params (forward pass, no grad)
    2. Building a differentiable proxy: for each CAD vertex, compute
       its distance to the nearest target vertex as a function of a
       continuous perturbation δ of the parameters
    3. Using autograd to compute ∂(total_distance)/∂δ
    4. Taking a gradient step on the original parameters

    When PyTorch is unavailable, falls back to finite-difference
    gradients (central differences) which is still better than
    coordinate descent for high-dimensional parameter spaces.
    """

    def __init__(self, lr=0.01, max_iter=50):
        self.lr = lr
        self.max_iter = max_iter

    def refine(self, program, op_index, target_v, target_f):
        """Refine operation parameters via gradient descent.

        Args:
            program: CadProgram instance
            op_index: index of operation to refine
            target_v: (N, 3) target vertices
            target_f: (M, 3) target faces

        Returns:
            float: final cost (lower is better)
        """
        if op_index < 0 or op_index >= len(program.operations):
            return float('inf')

        target_v = np.asarray(target_v, dtype=np.float64)
        target_f = np.asarray(target_f)
        op = program.operations[op_index]

        # Extract continuous parameters
        param_vec, param_keys, param_types = _extract_continuous_params(op)
        if len(param_vec) == 0:
            return program.total_cost(target_v, target_f)

        best_cost = program.total_cost(target_v, target_f)
        best_params = param_vec.copy()

        if _HAS_TORCH:
            final_cost = self._refine_torch(
                program, op, param_vec, param_keys, param_types,
                target_v, target_f, best_cost, best_params)
        else:
            final_cost = self._refine_fd(
                program, op, param_vec, param_keys, param_types,
                target_v, target_f, best_cost, best_params)

        return final_cost

    def _refine_torch(self, program, op, param_vec, param_keys, param_types,
                      target_v, target_f, best_cost, best_params):
        """Gradient descent using PyTorch autograd on a proxy loss."""
        from scipy.spatial import KDTree

        delta = torch.zeros(len(param_vec), dtype=torch.float64,
                            requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=self.lr)

        for iteration in range(self.max_iter):
            optimizer.zero_grad()

            # Apply delta to params
            current_params = param_vec + delta.detach().numpy()
            _set_continuous_params(op, current_params, param_keys, param_types)
            program.invalidate_cache()

            # Forward: evaluate mesh (non-differentiable)
            try:
                cad_v, cad_f = program.evaluate()
            except Exception:
                break
            if len(cad_v) == 0:
                break

            # Build differentiable proxy: perturb cad vertices by delta
            # Each cad vertex v_i depends linearly on params near current point
            # Approximate: v_i(θ+δ) ≈ v_i(θ) + J_i @ δ
            # We estimate J_i numerically for a few representative vertices
            cad_v_t = torch.tensor(cad_v, dtype=torch.float64)
            target_tree = KDTree(target_v)
            dists_np, idx_np = target_tree.query(cad_v)
            nearest_target = torch.tensor(
                target_v[idx_np], dtype=torch.float64)

            # Proxy loss: sum of squared distances to nearest target
            # Make it depend on delta through a linear correction
            diff = cad_v_t - nearest_target
            # Approximate gradient coupling: loss decreases when delta
            # moves params to reduce distances
            base_loss = (diff * diff).sum()

            # Compute finite-diff Jacobian for the loss w.r.t. delta
            eps = 1e-4
            grad_est = torch.zeros_like(delta)
            base_cost = float(np.sum(dists_np ** 2))

            for j in range(len(param_vec)):
                perturbed = param_vec.copy()
                perturbed[j] += eps
                _set_continuous_params(op, perturbed, param_keys, param_types)
                program.invalidate_cache()
                try:
                    pv, pf = program.evaluate()
                    if len(pv) > 0:
                        pd, _ = target_tree.query(pv)
                        perturbed_cost = float(np.sum(pd ** 2))
                        grad_est[j] = (perturbed_cost - base_cost) / eps
                except Exception:
                    pass

            # Restore current params
            _set_continuous_params(op, current_params, param_keys, param_types)
            program.invalidate_cache()

            # Use estimated gradient
            delta.grad = grad_est
            optimizer.step()

            # Evaluate actual cost
            new_params = param_vec + delta.detach().numpy()
            _set_continuous_params(op, new_params, param_keys, param_types)
            program.invalidate_cache()
            cost = program.total_cost(target_v, target_f)

            if cost < best_cost:
                best_cost = cost
                best_params = new_params.copy()
            elif iteration > 10 and cost > best_cost * 1.5:
                break  # diverging

        # Restore best
        _set_continuous_params(op, best_params, param_keys, param_types)
        program.invalidate_cache()
        return best_cost

    def _refine_fd(self, program, op, param_vec, param_keys, param_types,
                   target_v, target_f, best_cost, best_params):
        """Finite-difference gradient descent (numpy fallback)."""
        current = param_vec.copy()
        lr = self.lr

        for iteration in range(self.max_iter):
            # Central-difference gradient estimation
            grad = np.zeros_like(current)
            eps = max(1e-4, np.abs(current).mean() * 1e-3)

            for j in range(len(current)):
                # Forward
                p_fwd = current.copy()
                p_fwd[j] += eps
                _set_continuous_params(op, p_fwd, param_keys, param_types)
                program.invalidate_cache()
                cost_fwd = program.total_cost(target_v, target_f)

                # Backward
                p_bwd = current.copy()
                p_bwd[j] -= eps
                _set_continuous_params(op, p_bwd, param_keys, param_types)
                program.invalidate_cache()
                cost_bwd = program.total_cost(target_v, target_f)

                grad[j] = (cost_fwd - cost_bwd) / (2 * eps)

            # Gradient step with adaptive learning rate
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-10:
                break

            step = -lr * grad / max(grad_norm, 1.0)  # normalized step
            candidate = current + step
            _set_continuous_params(op, candidate, param_keys, param_types)
            program.invalidate_cache()
            cost = program.total_cost(target_v, target_f)

            if cost < best_cost:
                best_cost = cost
                best_params = candidate.copy()
                current = candidate
            else:
                lr *= 0.5
                if lr < 1e-6:
                    break

        # Restore best
        _set_continuous_params(op, best_params, param_keys, param_types)
        program.invalidate_cache()
        return best_cost


# ---------------------------------------------------------------------------
# Helpers for parameter extraction/injection
# ---------------------------------------------------------------------------

# Keys that represent discrete counts and should not be refined continuously
_SKIP_KEYS = {"divs", "h_divs", "subdivs", "cross_divs"}


def _extract_continuous_params(op):
    """Extract continuous parameters from a CadOp as a flat vector.

    Returns:
        param_vec: numpy array of parameter values
        param_keys: list of (key, sub_index_or_None) for reconstruction
        param_types: list of original types ('scalar' or 'vector_element')
    """
    vec = []
    keys = []
    types = []

    for key, val in op.params.items():
        if key in _SKIP_KEYS:
            continue
        if isinstance(val, (int, float)):
            vec.append(float(val))
            keys.append((key, None))
            types.append("scalar")
        elif isinstance(val, list) and all(isinstance(x, (int, float)) for x in val):
            for i, x in enumerate(val):
                vec.append(float(x))
                keys.append((key, i))
                types.append("vector_element")

    return np.array(vec, dtype=np.float64), keys, types


def _set_continuous_params(op, param_vec, param_keys, param_types):
    """Set operation parameters from a flat vector."""
    for idx, ((key, sub_idx), ptype) in enumerate(zip(param_keys, param_types)):
        val = param_vec[idx]
        if ptype == "scalar":
            if isinstance(op.params.get(key), int):
                op.params[key] = int(round(val))
            else:
                op.params[key] = float(val)
        elif ptype == "vector_element":
            if isinstance(op.params[key][sub_idx], int):
                op.params[key][sub_idx] = int(round(val))
            else:
                op.params[key][sub_idx] = float(val)
