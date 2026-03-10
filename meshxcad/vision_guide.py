"""Vision LLM guidance for iterative mesh-to-CAD refinement.

Uses an OpenAI-compatible vision endpoint to analyze comparison images
and provide structured feedback on how to improve the CAD reconstruction.

Configuration via environment variables:
    LOCAL_OPENAI_KEY  — API key for the vision LLM
    LOCAL_OPENAI_URL  — Base URL of the OpenAI-compatible endpoint
"""

import base64
import json
import os


def _get_client():
    """Create an OpenAI-compatible client from environment variables.

    Returns (client, model_name) or (None, None) if not configured.
    """
    api_key = os.environ.get("LOCAL_OPENAI_KEY")
    base_url = os.environ.get("LOCAL_OPENAI_URL")

    if not api_key or not base_url:
        return None, None

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url)
        return client, None  # model resolved at call time
    except ImportError:
        return None, None


def is_available():
    """Check whether vision LLM guidance is configured and usable."""
    client, _ = _get_client()
    return client is not None


def _encode_image(image_path):
    """Base64-encode an image file for the vision API."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def analyze_comparison(image_path, iteration, previous_distance,
                       current_distance, model=None):
    """Ask the vision LLM to analyze a comparison image and suggest improvements.

    Args:
        image_path: path to the 3-panel comparison PNG
        iteration: current iteration number
        previous_distance: mean distance from previous iteration (or None)
        current_distance: mean distance for current iteration
        model: model name to use (auto-detected if None)

    Returns:
        dict with:
            suggestions  — list of text suggestions
            focus_areas  — list of regions to focus on
            confidence   — 0-1 confidence that improvements are still possible
            raw_response — full LLM response text
        or None if vision LLM is not available
    """
    client, _ = _get_client()
    if client is None:
        return None

    if model is None:
        model = os.environ.get("LOCAL_OPENAI_MODEL", "gpt-4o")

    b64_image = _encode_image(image_path)

    progress_info = f"Iteration {iteration}. "
    progress_info += f"Current mean distance to target: {current_distance:.6f}. "
    if previous_distance is not None:
        delta = previous_distance - current_distance
        progress_info += f"Improvement from last iteration: {delta:.6f}. "

    prompt = f"""You are analyzing a 3-panel comparison image from a mesh-to-CAD detail transfer process.

The three panels show (left to right):
1. **Plain CAD** — the current CAD approximation
2. **Detail Mesh** — the target mesh with fine detail
3. **Result** — the output after transferring detail from the mesh onto the CAD

{progress_info}

Please analyze the visual differences and respond in JSON format:
{{
    "suggestions": ["list of specific geometric improvements to make"],
    "focus_areas": ["list of regions/features that need the most work"],
    "confidence": 0.0 to 1.0 (how confident are you that further iteration will improve results),
    "scale_issues": true/false (are there obvious scale mismatches),
    "alignment_issues": true/false (are there obvious alignment/rotation problems),
    "missing_features": ["list of features visible in Detail Mesh but absent in Result"]
}}

Focus on actionable geometric feedback. Be honest about confidence — if the result
looks very close to the detail mesh, confidence should be low (little room to improve)."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64_image}",
                        },
                    },
                ],
            }],
            max_tokens=1024,
            temperature=0.2,
        )

        raw = response.choices[0].message.content.strip()

        # Parse JSON from response (handle markdown code blocks)
        json_text = raw
        if "```json" in json_text:
            json_text = json_text.split("```json")[1].split("```")[0].strip()
        elif "```" in json_text:
            json_text = json_text.split("```")[1].split("```")[0].strip()

        try:
            parsed = json.loads(json_text)
        except json.JSONDecodeError:
            parsed = {
                "suggestions": [],
                "focus_areas": [],
                "confidence": 0.5,
            }

        return {
            "suggestions": parsed.get("suggestions", []),
            "focus_areas": parsed.get("focus_areas", []),
            "confidence": float(parsed.get("confidence", 0.5)),
            "scale_issues": parsed.get("scale_issues", False),
            "alignment_issues": parsed.get("alignment_issues", False),
            "missing_features": parsed.get("missing_features", []),
            "raw_response": raw,
        }

    except Exception as e:
        print(f"  Vision LLM error: {e}")
        return None
