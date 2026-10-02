"""Proven workflow recipes, emitted as ordinary graph operations.

A small local model wiring ten nodes from nothing is the least reliable thing
an agent can do. Recipes are deterministic builders for the common cases; they
produce the same operation language as the edit tools, so they go through the
same validation, diff preview and undo. The model's job shrinks to choosing a
recipe and filling in the creative fields, which it does well.

A recipe is offered only when its node types and model files are installed.
"""

from __future__ import annotations

from typing import Any, Callable

from . import comfy_bridge as bridge


def _have_models(folder: str, *needles: str) -> list[str]:
    try:
        files = bridge.list_models(folder, limit=500).get("files", [])
    except Exception:  # noqa: BLE001
        return []
    out = []
    for needle in needles:
        hit = next((f for f in files if needle.lower() in f.lower()), None)
        if hit:
            out.append(hit)
    return out


def _pick_option(node_type: str, widget: str, *preferred: str) -> str:
    """The first preferred value this install actually offers.

    Measured: the Z-Image recipe hard-coded res_2s / beta57, which only exist
    when the RES4LYF custom node pack is installed; elsewhere the workflow was
    rejected. Preferences now degrade to whatever the KSampler really lists.
    """
    schema = bridge.schema(node_type) or {}
    spec = next((i for i in schema.get("inputs", []) if i["name"] == widget), None)
    options = (spec or {}).get("options") or []
    for value in preferred:
        if value in options:
            return value
    return options[0] if options else preferred[-1]


def _nodes_present(*types: str) -> list[str]:
    info = bridge.object_info()
    return [t for t in types if t not in info]


# --- recipes ----------------------------------------------------------------


def z_image_turbo(p: dict[str, Any]) -> list[dict[str, Any]]:
    """Z-Image Turbo text-to-image, 8 steps. Settings from the user's own
    'Z-Image Lia' workflow, verified to render correctly on this PC."""
    unet = _have_models("diffusion_models", "z_image_turbo_bf16", "z-image-turbo")[0]
    clip = _have_models("text_encoders", "qwen_3_4b")[0]
    return [
        {"op": "add_node", "ref": "unet", "type": "UNETLoader",
         "widgets": {"unet_name": unet, "weight_dtype": "default"}},
        {"op": "add_node", "ref": "clip", "type": "CLIPLoader",
         "widgets": {"clip_name": clip, "type": "lumina2"}},
        {"op": "add_node", "ref": "vae", "type": "VAELoader", "widgets": {"vae_name": "ae.safetensors"}},
        {"op": "add_node", "ref": "shift", "type": "ModelSamplingAuraFlow", "widgets": {"shift": 3.0}},
        {"op": "add_node", "ref": "pos", "type": "CLIPTextEncode", "title": "Positive",
         "widgets": {"text": p.get("prompt", "")}},
        {"op": "add_node", "ref": "neg", "type": "CLIPTextEncode", "title": "Negative",
         "widgets": {"text": p.get("negative", "blurry, low quality, text, watermark")}},
        {"op": "add_node", "ref": "latent", "type": "EmptySD3LatentImage",
         "widgets": {"width": p.get("width", 896), "height": p.get("height", 1152), "batch_size": 1}},
        # res_2s / beta57 are the user's own tuned choice (RES4LYF); euler / simple
        # is ComfyUI's stock Z-Image setting and works everywhere.
        {"op": "add_node", "ref": "ks", "type": "KSampler",
         "widgets": {"seed": p.get("seed", 0), "steps": 8, "cfg": 1.0,
                     "sampler_name": _pick_option("KSampler", "sampler_name", "res_2s", "euler"),
                     "scheduler": _pick_option("KSampler", "scheduler", "beta57", "simple"),
                     "denoise": 1.0}},
        {"op": "add_node", "ref": "decode", "type": "VAEDecode"},
        {"op": "add_node", "ref": "save", "type": "SaveImage",
         "widgets": {"filename_prefix": p.get("prefix", "EzeAi_Agent/zimage")}},
        {"op": "connect", "from": {"node": "unet", "output": "MODEL"}, "to": {"node": "shift", "input": "model"}},
        {"op": "connect", "from": {"node": "clip", "output": "CLIP"}, "to": {"node": "pos", "input": "clip"}},
        {"op": "connect", "from": {"node": "clip", "output": "CLIP"}, "to": {"node": "neg", "input": "clip"}},
        {"op": "connect", "from": {"node": "shift", "output": "MODEL"}, "to": {"node": "ks", "input": "model"}},
        {"op": "connect", "from": {"node": "pos", "output": "CONDITIONING"}, "to": {"node": "ks", "input": "positive"}},
        {"op": "connect", "from": {"node": "neg", "output": "CONDITIONING"}, "to": {"node": "ks", "input": "negative"}},
        {"op": "connect", "from": {"node": "latent", "output": "LATENT"}, "to": {"node": "ks", "input": "latent_image"}},
        {"op": "connect", "from": {"node": "ks", "output": "LATENT"}, "to": {"node": "decode", "input": "samples"}},
        {"op": "connect", "from": {"node": "vae", "output": "VAE"}, "to": {"node": "decode", "input": "vae"}},
        {"op": "connect", "from": {"node": "decode", "output": "IMAGE"}, "to": {"node": "save", "input": "images"}},
    ]


def checkpoint_t2i(p: dict[str, Any]) -> list[dict[str, Any]]:
    """Classic checkpoint text-to-image (SD1.5 / SDXL style checkpoints)."""
    ckpt = p.get("checkpoint") or (bridge.list_models("checkpoints", limit=1).get("files") or [""])[0]
    return [
        {"op": "add_node", "ref": "ckpt", "type": "CheckpointLoaderSimple", "widgets": {"ckpt_name": ckpt}},
        {"op": "add_node", "ref": "pos", "type": "CLIPTextEncode", "title": "Positive",
         "widgets": {"text": p.get("prompt", "")}},
        {"op": "add_node", "ref": "neg", "type": "CLIPTextEncode", "title": "Negative",
         "widgets": {"text": p.get("negative", "blurry, low quality, watermark")}},
        {"op": "add_node", "ref": "latent", "type": "EmptyLatentImage",
         "widgets": {"width": p.get("width", 1024), "height": p.get("height", 1024), "batch_size": 1}},
        {"op": "add_node", "ref": "ks", "type": "KSampler",
         "widgets": {"seed": p.get("seed", 0), "steps": p.get("steps", 25), "cfg": p.get("cfg", 6.0),
                     "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0}},
        {"op": "add_node", "ref": "decode", "type": "VAEDecode"},
        {"op": "add_node", "ref": "save", "type": "SaveImage",
         "widgets": {"filename_prefix": p.get("prefix", "EzeAi_Agent/t2i")}},
        {"op": "connect", "from": {"node": "ckpt", "output": "MODEL"}, "to": {"node": "ks", "input": "model"}},
        {"op": "connect", "from": {"node": "ckpt", "output": "CLIP"}, "to": {"node": "pos", "input": "clip"}},
        {"op": "connect", "from": {"node": "ckpt", "output": "CLIP"}, "to": {"node": "neg", "input": "clip"}},
        {"op": "connect", "from": {"node": "pos", "output": "CONDITIONING"}, "to": {"node": "ks", "input": "positive"}},
        {"op": "connect", "from": {"node": "neg", "output": "CONDITIONING"}, "to": {"node": "ks", "input": "negative"}},
        {"op": "connect", "from": {"node": "latent", "output": "LATENT"}, "to": {"node": "ks", "input": "latent_image"}},
        {"op": "connect", "from": {"node": "ks", "output": "LATENT"}, "to": {"node": "decode", "input": "samples"}},
        {"op": "connect", "from": {"node": "ckpt", "output": "VAE"}, "to": {"node": "decode", "input": "vae"}},
        {"op": "connect", "from": {"node": "decode", "output": "IMAGE"}, "to": {"node": "save", "input": "images"}},
    ]


def upscale_image(p: dict[str, Any]) -> list[dict[str, Any]]:
    """Upscale an image with an installed ESRGAN-style upscale model."""
    model = p.get("upscale_model") or (bridge.list_models("upscale_models", limit=1).get("files") or [""])[0]
    return [
        {"op": "add_node", "ref": "img", "type": "LoadImage", "widgets": ({"image": p["image"]} if p.get("image") else {})},
        {"op": "add_node", "ref": "um", "type": "UpscaleModelLoader", "widgets": {"model_name": model}},
        {"op": "add_node", "ref": "up", "type": "ImageUpscaleWithModel"},
        {"op": "add_node", "ref": "save", "type": "SaveImage",
         "widgets": {"filename_prefix": p.get("prefix", "EzeAi_Agent/upscaled")}},
        {"op": "connect", "from": {"node": "um", "output": "UPSCALE_MODEL"}, "to": {"node": "up", "input": "upscale_model"}},
        {"op": "connect", "from": {"node": "img", "output": "IMAGE"}, "to": {"node": "up", "input": "image"}},
        {"op": "connect", "from": {"node": "up", "output": "IMAGE"}, "to": {"node": "save", "input": "images"}},
    ]


RECIPES: dict[str, dict[str, Any]] = {
    "z_image_turbo": {
        "label": "Z-Image Turbo text-to-image (8 steps, photoreal)",
        "build": z_image_turbo,
        "nodes": ("UNETLoader", "CLIPLoader", "VAELoader", "ModelSamplingAuraFlow", "CLIPTextEncode",
                  "EmptySD3LatentImage", "KSampler", "VAEDecode", "SaveImage"),
        "models": [("diffusion_models", "z_image_turbo"), ("text_encoders", "qwen_3_4b"), ("vae", "ae.safetensors")],
        "params": "prompt, negative, width, height, seed",
        "schema": {
            "prompt": {"type": "string", "description": "the complete image description, 40-120 words"},
            "negative": {"type": "string", "description": "short list of things to avoid"},
            "width": {"type": "integer", "description": "default 896"},
            "height": {"type": "integer", "description": "default 1152"},
            "seed": {"type": "integer"}},
        "required": ["prompt"],
        "tool": "create_z_image_turbo",
        "tool_description": "Build a complete Z-Image Turbo text-to-image workflow (photoreal, 8 steps, "
                            "proven on this PC). Use for any new photo / image request.",
    },
    "checkpoint_t2i": {
        "label": "Checkpoint text-to-image (SD1.5 / SDXL)",
        "build": checkpoint_t2i,
        "nodes": ("CheckpointLoaderSimple", "CLIPTextEncode", "EmptyLatentImage", "KSampler",
                  "VAEDecode", "SaveImage"),
        "models": [("checkpoints", "")],
        "params": "prompt, negative, checkpoint, width, height, steps, cfg, seed",
        "schema": {
            "prompt": {"type": "string"}, "negative": {"type": "string"},
            "checkpoint": {"type": "string", "description": "exact filename from list_models('checkpoints')"},
            "width": {"type": "integer"}, "height": {"type": "integer"},
            "steps": {"type": "integer"}, "cfg": {"type": "number"}, "seed": {"type": "integer"}},
        "required": ["prompt"],
        "tool": "create_checkpoint_t2i",
        "tool_description": "Build a classic checkpoint (SD1.5 / SDXL) text-to-image workflow.",
    },
    "upscale_image": {
        "label": "Upscale an image with an upscale model",
        "build": upscale_image,
        "nodes": ("LoadImage", "UpscaleModelLoader", "ImageUpscaleWithModel", "SaveImage"),
        "models": [("upscale_models", "")],
        "params": "image, upscale_model",
        "schema": {
            "image": {"type": "string", "description": "input image filename, optional"},
            "upscale_model": {"type": "string", "description": "exact filename from list_models('upscale_models')"}},
        "required": [],
        "tool": "create_upscale_image",
        "tool_description": "Build an image upscaling workflow with an installed upscale model.",
    },
}


def available() -> dict[str, dict[str, Any]]:
    out = {}
    for key, recipe in RECIPES.items():
        if _nodes_present(*recipe["nodes"]):
            continue
        ok = True
        for folder, needle in recipe["models"]:
            try:
                files = bridge.list_models(folder, limit=500).get("files", [])
            except Exception:  # noqa: BLE001
                files = []
            if not files or (needle and not any(needle.lower() in f.lower() for f in files)):
                ok = False
        if ok:
            out[key] = {"label": recipe["label"], "params": recipe["params"], "tool": recipe["tool"]}
    return out


def build(name: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    recipe = RECIPES.get(name)
    if recipe is None:
        raise ValueError(f"unknown recipe {name!r}; available: {', '.join(available()) or 'none'}")
    if name not in available():
        raise ValueError(f"recipe {name!r} needs nodes or models that are not installed")
    return recipe["build"](params or {})


def tool_specs() -> list[dict[str, Any]]:
    """One concrete tool per AVAILABLE recipe.

    Measured: a 4B model given a generic build_workflow(recipe=<id>, params={})
    confused the recipe id with a node type and never produced a workflow. A
    named, typed tool per recipe removes that ambiguity.
    """
    specs = []
    for key in available():
        recipe = RECIPES[key]
        specs.append({"type": "function", "function": {
            "name": recipe["tool"], "description": recipe["tool_description"],
            "parameters": {"type": "object", "properties": recipe["schema"],
                           **({"required": recipe["required"]} if recipe["required"] else {})}}})
    return specs


def by_tool(tool_name: str) -> str | None:
    return next((k for k, r in RECIPES.items() if r.get("tool") == tool_name), None)
