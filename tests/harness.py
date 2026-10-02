"""Offline harness: real LLM, fixture node catalogue, stubbed ComfyUI state."""
import json, os, sys, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("EZEAI_AGENT_DATA", tempfile.mkdtemp(prefix="ezeai_agent_"))
from agent import comfy_bridge as b

FIXTURE = json.load(open(os.path.join(ROOT, "tests", "object_info_fixture.json")))
b.set_source(lambda: FIXTURE)
MODELS = {
    "diffusion_models": [r"z-Image-Turbo\z_image_turbo_bf16.safetensors", "flux1-dev.safetensors"],
    "text_encoders": [r"qwe3 vl gguf\qwen_3_4b.safetensors", "t5xxl_fp16.safetensors"],
    "vae": ["ae.safetensors", "sdxl_vae.safetensors"],
    "checkpoints": [r"sdxl\juggernautXL_v9.safetensors", "dreamshaper_8.safetensors"],
    "loras": ["detail_tweaker.safetensors"],
    "upscale_models": ["4x-UltraSharp.pth"],
}
def list_models(folder, query="", limit=60):
    files = MODELS.get(folder)
    if files is None:
        return {"error": f"unknown model folder {folder!r}", "folders": sorted(MODELS)}
    hits = [f for f in files if query.lower() in f.lower()]
    return {"folder": folder, "total": len(files), "matched": len(hits), "files": hits[:limit], "truncated": False}
b.list_models = list_models
b.queue_status = lambda: {"running": [], "pending": [], "remaining": 0}
b.recent_runs = lambda limit=5: [{"prompt_id": "p1", "status": "error", "completed": False, "images": [],
    "error": {"node_id": "7", "node_type": "VAELoader", "exception": "ValueError",
              "message": "Value not in list: vae_name: 'sdxl_vae_fp16.safetensors' not in ['ae.safetensors', 'sdxl_vae.safetensors']"}}]
b.list_workflows = lambda query="", limit=30: ["Z-Image Lia.json", "flux2 Klein T2I 4b.json"]
b.system_stats = lambda: {"device": "cuda:0", "vram_total_gb": 17.2, "vram_free_gb": 15.8}

# Keep loader dropdowns consistent with the stubbed model folders: in live
# ComfyUI both come from folder_paths, so they always agree.
_LOADER_WIDGETS = {
    ("LoraLoader", "lora_name"): "loras", ("LoraLoaderModelOnly", "lora_name"): "loras",
    ("CheckpointLoaderSimple", "ckpt_name"): "checkpoints", ("VAELoader", "vae_name"): "vae",
    ("UNETLoader", "unet_name"): "diffusion_models", ("CLIPLoader", "clip_name"): "text_encoders",
    ("UpscaleModelLoader", "model_name"): "upscale_models",
}
for (node, widget), folder in _LOADER_WIDGETS.items():
    spec = FIXTURE.get(node, {}).get("input", {}).get("required", {}).get(widget)
    if spec is None:
        continue
    if isinstance(spec[0], list):
        spec[0] = list(MODELS[folder])
    elif spec[0] == "COMBO" and len(spec) > 1:
        spec[1]["options"] = list(MODELS[folder])
b.set_source(lambda: FIXTURE)
