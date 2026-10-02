import json, sys, time, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa
from agent import config, loop

SNAP = {"nodes": [
  {"id": 4, "type": "CheckpointLoaderSimple", "widgets": {"ckpt_name": "dreamshaper_8.safetensors"}, "inputs": []},
  {"id": 6, "type": "CLIPTextEncode", "title": "Positive", "widgets": {"text": "a castle"}, "inputs": [{"name": "clip", "from": [4, 1]}]},
  {"id": 7, "type": "CLIPTextEncode", "title": "Negative", "widgets": {"text": "blurry"}, "inputs": [{"name": "clip", "from": [4, 1]}]},
  {"id": 5, "type": "EmptyLatentImage", "widgets": {"width": 512, "height": 512, "batch_size": 1}, "inputs": []},
  {"id": 3, "type": "KSampler", "widgets": {"seed": 1, "steps": 20, "cfg": 8.0, "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0},
   "inputs": [{"name": "model", "from": [4, 0]}, {"name": "positive", "from": [6, 0]}, {"name": "negative", "from": [7, 0]}, {"name": "latent_image", "from": [5, 0]}]},
  {"id": 8, "type": "VAEDecode", "widgets": {}, "inputs": [{"name": "samples", "from": [3, 0]}, {"name": "vae", "from": [4, 2]}]},
  {"id": 9, "type": "SaveImage", "widgets": {"filename_prefix": "ComfyUI"}, "inputs": [{"name": "images", "from": [8, 0]}]},
]}
TASKS = [
  ("set_steps", "Set my sampler to 30 steps and cfg 6.", SNAP),
  ("zimage", "Make me a Z-Image photo of a red fox standing in fresh snow at sunrise.", {"nodes": []}),
  ("explain", "What does my current workflow do, in two sentences?", SNAP),
  ("error", "My last run failed. Why, and what's the fix?", SNAP),
]
model = sys.argv[1] if len(sys.argv) > 1 else ""
config.save_settings({"agent_model": model, "permission_mode": "ask"})
for name, text, snap in TASKS:
    tools_seen = []
    def emit(ev):
        if ev["type"] == "tool": tools_seen.append(ev["name"])
    t = time.time()
    try:
        r = loop.Agent(emit).run({"turns": []}, text, snap)
        props = [(p["kind"], p.get("risk"), (p.get("diff") or [p.get("summary")])[:4]) for p in r["proposals"]]
        print(f"\n[{name}] {time.time()-t:.1f}s rounds={r['rounds']} tools={tools_seen}")
        print("   proposals:", json.dumps(props, ensure_ascii=False)[:600])
        print("   answer:", r["text"][:260].replace("\n", " "))
    except Exception as e:
        print(f"\n[{name}] FAILED: {type(e).__name__}: {e}")
