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
