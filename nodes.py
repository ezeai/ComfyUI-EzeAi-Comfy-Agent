"""Graph nodes: the local LLM inside a workflow.

EzeAi Comfy Agent (LLM) runs a local model as a step in a graph - write or
improve a prompt, describe an image, answer as JSON. By default it unloads the
model when it finishes, because the diffusion model that usually comes next
needs the VRAM.

The sidebar agent (which edits and runs workflows) is separate; this node never
touches the canvas.
"""

from __future__ import annotations

import base64
import io
import json
import re

from .agent import models, providers

CATEGORY = "EzeAi/Comfy Agent"

TASKS = {
    "prompt_writer": (
        "You write image-generation prompts. Turn the user's idea into ONE vivid, specific prompt "
        "of 40-120 words: subject first, then action, setting, light, framing and style. Plain "
        "descriptive sentences, no weight syntax, no quotation marks, no preamble. Reply with the "
        "prompt only."),
    "prompt_improver": (
        "You improve image-generation prompts. Keep the user's intent and subject, make it more "
        "specific and visual, fix contradictions. Reply with the improved prompt only."),
    "describe_image": (
        "Describe the image precisely for use as a generation prompt: subject, clothing, pose, "
        "setting, light, colours, framing, style. One paragraph, no preamble."),
    "chat": "You are a helpful, concise assistant.",
    "json": "Reply with valid JSON only, no prose, no code fences.",
}


def _model_choices() -> list[str]:
    try:
        refs = [m["ref"] for m in models.catalogue()["models"]]
    except Exception:  # noqa: BLE001
        refs = []
    return ["auto"] + refs


def _tensor_to_b64(image) -> list[str]:
    from PIL import Image
    import numpy as np

    out = []
    for frame in image[:4]:  # a batch: at most four frames go to the model
        array = (frame.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        pil = Image.fromarray(array)
        pil.thumbnail((1024, 1024))
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        out.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    return out


def _strip(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()
    return text


class EzeAi_ComfyAgentLLM:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "task": (list(TASKS), {"default": "prompt_writer"}),
                "instruction": ("STRING", {"multiline": True, "default": "",
                                           "tooltip": "Your idea, question or instruction"}),
                "model": (_model_choices(), {"default": "auto",
                                             "tooltip": "auto = the best installed model for this task"}),
                "unload_after": ("BOOLEAN", {"default": True,
                                             "tooltip": "Free the VRAM for the diffusion model that follows"}),
            },
            "optional": {
                "text_in": ("STRING", {"forceInput": True}),
                "image": ("IMAGE",),
                "system": ("STRING", {"multiline": True, "default": "",
                                      "tooltip": "Replaces the task's built-in instructions"}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF,
                                 "tooltip": "Change to get a different answer for the same input"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "info")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Runs a local language model (Ollama or a GGUF via llama.cpp) as a workflow step: "
                   "write or improve prompts, describe images, answer as JSON.")

    def run(self, task, instruction, model, unload_after, text_in="", image=None, system="",
            temperature=0.6, seed=0):
        need = ("vision",) if image is not None else ()
        chosen = models.resolve(None if model == "auto" else model, need=need) if model == "auto" \
            else models.resolve(model)
        user = "\n\n".join(p for p in (instruction.strip(), (text_in or "").strip()) if p)
        if not user and image is None:
            raise ValueError("Give the node an instruction, text_in or an image.")
        message = {"role": "user", "content": user or "Describe this image."}
        note = ""
        if image is not None:
            frames = _tensor_to_b64(image)
            if chosen["caps"].get("vision"):
                message["images"] = frames
            else:
                # a text-only brain gets the image through a vision model first
                vision = models.resolve(None, need=("vision",))
                described = providers.chat(vision, [{"role": "user", "content": TASKS["describe_image"],
                                                     "images": frames}],
                                           options={"keep_alive": 0, "think": False})
                message["content"] += "\n\n[The image, described by " + vision["name"] + "]\n" + described["content"]
                note = f"image routed through {vision['name']}"
        messages = [{"role": "system", "content": system.strip() or TASKS[task]}, message]
        options = {"temperature": temperature, "think": False,
                   "keep_alive": 0 if unload_after else "5m", "seed": seed}
        if task == "json":
            options["format"] = "json"
        reply = providers.chat(chosen, messages, options=options)
        if unload_after:
            providers.unload(chosen)
        text = _strip(reply["content"])
        if task == "json":
            try:
                text = json.dumps(json.loads(text), ensure_ascii=False, indent=1)
            except ValueError:
                pass  # leave the raw text visible rather than hiding a model mistake
        info = json.dumps({"model": chosen["name"], "provider": chosen["provider"],
                           "elapsed_s": reply["elapsed"], "tokens": reply.get("usage"),
                           "unloaded": bool(unload_after), "note": note}, ensure_ascii=False)
        return (text, info)


NODE_CLASS_MAPPINGS = {"EzeAi_ComfyAgentLLM": EzeAi_ComfyAgentLLM}
NODE_DISPLAY_NAME_MAPPINGS = {"EzeAi_ComfyAgentLLM": "EzeAi Comfy Agent (LLM)"}
