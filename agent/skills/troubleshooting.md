description: diagnosing a failed ComfyUI run from its execution error
# Troubleshooting a failed run

Call check_last_run first. Read node_type and the exception message, then:

- "out of memory" / CUDA OOM: lower width/height or batch_size; free VRAM (the
  agent's own LLM is unloaded before queueing); use a smaller or fp8 model.
- "Value not in list" / "is not an option": a model filename or combo value is
  wrong. list_models for the right folder and set_widget to an exact name.
- "Required input is missing": an input socket is not connected. read_graph shows
  Problems; connect the missing link.
- Shape / size mismatch (e.g. "mat1 and mat2 shapes cannot be multiplied"): the
  model, text encoder and VAE are from different families (an SDXL VAE with a
  Flux model, a wrong CLIP type). Match the family.
- "No module named" / node shows red: a custom node failed to import. Say so; do
  not try to install anything - the user decides that.
- Black or noise-only images: wrong VAE, wrong sampler/scheduler for the model, or
  cfg far too high for a distilled/turbo model.

Explain the cause in one or two sentences, then propose the smallest fix.
