description: core ComfyUI concepts for answering questions about how it works
# ComfyUI basics

- A workflow is a graph of nodes. Data flows left to right through typed links.
- Loaders (CheckpointLoaderSimple, UNETLoader, CLIPLoader, VAELoader, LoraLoader)
  read model files from ComfyUI/models/<folder>.
- CLIPTextEncode turns text into CONDITIONING. The KSampler denoises a LATENT
  guided by positive/negative conditioning. VAEDecode turns the LATENT into an IMAGE.
- Only output nodes (SaveImage, PreviewImage, video combine nodes) cause work to run.
- Widgets are the editable values on a node (steps, cfg, seed, filenames).
  Inputs are the sockets on the left; outputs are on the right.
- seed with "randomize" changes every run; "fixed" repeats the same image.
- Bypass (mode 4) skips a node and passes its input through; mute (mode 2) disables it.
- Subgraphs group nodes into one; their insides are edited by opening them.
