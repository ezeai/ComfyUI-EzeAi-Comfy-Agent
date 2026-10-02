description: how to build or change a ComfyUI workflow reliably (recipes first, then edit_graph, then verify)
# Building workflows

1. LOOK first: read_graph to see what is on the canvas. Never assume it is empty.
2. Prefer a create_* tool. They build proven workflows (text-to-image, upscale)
   from the creative fields you fill in. Hand-wiring ten nodes is where mistakes happen.
3. To change an existing graph, make the smallest set of edits: set_widgets to
   change values, connect/disconnect to rewire, add_node only for what is missing.
   Reference existing nodes by their #id from read_graph. All edits in one turn
   become one card the user reviews.
4. Before add_node with an unfamiliar type, call node_info(type) for its exact
   input names, widget names and allowed values. Use search_nodes to find a type.
5. Model files: call list_models(folder) and use an exact filename. Never invent one.
6. After an edit, read the WARNINGS in the tool result. "required input not connected" or
   "no output node" means the graph will fail when queued - fix it in the same plan.
7. Queue only when the graph is complete. After a run, check_last_run, then
   look_at_output to judge the image before claiming it is good.

Wiring rules
- Outputs connect to inputs of the SAME type (MODEL->model, CLIP->clip,
  CONDITIONING->positive/negative, LATENT->latent_image/samples, VAE->vae, IMAGE->images).
- A sampler needs model, positive, negative and latent_image.
- Images become visible only through an output node (SaveImage, PreviewImage).
