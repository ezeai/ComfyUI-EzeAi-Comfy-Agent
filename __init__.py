"""EzeAi Comfy Agent: a local-LLM agent for ComfyUI.

Registered defensively: if anything here fails to import, ComfyUI and every
other custom node keep loading, and the reason is printed with a traceback.
"""

WEB_DIRECTORY = "./web"
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    from .nodes import NODE_CLASS_MAPPINGS as _N, NODE_DISPLAY_NAME_MAPPINGS as _D

    NODE_CLASS_MAPPINGS.update(_N)
    NODE_DISPLAY_NAME_MAPPINGS.update(_D)
    print(f"[EzeAi Comfy Agent] {len(_N)} node(s) loaded")
except Exception as _err:  # pragma: no cover
    import traceback as _tb

    print(f"[EzeAi Comfy Agent] nodes NOT loaded: {type(_err).__name__}: {_err}")
    _tb.print_exc()

try:
    from server import PromptServer

    from .api import register_routes

    if register_routes(PromptServer.instance):
        print("[EzeAi Comfy Agent] agent API at /ezeai/agent/ - open the sidebar tab")
    else:
        print("[EzeAi Comfy Agent] agent API UNAVAILABLE (aiohttp missing): the sidebar will not work")
except Exception as _err:  # pragma: no cover
    import traceback as _tb

    print(f"[EzeAi Comfy Agent] agent API FAILED: {type(_err).__name__}: {_err}")
    _tb.print_exc()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
