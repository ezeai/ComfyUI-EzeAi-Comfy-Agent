import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa
from agent import config, loop
from live_agent_snap import SNAP
config.save_settings({"agent_model": sys.argv[1], "permission_mode": "ask", "max_rounds": 4})
task = sys.argv[2]
snap = SNAP if task != "zimage" else {"nodes": []}
text = {"set_steps": "Set my sampler to 30 steps and cfg 6.",
        "zimage": "Make me a Z-Image photo of a red fox standing in fresh snow at sunrise."}[task]
def emit(ev):
    if ev["type"] == "tool" and ev["kind"] == "do":
        print("CALL", ev["name"], json.dumps(ev["args"])[:500]); print("  ->", ev["result"][:400])
loop.Agent(emit).run({"turns": []}, text, snap)
