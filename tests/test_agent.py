"""Offline tests for EzeAi Comfy Agent. Stdlib unittest; no LLM, no ComfyUI server.

Several tests pin down failures that were actually observed during live testing
(marked "Regression"), so they cannot quietly come back.

Run:  python -m unittest discover -s tests -t .      (from the package folder)
"""

from __future__ import annotations

import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness  # noqa: E402,F401  (fixture catalogue + stubbed ComfyUI state)
from live_agent_snap import SNAP  # noqa: E402

from agent import comfy_bridge, config, graph_ops, memory, models, policy, providers  # noqa: E402
from agent import recipes, sessions, tools  # noqa: E402
from agent.loop import _CLAIM  # noqa: E402


def _gguf(path: Path, kv: dict) -> Path:
    """Write a minimal GGUF header with string/uint32 metadata."""
    with path.open("wb") as f:
        f.write(b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(kv)))
        for key, value in kv.items():
            kb = key.encode()
            f.write(struct.pack("<Q", len(kb)) + kb)
            if isinstance(value, int):
                f.write(struct.pack("<I", 4) + struct.pack("<I", value))
            else:
                vb = str(value).encode()
                f.write(struct.pack("<I", 8) + struct.pack("<Q", len(vb)) + vb)
    return path


class TestGGUF(unittest.TestCase):
    def test_reads_architecture_context_and_tool_template(self):
        with tempfile.TemporaryDirectory() as d:
            p = _gguf(Path(d, "m.gguf"), {"general.architecture": "qwen35", "general.name": "Q",
                                          "qwen35.context_length": 32768,
                                          "tokenizer.chat_template": "{% if tools %}...{% endif %}"})
            meta = models.read_gguf_metadata(p)
            self.assertEqual(meta["architecture"], "qwen35")
            self.assertEqual(meta["context_length"], 32768)
            self.assertTrue(meta["tools"])

    def test_no_tool_template_means_no_tools(self):
        with tempfile.TemporaryDirectory() as d:
            p = _gguf(Path(d, "m.gguf"), {"general.architecture": "llama",
                                          "tokenizer.chat_template": "[INST] {{ messages }} [/INST]"})
            self.assertFalse(models.read_gguf_metadata(p)["tools"])

    def test_projector_detected_by_metadata_not_just_name(self):
        """Regression: Qwen3.8-27B-mmproj-F16.gguf was listed as a chat model."""
        with tempfile.TemporaryDirectory() as d:
            p = _gguf(Path(d, "Qwen3.8-27B-projector-F16.gguf"), {"general.architecture": "clip"})
            self.assertTrue(models._is_projector(p))

    def test_projector_pairs_only_within_its_family(self):
        with tempfile.TemporaryDirectory() as d:
            gem = Path(d, "gemma-4-12b-it-Q5.gguf")
            qwen = Path(d, "Qwen3.5-9B-Q6.gguf")
            pg = Path(d, "mmproj-gemma-4-12b-it-F16.gguf")
            pq = Path(d, "mmproj-Qwen3.5-9B-BF16.gguf")
            for f in (gem, qwen, pg, pq):
                f.write_bytes(b"x")
            self.assertEqual(models._pair_projector(gem, [pg, pq]), pg)
            self.assertEqual(models._pair_projector(qwen, [pg, pq]), pq)


class TestProviders(unittest.TestCase):
    def test_tool_arguments_normalised_from_string_and_object(self):
        """llama-server returns a JSON string; Ollama returns an object."""
        self.assertEqual(providers._parse_args('{"folder":"loras"}'), {"folder": "loras"})
        self.assertEqual(providers._parse_args({"folder": "loras"}), {"folder": "loras"})

    def test_tool_calls_written_into_text_are_recovered(self):
        calls, rest = providers.parse_text_tool_calls(
            'Sure. <tool_call>{"name": "read_graph", "arguments": {}}</tool_call>', {"read_graph"})
        self.assertEqual(calls[0]["name"], "read_graph")
        self.assertEqual(rest, "Sure.")

    def test_unknown_text_tool_calls_are_ignored(self):
        calls, _ = providers.parse_text_tool_calls(
            '<tool_call>{"name": "format_disk", "arguments": {}}</tool_call>', {"read_graph"})
        self.assertEqual(calls, [])

    def test_remote_model_servers_are_refused(self):
        with self.assertRaises(providers.ProviderError):
            providers._check_local("http://203.0.113.9:11434/api/chat")
        providers._check_local("http://127.0.0.1:11434/api/chat")  # local is fine

    def test_repetition_is_detected_without_false_positives(self):
        """Regression: a 4B model looped and generated 8000+ tokens."""
        self.assertTrue(providers._repeating("The fox stands still. " * 60))
        prose = " ".join(f"line {i} about a lighthouse keeper and the sea at dusk" for i in range(30))
        self.assertFalse(providers._repeating(prose))
        nodes = "\n".join(f"- #{i} CLIPTextEncode encodes the prompt for the sampler" for i in range(40))
        self.assertFalse(providers._repeating(nodes))
        self.assertTrue(providers._repeating("ok " * 400))

    def test_every_request_has_a_token_cap(self):
        self.assertGreater(config.load_settings()["max_tokens"], 0)

    def test_images_convert_to_openai_parts(self):
        out = providers._to_openai([{"role": "user", "content": "see", "images": ["AAA"]}])
        self.assertEqual(out[0]["content"][1]["type"], "image_url")


class TestGraphOps(unittest.TestCase):
    def test_valid_plan_has_a_precise_diff(self):
        r = graph_ops.plan([{"op": "set_widget", "node": "3", "name": "steps", "value": 30}], SNAP)
        self.assertTrue(r["ok"])
        self.assertEqual(r["diff"], ["~ #3 KSampler.steps: 20 → 30"])

    def test_invalid_operations_are_rejected_with_reasons(self):
        r = graph_ops.plan([
            {"op": "add_node", "type": "KSamplerXL"},
            {"op": "set_widget", "node": "3", "name": "steps", "value": "lots"},
            {"op": "set_widget", "node": "3", "name": "sampler_name", "value": "eulr"},
            {"op": "connect", "from": {"node": "4", "output": "MODEL"}, "to": {"node": "3", "input": "latent_image"}},
            {"op": "remove_node", "node": "99"}], SNAP)
        self.assertFalse(r["ok"])
        text = " ".join(r["errors"])
        for needle in ("Did you mean: KSampler", "whole number", "not an option", "expects LATENT", "no node '99'"):
            self.assertIn(needle, text)

    def test_out_of_range_values_are_rejected(self):
        r = graph_ops.plan([{"op": "set_widget", "node": "3", "name": "denoise", "value": 7}], SNAP)
        self.assertFalse(r["ok"])

    def test_removing_existing_work_is_destructive(self):
        self.assertEqual(graph_ops.plan([{"op": "remove_node", "node": "9"}], SNAP)["risk"], "destructive")

    def test_noop_changes_are_dropped(self):
        """Regression: the model re-set five values to what they already were."""
        r = graph_ops.plan([{"op": "set_widget", "node": "3", "name": "cfg", "value": 8},
                            {"op": "set_widget", "node": "3", "name": "steps", "value": 20}], SNAP)
        self.assertEqual(r["diff"], [])
        self.assertEqual(r["ops"], [])

    def test_missing_inputs_and_output_node_are_warned(self):
        r = graph_ops.plan([{"op": "add_node", "ref": "k", "type": "KSampler"}], {"nodes": []})
        text = " ".join(r["warnings"])
        self.assertIn("required input 'model'", text)
        self.assertIn("no output node", text)

    def test_half_inserted_passthrough_is_warned(self):
        """Regression: LoRA CLIP output unused while encoders read the checkpoint."""
        r = graph_ops.plan([
            {"op": "add_node", "ref": "lora", "type": "LoraLoader", "widgets": {"lora_name": "detail_tweaker"}},
            {"op": "connect", "from": {"node": "4", "output": 0}, "to": {"node": "lora", "input": "model"}},
            {"op": "connect", "from": {"node": "4", "output": 1}, "to": {"node": "lora", "input": "clip"}},
            {"op": "connect", "from": {"node": "lora", "output": 0}, "to": {"node": "3", "input": "model"}}], SNAP)
        self.assertTrue(any("insert_after" in w for w in r["warnings"]))

    def test_insert_after_rewires_every_consumer(self):
        ops, problems = graph_ops.insert_after_ops(SNAP, [], "4", "LoraLoader", "lora",
                                                   {"lora_name": "detail_tweaker"})
        self.assertEqual(problems, [])
        r = graph_ops.plan(ops, SNAP)
        self.assertTrue(r["ok"])
        self.assertEqual(r["warnings"], [])
        joined = " ".join(r["diff"])
        for target in ("#3 KSampler.model", "#6 Positive.clip", "#7 Negative.clip"):
            self.assertIn(target, joined)

    def test_unique_partial_model_name_resolves(self):
        r = graph_ops.plan([{"op": "add_node", "ref": "l", "type": "LoraLoader",
                             "widgets": {"lora_name": "detail_tweaker"}}], SNAP)
        self.assertEqual(r["ops"][0]["widgets"]["lora_name"], "detail_tweaker.safetensors")

    def test_card_titles_come_from_the_diff(self):
        self.assertEqual(graph_ops.describe(["~ #3 KSampler.steps: 20 → 30"]), "Change steps")


class TestToolsAndModes(unittest.TestCase):
    def test_inspect_mode_has_no_change_tools(self):
        names = {t["function"]["name"] for t in tools.all_tools("inspect")}
        self.assertFalse(names & tools.DO_NAMES)
        self.assertFalse(any(n.startswith("create_") for n in names))
        self.assertIn("read_graph", names)

    def test_ask_and_auto_have_change_tools(self):
        for mode in ("ask", "auto"):
            names = {t["function"]["name"] for t in tools.all_tools(mode)}
            self.assertIn("set_widgets", names)
            self.assertIn("create_z_image_turbo", names)

    def test_edits_accumulate_into_one_card(self):
        ctx = tools.ToolContext(snapshot=SNAP)
        tools.run("set_widgets", {"node": "#3", "values": {"steps": 30}}, ctx)
        tools.run("set_widgets", {"node": 5, "values": '{"width": 768}'}, ctx)
        card = tools.finalize_draft(ctx, "x")
        self.assertEqual(len(ctx.proposals), 1)
        self.assertEqual(len(card["diff"]), 2)

    def test_read_graph_shows_pending_edits(self):
        """Regression: the model re-read an unchanged canvas five times."""
        ctx = tools.ToolContext(snapshot=SNAP)
        tools.run("set_widgets", {"node": "3", "values": {"steps": 30}}, ctx)
        self.assertIn("PENDING", tools.run("read_graph", {}, ctx))

    def test_recipe_picks_installed_sampler(self):
        """Regression: res_2s / beta57 only exist with the RES4LYF custom nodes."""
        ctx = tools.ToolContext(snapshot={"nodes": []})
        result = json.loads(tools.run("create_z_image_turbo", {"prompt": "a fox"}, ctx))
        self.assertTrue(result["ok"])
        ks = [o for o in ctx.proposals[0]["ops"] if o.get("type") == "KSampler"][0]["widgets"]
        self.assertIn(ks["sampler_name"], ("res_2s", "euler"))

    def test_save_workflow_existing_name_needs_confirmation(self):
        with tempfile.TemporaryDirectory() as d:
            original = comfy_bridge.workflows_dir
            comfy_bridge.workflows_dir = lambda: Path(d)
            try:
                Path(d, "Mine.json").write_text("{}", encoding="utf-8")
                ctx = tools.ToolContext(snapshot=SNAP)
                tools.run("save_workflow", {"name": "Mine"}, ctx)
                tools.run("save_workflow", {"name": "Fresh"}, ctx)
                self.assertEqual(ctx.proposals[0]["risk"], "replace")
                self.assertEqual(ctx.proposals[1]["risk"], "edit")
            finally:
                comfy_bridge.workflows_dir = original

    def test_queue_refuses_a_broken_graph(self):
        ctx = tools.ToolContext(snapshot={"nodes": [{"id": 1, "type": "KSampler", "inputs": []}]})
        self.assertIn("errors", tools.run("queue_run", {}, ctx))
        self.assertEqual(ctx.proposals, [])


class TestPolicy(unittest.TestCase):
    def test_policy_table(self):
        self.assertFalse(policy.auto_apply("edit", "ask"))
        self.assertTrue(policy.auto_apply("edit", "auto"))
        self.assertTrue(policy.auto_apply("queue", "auto"))
        self.assertFalse(policy.auto_apply("destructive", "auto"))
        self.assertFalse(policy.auto_apply("replace", "auto"))
        self.assertFalse(policy.auto_apply("edit", "inspect"))


class TestClaimGuard(unittest.TestCase):
    def test_action_claims_are_detected(self):
        """Regression: "I've saved the current canvas" with no tool called."""
        for text in ("I've saved the current canvas.", "Done. I set steps to 30.", "I added a LoRA."):
            self.assertTrue(_CLAIM.search(text), text)

    def test_honest_wording_is_not_flagged(self):
        for text in ("I've prepared a card that saves it.", "Your workflow saves an image.",
                     "The run was queued earlier by you."):
            self.assertFalse(_CLAIM.search(text), text)


class TestStorage(unittest.TestCase):
    def test_settings_ignore_unknown_keys_and_bad_modes(self):
        s = config.save_settings({"permission_mode": "godmode", "evil": 1, "max_rounds": 999})
        self.assertEqual(s["permission_mode"], "ask")
        self.assertNotIn("evil", s)
        self.assertEqual(s["max_rounds"], 20)

    def test_chat_ids_cannot_traverse(self):
        for bad in ("../x", "c_../../etc", ""):
            with self.assertRaises(ValueError):
                sessions.load(bad)

    def test_memory_remember_and_forget(self):
        memory.remember("I post vertical 9:16 videos")
        self.assertTrue(any("9:16" in n["note"] for n in memory.notes()))
        memory.remember("9:16", forget=True)
        self.assertFalse(any("9:16" in n["note"] for n in memory.notes()))

    def test_skills_load(self):
        self.assertIn("workflow-building", memory.skills())
        self.assertTrue(memory.load_skill("prompting")["ok"])
        self.assertFalse(memory.load_skill("nope")["ok"])


class TestApi(unittest.TestCase):
    def setUp(self):
        # api.py imports the package relatively; load it as part of a package
        import importlib.util

        pkg = HERE.parent
        spec = importlib.util.spec_from_file_location(
            "ezeai_agent_pkg", pkg / "__init__.py", submodule_search_locations=[str(pkg)])
        self.mod = None
        try:
            import types

            package = types.ModuleType("ezeai_agent_pkg")
            package.__path__ = [str(pkg)]
            sys.modules["ezeai_agent_pkg"] = package
            sys.modules.setdefault("ezeai_agent_pkg.agent", sys.modules["agent"])
            for name in ("config", "graph_ops", "memory", "models", "providers", "recipes", "runtime",
                         "sessions", "loop", "comfy_bridge", "tools", "policy"):
                sys.modules[f"ezeai_agent_pkg.agent.{name}"] = sys.modules[f"agent.{name}"]
            api_spec = importlib.util.spec_from_file_location("ezeai_agent_pkg.api", pkg / "api.py")
            self.mod = importlib.util.module_from_spec(api_spec)
            api_spec.loader.exec_module(self.mod)
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"api module not importable offline: {exc}")

    def test_save_workflow_rejects_traversal_and_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            original = comfy_bridge.workflows_dir
            comfy_bridge.workflows_dir = lambda: Path(d)
            try:
                g = {"nodes": [], "links": []}
                self.assertTrue(self.mod.save_workflow({"name": "A", "graph": g})["ok"])
                self.assertFalse(self.mod.save_workflow({"name": "A", "graph": g})["ok"])
                self.assertTrue(self.mod.save_workflow({"name": "A", "graph": g, "overwrite": True})["ok"])
                r = self.mod.save_workflow({"name": "../../evil", "graph": g})
                self.assertTrue(not r["ok"] or Path(r["path"]).parent == Path(d).resolve())
            finally:
                comfy_bridge.workflows_dir = original

    def test_revalidate_catches_a_changed_canvas(self):
        ctx = tools.ToolContext(snapshot=SNAP)
        tools.run("set_widgets", {"node": "3", "values": {"steps": 30}}, ctx)
        card = tools.finalize_draft(ctx, "x")
        self.assertTrue(self.mod.revalidate({"ops": card["ops"], "snapshot": SNAP})["ok"])
        gone = {"nodes": [n for n in SNAP["nodes"] if n["id"] != 3]}
        self.assertFalse(self.mod.revalidate({"ops": card["ops"], "snapshot": gone})["ok"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
