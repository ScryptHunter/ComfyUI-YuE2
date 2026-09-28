"""Static regression checks for the maintained YuE2 example workflows."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

WORKFLOW_DIR = Path(__file__).resolve().parents[1] / "example_workflows"
WORKFLOW_FILES = (
    "YuE2_Native_Text_to_Song_Simple_LoRA.json",
    "YuE2_Native_Text_to_Song.json",
    "YuE2_Native_Reference_Remix.json",
    "YuE2_Reference_Remix.json",
    "YuE2_Text_to_Song_Simple_LoRA.json",
    "YuE2_Text_to_Song.json",
    "YuE2_All_Nodes_Showcase.json",
)


def read_workflow(filename):
    return json.loads((WORKFLOW_DIR / filename).read_text(encoding="utf-8-sig"))


def graph_nodes(graph):
    return {node["id"]: node for node in graph.get("nodes", [])}


def link_source(graph, link_id):
    if isinstance(graph.get("links", []), list) and graph["links"] and isinstance(graph["links"][0], list):
        link = next(link for link in graph["links"] if link[0] == link_id)
        return link[1], link[2]
    link = next(link for link in graph["links"] if link["id"] == link_id)
    return link["origin_id"], link["origin_slot"]


def link_target(graph, link_id):
    if graph.get("links") and isinstance(graph["links"][0], list):
        link = next(link for link in graph["links"] if link[0] == link_id)
        return link[3], link[4]
    link = next(link for link in graph["links"] if link["id"] == link_id)
    return link["target_id"], link["target_slot"]


def node_by_type(graph, node_type):
    found = [node for node in graph.get("nodes", []) if node.get("type") == node_type]
    assert len(found) == 1, f"expected one {node_type}, found {len(found)}"
    return found[0]


def assert_graph_links(graph, virtual_nodes=()):
    nodes = graph_nodes(graph)
    virtual_nodes = set(virtual_nodes)
    if graph.get("links") and isinstance(graph["links"][0], list):
        links = graph["links"]
        ids = [link[0] for link in links]
        assert len(ids) == len(set(ids)), "duplicate link ids"
        link_rows = [(lid, src, ss, dst, ds) for lid, src, ss, dst, ds, _ in links]
        maximum_link = max(ids, default=0)
        maximum_node = max((node_id for node_id in nodes if isinstance(node_id, int)), default=0)
        assert graph.get("last_link_id", 0) >= maximum_link
        assert graph.get("last_node_id", 0) >= maximum_node
    else:
        links = graph.get("links", [])
        ids = [link["id"] for link in links]
        assert len(ids) == len(set(ids)), "duplicate subgraph link ids"
        link_rows = [(l["id"], l["origin_id"], l["origin_slot"], l["target_id"], l["target_slot"]) for l in links]
        state = graph.get("state", {})
        assert state.get("lastLinkId", 0) >= max(ids, default=0)
        assert state.get("lastNodeId", 0) >= max((i for i in nodes if isinstance(i, int)), default=0)

    indexed = {row[0]: row for row in link_rows}
    for lid, src, source_slot, dst, target_slot in link_rows:
        assert src in nodes or src in virtual_nodes, f"link {lid}: missing source node {src}"
        assert dst in nodes or dst in virtual_nodes, f"link {lid}: missing target node {dst}"
        if src in nodes:
            assert 0 <= source_slot < len(nodes[src].get("outputs", [])), f"link {lid}: bad source slot"
        if dst in nodes:
            assert 0 <= target_slot < len(nodes[dst].get("inputs", [])), f"link {lid}: bad target slot"
            assert nodes[dst]["inputs"][target_slot].get("link") == lid, f"input cache mismatch for link {lid}"
        if src in nodes:
            assert lid in (nodes[src]["outputs"][source_slot].get("links") or []), f"output cache mismatch for link {lid}"

    for node in nodes.values():
        for slot, inp in enumerate(node.get("inputs", [])):
            lid = inp.get("link")
            if lid is not None:
                assert lid in indexed, f"{node['id']} input {slot} references missing link {lid}"
                assert indexed[lid][3] == node["id"] and indexed[lid][4] == slot
        for slot, output in enumerate(node.get("outputs", [])):
            for lid in output.get("links") or []:
                assert lid in indexed, f"{node['id']} output {slot} references missing link {lid}"
                assert indexed[lid][1] == node["id"] and indexed[lid][2] == slot


def assert_acyclic(graph):
    nodes = graph_nodes(graph)
    edges = {node_id: [] for node_id in nodes}
    links = graph.get("links", [])
    for link in links:
        if isinstance(link, list):
            src, dst = link[1], link[3]
        else:
            src, dst = link["origin_id"], link["target_id"]
        if src in nodes and dst in nodes:
            edges[src].append(dst)
    visiting, visited = set(), set()

    def visit(node):
        assert node not in visiting, f"workflow dependency cycle at node {node}"
        if node in visited:
            return
        visiting.add(node)
        for target in edges[node]:
            visit(target)
        visiting.remove(node)
        visited.add(node)

    for node in edges:
        visit(node)


@pytest.mark.parametrize("filename", WORKFLOW_FILES)
def test_workflow_json_and_all_links_are_consistent(filename):
    data = read_workflow(filename)
    assert_graph_links(data)
    for subgraph in data.get("definitions", {}).get("subgraphs", []):
        assert_graph_links(subgraph, virtual_nodes=(-10, -20))
        nodes = graph_nodes(subgraph)
        for slot, item in enumerate(subgraph.get("inputs", [])):
            expected = sorted(link["id"] for link in subgraph["links"] if link["origin_id"] == -10 and link["origin_slot"] == slot)
            assert sorted(item.get("linkIds", [])) == expected
            assert expected, f"dead exported subgraph input {item.get('label', item['name'])}"
        for slot, item in enumerate(subgraph.get("outputs", [])):
            expected = sorted(link["id"] for link in subgraph["links"] if link["target_id"] == -20 and link["target_slot"] == slot)
            assert sorted(item.get("linkIds", [])) == expected
        external = next(node for node in data["nodes"] if node["type"] == subgraph["id"])
        assert len(external["inputs"]) == len(subgraph["inputs"])
        for outer_input, inner_input in zip(external["inputs"], subgraph["inputs"]):
            assert outer_input["name"] == inner_input["name"]
            assert outer_input["type"] == inner_input["type"]
        assert_acyclic(subgraph)
    assert_acyclic(data)


def assert_native_duration(subgraph):
    nodes = graph_nodes(subgraph)
    generate = node_by_type(subgraph, "YuE2NativeGenerateMusic")
    budget = node_by_type(subgraph, "YuE2ABCDurationBudget")
    final_abc = link_source(subgraph, generate["inputs"][3]["link"])
    budget_abc = link_source(subgraph, budget["inputs"][0]["link"])
    assert final_abc == budget_abc
    duration_link = next(link for link in subgraph["links"] if link["id"] == generate["inputs"][6]["link"])
    assert (duration_link["origin_id"], duration_link["origin_slot"]) == (budget["id"], 1)
    sampler = node_by_type(subgraph, "KSampler")
    zero = node_by_type(subgraph, "ConditioningZeroOut")
    assert link_source(subgraph, sampler["inputs"][1]["link"]) == (generate["id"], 2)
    assert link_source(subgraph, zero["inputs"][0]["link"]) == (generate["id"], 2)
    assert link_source(subgraph, sampler["inputs"][2]["link"]) == (zero["id"], 0)
    assert sampler["widgets_values_named"]["sampler_name"] == "dpm_2"
    assert sampler["widgets_values_named"]["scheduler"] == "sgm_uniform"


def test_native_simple_duration_negative_and_sampler_defaults():
    data = read_workflow("YuE2_Native_Text_to_Song_Simple_LoRA.json")
    nodes = graph_nodes(data)
    generate = node_by_type(data, "YuE2NativeGenerateMusic")
    budget = node_by_type(data, "YuE2ABCDurationBudget")
    assert link_source(data, generate["inputs"][3]["link"]) == link_source(data, budget["inputs"][0]["link"])
    assert link_source(data, generate["inputs"][6]["link"]) == (budget["id"], 1)
    sampler = node_by_type(data, "KSampler")
    zero = node_by_type(data, "ConditioningZeroOut")
    assert link_source(data, sampler["inputs"][1]["link"]) == (generate["id"], 2)
    assert link_source(data, zero["inputs"][0]["link"]) == (generate["id"], 2)
    assert link_source(data, sampler["inputs"][2]["link"]) == (zero["id"], 0)
    values = sampler["widgets_values_named"]
    assert (values["sampler_name"], values["scheduler"], values["steps"], values["cfg"], values["denoise"]) == ("dpm_2", "sgm_uniform", 32, 1, 1)
    generate_abc = node_by_type(data, "YuE2NativeGenerateABC")
    for slot in (1, 2):
        assert link_source(data, generate_abc["inputs"][slot]["link"]) == link_source(data, generate["inputs"][slot]["link"])
    for source in (generate_abc, generate, sampler):
        seed_slot = {"YuE2NativeGenerateABC": 3, "YuE2NativeGenerateMusic": 4, "KSampler": 4}[source["type"]]
        assert link_source(data, source["inputs"][seed_slot]["link"]) == link_source(data, generate_abc["inputs"][3]["link"])
    assert generate_abc["widgets_values_named"]["mode"] == generate["widgets_values_named"]["mode"]
    assert link_source(data, generate_abc["inputs"][4]["link"]) == link_source(data, generate["inputs"][5]["link"])
    assert nodes[link_source(data, generate_abc["inputs"][4]["link"])[0]]["type"] == "PrimitiveNode"


def test_native_planned_subgraphs_have_auto_duration_manual_vocal_and_universal_adapter():
    for filename in ("YuE2_Native_Text_to_Song.json", "YuE2_Native_Reference_Remix.json"):
        data = read_workflow(filename)
        subgraph = data["definitions"]["subgraphs"][0]
        assert_native_duration(subgraph)
        nodes = graph_nodes(subgraph)
        assert not any(n["type"] == "YuE2NativeLoraLoader" for n in subgraph["nodes"])
        assert "vocal_manual_semitones" in {i["name"] for i in subgraph["inputs"]}
        assert sum(node["type"] == "MarkdownNote" for node in data["nodes"]) == 1
        note = node_by_type(data, "MarkdownNote")["widgets_values"][0]
        assert "Vocal Manual Semitones" in note
        assert "pitch range only" in note
        assert "max_duration" not in note
        retarget = node_by_type(subgraph, "YuE2VocalRangeRetarget")
        manual_source = link_source(subgraph, retarget["inputs"][3]["link"])
        assert manual_source[0] == -10
        universal = node_by_type(subgraph, "YuE2NativeUniversalAdapterLoader")
        for slot in (1, 2, 3):
            assert universal["inputs"][slot]["link"] is not None
        adapter_input = next(i for i in subgraph["inputs"] if i["name"] == "adapter")
        assert adapter_input["label"] == "Adapter (LoRA / LoKr)"
    text = read_workflow("YuE2_Native_Text_to_Song.json")["definitions"]["subgraphs"][0]
    assert "value_2" not in {i["name"] for i in text["inputs"]}
    assert "max_duration" not in {i["name"] for i in text["inputs"]}
    assert not any(n["type"] == "PrimitiveBoolean" for n in text["nodes"])
    mode_input = next(i for i in text["inputs"] if i["name"] == "mode_1")
    assert mode_input["label"] == "Vocal Mode"
    outer = next(n for n in read_workflow("YuE2_Native_Text_to_Song.json")["nodes"] if n["type"] == text["id"])
    assert next(i for i in outer["inputs"] if i["name"] == "mode_1")["label"] == "Vocal Mode"


def test_native_notes_explain_style_lyrics_tempo_pairing_and_ceiling():
    for filename in ("YuE2_Native_Text_to_Song.json", "YuE2_Native_Reference_Remix.json"):
        data = read_workflow(filename)
        note = node_by_type(data, "MarkdownNote")
        text = note["widgets_values"][0]
        assert text == note["widgets_values_named"]["text"]
        for phrase in ("Style", "Lyrics", "Tempo Mode = keep", "Tempo Mode = override",
                       "AR", "NAR", "hard ceiling", "acoustic_adapter"):
            assert phrase in text
    text = read_workflow("YuE2_Native_Text_to_Song.json")
    subgraph = text["definitions"]["subgraphs"][0]
    outer = next(node for node in text["nodes"] if node["type"] == subgraph["id"])
    for inputs in (subgraph["inputs"], outer["inputs"]):
        bpm = next(item for item in inputs if item["name"] == "bpm")
        assert bpm["label"] == "BPM (used with Tempo Mode = override)"


def test_legacy_remix_duration_uses_sampling_settings_override_and_memory_preset():
    data = read_workflow("YuE2_Reference_Remix.json")
    nodes = graph_nodes(data)
    budget = node_by_type(data, "YuE2ABCDurationBudget")
    settings = node_by_type(data, "YuE2SamplingSettings")
    sampler = node_by_type(data, "YuE2Sampler")
    loader = node_by_type(data, "YuE2Loader")
    preset = node_by_type(data, "YuE2MemoryPreset")
    assert link_source(data, settings["inputs"][0]["link"]) == (budget["id"], 2)
    assert link_source(data, sampler["inputs"][2]["link"]) == (settings["id"], 0)
    assert sampler["inputs"][10]["link"] is None
    assert link_source(data, loader["inputs"][4]["link"]) == (preset["id"], 1)
    assert link_source(data, sampler["inputs"][16]["link"]) == (preset["id"], 3)


def test_legacy_simple_renders_the_budgeted_plan_and_shares_seed():
    data = read_workflow("YuE2_Text_to_Song_Simple_LoRA.json")
    nodes = graph_nodes(data)
    plan = node_by_type(data, "YuE2Plan")
    sampler = node_by_type(data, "YuE2Sampler")
    budget = node_by_type(data, "YuE2ABCDurationBudget")
    assert link_source(data, sampler["inputs"][1]["link"]) == (plan["id"], 1)
    assert link_source(data, budget["inputs"][0]["link"]) == (plan["id"], 1)
    assert sampler["inputs"][2]["link"] is None
    assert link_source(data, sampler["inputs"][10]["link"]) == (budget["id"], 2)
    assert sampler["widgets_values_named"]["vae_tile_frames"] == 0
    assert link_source(data, sampler["inputs"][6]["link"]) == link_source(data, plan["inputs"][6]["link"])
    assert link_source(data, sampler["inputs"][3]["link"]) == link_source(data, plan["inputs"][3]["link"])
    assert link_source(data, sampler["inputs"][4]["link"]) == link_source(data, plan["inputs"][4]["link"])
    assert sampler["widgets_values_named"]["cot"] == plan["widgets_values_named"]["cot"]
    assert link_source(data, sampler["inputs"][5]["link"]) == link_source(data, plan["inputs"][5]["link"])
    assert nodes[link_source(data, plan["inputs"][5]["link"])[0]]["type"] == "PrimitiveNode"


def test_legacy_staged_plan_budget_render_and_memory_preset_are_acyclic():
    data = read_workflow("YuE2_Text_to_Song.json")
    nodes = graph_nodes(data)
    plan = node_by_type(data, "YuE2Plan")
    budget = node_by_type(data, "YuE2ABCDurationBudget")
    settings = node_by_type(data, "YuE2SamplingSettings")
    render = node_by_type(data, "YuE2RenderPlan")
    loader = node_by_type(data, "YuE2Loader")
    preset = node_by_type(data, "YuE2MemoryPreset")
    assert link_source(data, render["inputs"][1]["link"]) == (plan["id"], 0)
    assert link_source(data, budget["inputs"][0]["link"]) == (plan["id"], 1)
    assert link_source(data, settings["inputs"][0]["link"]) == (budget["id"], 2)
    assert link_source(data, render["inputs"][2]["link"]) == (settings["id"], 0)
    assert link_source(data, loader["inputs"][4]["link"]) == (preset["id"], 1)
    assert link_source(data, render["inputs"][6]["link"]) == (preset["id"], 3)


def test_all_nodes_showcase_separates_planning_and_render_sampling_and_uses_duration_budget():
    data = read_workflow("YuE2_All_Nodes_Showcase.json")
    nodes = graph_nodes(data)
    settings = [n for n in data["nodes"] if n["type"] == "YuE2SamplingSettings"]
    assert len(settings) == 2
    planning = next(n for n in settings if n["id"] == 5)
    rendering = next(n for n in settings if n["id"] == 30)
    plan = node_by_type(data, "YuE2Plan")
    render = node_by_type(data, "YuE2RenderPlan")
    budget = node_by_type(data, "YuE2ABCDurationBudget")
    sampler = node_by_type(data, "YuE2Sampler")
    assert link_source(data, plan["inputs"][2]["link"]) == (planning["id"], 0)
    assert link_source(data, rendering["inputs"][0]["link"]) == (budget["id"], 2)
    assert link_source(data, render["inputs"][2]["link"]) == (rendering["id"], 0)
    assert link_source(data, budget["inputs"][0]["link"]) == link_source(data, render["inputs"][3]["link"])
    assert sampler["widgets_values_named"]["save_flac"] is False
    assert sampler["widgets_values_named"]["save_abc"] is True
    assert sampler["widgets_values_named"]["vae_decode"] == "auto"
    assert sampler["widgets_values_named"]["vae_tile_frames"] == 0
    assert sampler["widgets_values_named"]["abort_after_plan"] is False
    assert sampler["widgets_values_named"]["save_artifacts"] is False
