import { app } from "../../../scripts/app.js";

const UNIVERSAL = new Set(["YuE2UniversalAdapterLoader", "YuE2NativeUniversalAdapterLoader"]);
const FL_ONLY = new Set(["YuE2LoraLoader", "YuE2NativeLoraLoader"]);
const REPOSITORIES = {
  trainer: "https://github.com/Starnodes2024/ComfyUI-YuE2-Trainer",
  fl: "https://github.com/filliptm/ComfyUI-FL-YuE2",
  hot: "https://github.com/scragnog/HOT-Step-CPP",
};

const styles = `
  position:fixed; inset:0; z-index:100000; display:flex; align-items:center;
  justify-content:center; padding:24px; background:rgba(4,8,14,.78);
  backdrop-filter:blur(4px); font:14px/1.5 system-ui,sans-serif; color:#e8edf5;
`;

function addText(parent, tag, text, css = {}) {
  const element = document.createElement(tag);
  element.textContent = text;
  Object.assign(element.style, css);
  parent.appendChild(element);
  return element;
}

function addLink(parent, label, href) {
  const a = document.createElement("a");
  a.textContent = label;
  a.href = href;
  a.target = "_blank";
  a.rel = "noopener noreferrer";
  a.style.color = "#76c8ff";
  a.style.textDecoration = "underline";
  parent.appendChild(a);
  return a;
}

function section(parent, title, body) {
  const block = document.createElement("section");
  block.style.margin = "0 0 18px";
  addText(block, "h3", title, { margin: "0 0 6px", color: "#fff", fontSize: "16px" });
  if (typeof body === "string") addText(block, "p", body, { margin: 0 });
  else block.appendChild(body);
  parent.appendChild(block);
}

function adapterTable() {
  const wrap = document.createElement("div");
  wrap.style.overflowX = "auto";
  const table = document.createElement("table");
  table.style.cssText = "width:100%;border-collapse:collapse;font-size:13px";
  const rows = [
    ["Adapter format", "Status"],
    ["FL-YuE2 · fl-yue2-lora-v1", "Supported"],
    ["Starnodes raw · yue2-lora-v1", "Supported"],
    ["Starnodes · comfyui-native-lora", "Supported"],
    ["Mothersuperior · YuE2 ComfyUI AR (format=pt)", "Supported · YuE2 signature required"],
    ["HOT-Step fused LoRA", "Supported"],
    ["HOT-Step native_split_v1", "Supported"],
    ["PEFT/HF LoRA", "Supported · adapter_config.json required"],
    ["HOT-Step native-split LoKr", "Supported"],
    ["Yue2 Studio acoustic", "Experimental / partial"],
    ["yue2-artist-ar-v1", "Unsupported · pinned decoder state"],
  ];
  for (const [index, row] of rows.entries()) {
    const tr = document.createElement("tr");
    for (const value of row) {
      const cell = document.createElement(index === 0 ? "th" : "td");
      cell.textContent = value;
      cell.style.cssText = "text-align:left;padding:6px 8px;border-bottom:1px solid #344052";
      tr.appendChild(cell);
    }
    table.appendChild(tr);
  }
  wrap.appendChild(table);
  return wrap;
}

function trainingLinks(parent) {
  const body = document.createElement("div");
  const items = [
    ["Starnodes2024/ComfyUI-YuE2-Trainer", REPOSITORIES.trainer,
      "ComfyUI-integrated training; the current trainer primarily targets NAR/acoustic adapters. Native comfyui-native-lora exports load directly- no manual conversion is needed."],
    ["ComfyUI-FL-YuE2", REPOSITORIES.fl,
      "Produces fl-yue2-lora-v1 AR/NAR pairs. The AR file may reference its paired acoustic/NAR adapter; Universal Loader supports the format, and the FL-only pair loader remains available."],
    ["HOT-Step-CPP", REPOSITORIES.hot,
      "Exports YuE2 LoRA and LoKr. Use fused or native_split_v1 LoRA for inference; for LoKr use the native_split_v1 inference export."],
  ];
  for (const [title, url, description] of items) {
    const p = document.createElement("p");
    p.style.margin = "0 0 10px";
    addLink(p, title, url);
    p.append(` - ${description}`);
    body.appendChild(p);
  }
  addText(body, "p", "Standard PEFT-style LoRA can load when targets unambiguously map to YuE2; keep adapter_config.json beside the weights. Do not assume an arbitrary LLM PEFT adapter is compatible.", { margin: "8px 0 0" });
  section(parent, "Where can I train YuE2 LoRAs?", body);
}

function openHelp(kind) {
  const isUniversal = kind === "universal";
  const overlay = document.createElement("div");
  overlay.setAttribute("role", "presentation");
  overlay.style.cssText = styles;
  const dialog = document.createElement("div");
  dialog.setAttribute("role", "dialog");
  dialog.setAttribute("aria-modal", "true");
  dialog.setAttribute("aria-label", isUniversal ? "YuE2 Universal Adapter Loader help" : "FL-YuE2 LoRA Pair Loader help");
  dialog.style.cssText = "width:min(940px,96vw);max-height:88vh;display:flex;flex-direction:column;background:#151b25;border:1px solid #465469;border-radius:12px;box-shadow:0 18px 60px #000a;overflow:hidden";
  const top = document.createElement("header");
  top.style.cssText = "display:flex;align-items:center;justify-content:space-between;padding:16px 20px;background:#1e2734;border-bottom:1px solid #344052";
  addText(top, "h2", isUniversal ? "YuE2 Universal Adapter Loader — Help" : "FL-YuE2 LoRA Pair Loader — Help", { margin: 0, fontSize: "20px", color: "#fff" });
  const close = document.createElement("button");
  close.type = "button"; close.textContent = "Close"; close.setAttribute("aria-label", "Close help");
  close.style.cssText = "border:1px solid #607086;border-radius:6px;padding:6px 12px;background:#293647;color:#fff;cursor:pointer";
  top.appendChild(close); dialog.appendChild(top);
  const body = document.createElement("main");
  body.style.cssText = "padding:18px 22px;overflow:auto";
  if (isUniversal) {
    section(body, "What this node does", "Automatically detects supported YuE2 adapter formats and applies patches to the matching AR and/or NAR branch. Supports LoRA and LoKr; the format does not need to be selected manually.");
    section(body, "Supported adapters", adapterTable());
    section(body, "Mothersuperior community AR family", "The raw split FP32, raw BF16, and ComfyUI fused variants are supported as AR-only adapters. They were checked against the three real files for all 28 layers and seven projection targets. Use cot=full for the planner/score path. ar_strength scales the adapter; nar_strength has no effect on these AR-only files.");
    section(body, "LoRA vs LoKr", "LoRA is a low-rank A/B adaptation: common, compact, and scaled by strength. LoKr is a Kronecker-factorized adapter with a different parameterization. This loader never turns LoKr into fake LoRA; Native materializes its correct dense delta. The supported HOT-Step inference export for LoKr is native_split_v1.");
    const advanced = document.createElement("details");
    addText(advanced, "summary", "Advanced: scaling and compatibility");
    addText(advanced, "p", "Intrinsic format scaling (for example alpha/rank) is applied before the independent AR/NAR runtime strengths. PEFT requires adapter_config.json; DoRA and fan-in/fan-out are rejected. HOT-Step fused LoKr checkpoint artifacts are not accepted directly- use their native_split_v1 inference export.", { margin: "6px 0 0" });
    section(body, "", advanced);
    section(body, "AR vs NAR", "AR primarily affects semantic/music-token generation: composition, structure, musical content, and high-level conditioning. NAR affects acoustic decoding: timbre, sound, production character, detail, and acoustic realization. The actual effect depends on what the adapter was trained to change.");
    section(body, "ar_strength", "Scales only AR adapter patches. It has no effect for a NAR-only adapter.");
    section(body, "nar_strength", "Scales only NAR adapter patches. It has no effect for an AR-only adapter.");
    section(body, "enabled", "When false, this node contributes no adapter patch and returns the clean input pipeline (or the other contributions already present in a chained stack).");
    section(body, "Stacking", "Chain Base → Adapter A → Adapter B to get Base + A + B. Changing the adapter selected inside one loader node replaces that node’s prior contribution; it does not accumulate stale selections. Additive adapters can be stacked; order remains part of workflow identity.");
    section(body, "Legacy vs Native", "Legacy uses the legacy HF/YuE2 runtime and request-scoped adapter hooks, which suits existing legacy workflows. Native applies patches through cloned ComfyUI ModelPatchers; the base patcher stays clean and Native pipeline workflows should use this backend. Neither backend is inherently higher quality- the choice is determined by the pipeline type.");
    trainingLinks(body);
  } else {
    section(body, "This node is FL-YuE2-specific- not universal", "Accepts only fl-yue2-lora-v1. It is intended for an FL-YuE2 AR/NAR pair and keeps its older workflow class ID for compatibility.");
    section(body, "Pairing and strengths", "The AR adapter may reference a paired NAR/acoustic adapter through acoustic_adapter metadata. Automatic pairing can find that file; nar_lora_override selects a manual FL NAR pair instead. ar_strength and nar_strength independently control the two branches.");
    section(body, "For other formats", "For Starnodes, comfyui-native-lora, PEFT, HOT-Step, LoKr, or community adapters, use YuE2 Universal Adapter Loader (LoRA / LoKr) with the matching Legacy or Native pipeline.");
    const bodyLink = document.createElement("p");
    addLink(bodyLink, "ComfyUI-FL-YuE2 training repository", REPOSITORIES.fl);
    section(body, "Training tool", bodyLink);
  }
  dialog.appendChild(body); overlay.appendChild(dialog); document.body.appendChild(overlay);
  const closePopup = () => { overlay.remove(); document.removeEventListener("keydown", onKey); };
  const onKey = (event) => { if (event.key === "Escape") closePopup(); };
  close.addEventListener("click", closePopup);
  overlay.addEventListener("click", event => { if (event.target === overlay) closePopup(); });
  dialog.addEventListener("click", event => event.stopPropagation());
  document.addEventListener("keydown", onKey);
  close.focus();
}

function helpKind(nodeData, nodeType) {
  const id = String(nodeData?.name || nodeType?.type || "");
  if (UNIVERSAL.has(id)) return "universal";
  if (FL_ONLY.has(id)) return "fl";
  return null;
}

function installHeaderHelp(nodeType, kind) {
  if (!nodeType?.prototype || nodeType.prototype.__yue2HelpInstalled) return;
  const proto = nodeType.prototype;
  proto.__yue2HelpInstalled = true;
  const priorDraw = proto.onDrawForeground;
  proto.onDrawForeground = function(ctx, ...args) {
    priorDraw?.call(this, ctx, ...args);
    if (!ctx || !this.size) return;
    const x = this.size[0] - 43;
    const y = 15;
    ctx.save();
    ctx.beginPath(); ctx.arc(x, y, 9, 0, Math.PI * 2);
    ctx.fillStyle = "#28384b"; ctx.fill();
    ctx.strokeStyle = "#a7c8e8"; ctx.lineWidth = 1; ctx.stroke();
    ctx.fillStyle = "#eaf5ff"; ctx.font = "bold 12px sans-serif";
    ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText("?", x, y + .5);
    ctx.restore();
  };
  const priorMouseDown = proto.onMouseDown;
  proto.onMouseDown = function(event, pos, canvas) {
    if (pos && this.size && this.pos) {
      const isGraph = pos[0] >= this.pos[0] && pos[0] <= this.pos[0] + this.size[0] && pos[1] >= this.pos[1] && pos[1] <= this.pos[1] + 30;
      const x = isGraph ? pos[0] - this.pos[0] : pos[0];
      const y = isGraph ? pos[1] - this.pos[1] : pos[1];
      // Keep a gap from the standard LiteGraph title controls at the far right.
      if (y >= 5 && y <= 25 && x >= this.size[0] - 53 && x <= this.size[0] - 33) {
        event?.preventDefault?.(); event?.stopPropagation?.();
        openHelp(kind);
        return true;
      }
    }
    return priorMouseDown?.call(this, event, pos, canvas);
  };
}

app.registerExtension({
  name: "ComfyUI-YuE2.AdapterHelp",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    const kind = helpKind(nodeData, nodeType);
    if (kind) installHeaderHelp(nodeType, kind);
  },
});
