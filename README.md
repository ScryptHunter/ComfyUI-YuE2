<p align="center">
  <img src="./assets/readme/hero.svg" width="100%" alt="ComfyUI-YuE2 — safe local song generation and audio-to-score workflows for ComfyUI">
</p>

<p align="center">
  <strong>Generate complete 48 kHz songs, extract editable ABC scores from reference audio, and build cover workflows without downgrading ComfyUI.</strong>
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#model-layout">Models</a> ·
  <a href="#nodes">Nodes</a> ·
  <a href="#what-this-fork-fixes">Fork fixes</a>
</p>

## What this is

**ComfyUI-YuE2** connects two complementary music models inside ComfyUI:

- **YuE2** turns a style prompt and sectioned lyrics into a complete stereo song. It can expose or accept an editable ABC melody-and-chord plan.
- **SheetSage2** turns reference audio into ABC, MIDI, and timed song sections for covers, rearrangements, and lyric alignment.

This fork focuses on running both models safely inside a modern shared ComfyUI environment. It does **not** install the official YuE2 wheel through pip, does **not** replace ComfyUI's Torch stack, and can load MERT2 entirely from a local model folder.

> **Fork notice:** This repository is a compatibility-focused fork of
> [`piscesbody/ComfyUI-YuE2`](https://github.com/piscesbody/ComfyUI-YuE2),
> updated for dependency-safe installation, current ComfyUI environments,
> modern Transformers releases, local MERT2 loading, and an English interface.

## Workflows

| Goal | Node path | Reference audio |
| --- | --- | --- |
| Generate a new song | `YuE2 Model Loader` → `YuE2 Song Generator` | Not required |
| Generate and edit a score first | `YuE2 Generate Plan` → ABC tools → `YuE2 Render Plan` | Not required |
| Create a cover or rearrangement | `Load Audio` → `SheetSage2 Audio Transcriber` → `YuE2 Song Generator` | Required |
| Extract only a score | `Load Audio` → `SheetSage2 Audio Transcriber` | Required |

For covers, SheetSage2 extracts a symbolic score. YuE2 then creates a new recording from that score, the target style, and the lyrics. This is **not voice cloning** and does not preserve the original waveform.

## What this fork fixes

- **Dependency-safe YuE2 runtime** — extracts the pure-Python `yue2_infer` wheel into a node-private runtime instead of allowing pip to downgrade Torch, Transformers, NumPy, or Hugging Face Hub.
- **Modern Transformers compatibility** — adapts legacy Transformers 4.x model APIs to current Transformers 5.x behavior.
- **SheetSage2 tied weights** — recognizes the intentionally shared decoder and output embeddings instead of rejecting the adapter checkpoint as incomplete.
- **Correct MERT2 rotary embeddings** — repairs corrupted `inv_freq` buffers that can otherwise produce rhythm and chords without reliable melody pitch.
- **Fully local MERT2 loading** — loads the parent encoder from `models/MERT2/` without copying model code into the global Hugging Face modules cache.
- **Windows attention fallback** — detects unavailable built-in Flash Attention and switches GraphAR to the cuDNN SDPA backend.
- **ComfyUI runtime isolation** — restores process-wide CUDA memory limits and precision flags after YuE2 uses them.
- **Legacy workflow migration** — accepts old workflows that stored `vae_decode=false` and maps the value to `tiled`.

## Quick start

### 1. Install the node dependencies

Run pip with the Python executable used by ComfyUI:

```powershell
python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-YuE2\requirements.txt
```

The requirements intentionally omit `torch`, `torchaudio`, `transformers`, and `numpy`. They also omit the obsolete `descript-audiotools` dependency, which can force a destructive Protobuf downgrade.

### 2. Do not pip-install the YuE2 wheel

Place the wheel shipped with the matching YuE2 model directly inside the model folder:

```text
ComfyUI/models/YuE2/YuE2-3B/yue2_infer-0.1.5-py3-none-any.whl
```

The node extracts it into `.yue2_runtime/` and imports it without resolving the wheel's pinned dependencies.

### 3. Download the models

See the complete folder layout below. Restart ComfyUI after installing dependencies or updating the node.

## Model layout

```text
ComfyUI/models/
├── YuE2/
│   ├── YuE2-3B/
│   │   ├── config.json
│   │   ├── ...weights...
│   │   └── yue2_infer-0.1.5-py3-none-any.whl
│   ├── YuE2-Vae/
│   └── YuE2-Vae-legacy/                 # optional benchmark decoder
├── SheetSage2/
│   ├── config.json
│   ├── model.safetensors
│   └── ...repository Python files...
└── MERT2/
    └── MERT-v2-FullSong/
        ├── config.json
        ├── model.safetensors
        └── ...repository Python files...
```

| Folder | Source |
| --- | --- |
| `YuE2/YuE2-3B/` | [`m-a-p/YuE2-3B`](https://huggingface.co/m-a-p/YuE2-3B) |
| `YuE2/YuE2-Vae/` | [`m-a-p/YuE2-Vae`](https://huggingface.co/m-a-p/YuE2-Vae) |
| `YuE2/YuE2-Vae-legacy/` | [`m-a-p/YuE2-Vae-legacy`](https://huggingface.co/m-a-p/YuE2-Vae-legacy) |
| `SheetSage2/` | [`m-a-p/SheetSage2`](https://huggingface.co/m-a-p/SheetSage2) |
| `MERT2/MERT-v2-FullSong/` | [`m-a-p/MERT-v2-FullSong`](https://huggingface.co/m-a-p/MERT-v2-FullSong) |

Download the MERT2 revision expected by the SheetSage2 adapter:

```powershell
hf download m-a-p/MERT-v2-FullSong `
  --revision d8ba1c745e733b3908ce6ad16ebeb17ac7600a42 `
  --local-dir ComfyUI/models/MERT2/MERT-v2-FullSong
```

The loader also accepts `MERT-v2-FullSong/` nested directly inside `models/SheetSage2/`. If neither local location exists, the `auto` option falls back to the standard Hugging Face cache/download behavior.

### Custom model paths

```yaml
yue2_music:
  base_path: F:/models/music
  yue2: YuE2/
  sheetsage2: SheetSage2/
  mert2: MERT2/
```

## First generation

1. Add **YuE2 Model Loader** and select the generator and VAE.
2. Add **YuE2 Song Generator**.
3. Enter a style prompt such as `English indie pop, warm female vocal, clean guitar, restrained drums`.
4. Provide sectioned lyrics:

```text
[verse]
City lights are fading softly
Morning waits beyond the blue

[chorus]
Carry every spark of wonder
Let the open road come through
```

5. Use `cot=full` for an editable melody-and-harmony plan, `cot=melody` for a melody-only cover plan, or `cot=off` for direct generation.

An importable example is available at [`example_workflows/example_workflows.json`](./example_workflows/example_workflows.json).

## Cover workflow

```text
Load Audio
    └─→ SheetSage2 Model Loader → SheetSage2 Audio Transcriber
                                      ├─→ ABC score ──────────────┐
                                      └─→ section structure ──┐   │
                                                              ▼   ▼
Plain/timed lyrics → Lyrics Formatter or Structurer → YuE2 Song Generator
                                                        cot=melody
```

Use `melody_only=true` in SheetSage2 and `cot=melody` in YuE2 when the goal is to preserve the tune while allowing a new arrangement.

For a controllable remix workflow, connect the plan's ABC output through **ABC
Analyzer**, **Melody Cleanup**, **Vocal Range Retarget**, and/or **ABC Modifier**,
then connect the edited text to `abc_override` on **YuE2 Render Plan**. The
original plan still carries the exact style, lyrics, seed, COT mode, and CFG
settings, so the render node can rebuild the symbolic plan without regenerating
it randomly.

## Nodes

| Node | Description |
| --- | --- |
| **YuE2 Model Loader** | Loads the generator and VAE through the isolated runtime. |
| **YuE2 Song Generator** | Creates a 48 kHz stereo song and optionally saves FLAC and ABC outputs. |
| **YuE2 Unload Model** | Releases the cached YuE2 pipeline and VRAM. |
| **YuE2 Memory Preset** | Produces loader and VAE settings for 12/16/24/32/48 GB GPU tiers. |
| **YuE2 Advanced Sampling Settings** | Exposes separate official `temperature`, `top_p`, `top_k`, repetition penalty/window, and token limits for ABC and semantic generation. |
| **YuE2 Generate Plan** | Runs only symbolic planning and returns the exact reusable plan object plus ABC. |
| **YuE2 Render Plan** | Renders a plan—or an edited ABC override—through semantic, NAR, and VAE stages. |
| **YuE2 Decode Latents** | Decodes cached acoustic latents without repeating earlier stages. |
| **YuE2 Plan Batch** / **Plan Selector** | Explores several inexpensive ABC plans from consecutive seeds and selects one for rendering. |
| **SheetSage2 Model Loader** | Loads the transcription adapter and local or cached MERT2 parent. |
| **SheetSage2 Audio Transcriber** | Produces ABC, MIDI, and timed structure; supports strict, snap, skip, full-score, and MIDI-only ABC failure policies. |
| **SheetSage2 Unload Model** | Explicitly releases SheetSage2/MERT2 VRAM. The transcriber can also auto-unload. |
| **YuE2 ABC Analyzer** | Reports BPM, key, meter, approximate duration, voice note counts, ranges, medians, and vocal-range warnings. |
| **YuE2 ABC Modifier** | Overrides or scales BPM, transposes the score or individual voices, shifts octaves, removes chords, and trims named sections. |
| **YuE2 Vocal Range Retarget** | Automatically applies an octave-safe Vocal shift toward common male/female ranges, or accepts a manual shift. |
| **YuE2 Melody Cleanup** | Replaces suspiciously short or extreme Vocal notes with equal-duration rests while preserving the timeline. |
| **YuE2 ABC File Loader / Saver** | Loads editable `.abc` files from input or saves them under output. |
| **YuE2 MIDI File Saver** | Saves the in-memory MIDI output from SheetSage2. |
| **YuE2 Style Prompt Builder** | Builds a compact prompt from language, genre, era, vocal, instrument, drum, mood, and tempo descriptors. |
| **YuE2 Lyrics / Melody Fit Analyzer** | Compares estimated English syllables with Vocal note counts per section. |
| **YuE2 Lyrics Formatter (Timed to Sections)** | Converts LRC, SRT, or aligned text into sectioned lyrics. |
| **YuE2 Lyrics Structurer (Text to Sections)** | Adds and distributes section tags across plain lyrics. |
| **YuE2 Lyrics File Loader (LRC/SRT)** | Reads lyric files from the ComfyUI input directory. |

## Generation controls

| Control | Meaning |
| --- | --- |
| `cot` | `full`: melody + harmony plan; `melody`: melody only; `off`: no score planning. |
| `attention_backend` | `auto`, `external-flash`, `cudnn`, or `sdpa`. Windows normally uses the automatic cuDNN fallback. |
| `vae_decode` | `tiled` for lower memory use or `full` for faster high-VRAM decoding with automatic OOM fallback. |
| `vae_tile_frames` | Tile size for VAE decoding. `0` chooses automatically; `256` is suitable for low-VRAM systems. |
| `quantization` | Experimental FP8 AR quantization. It saves memory but can be significantly slower. |
| `abort_after_plan` | Stops after generating the ABC score without synthesizing audio. |
| `cfg_scale=-1` | Uses official defaults: 1.0 for full/melody planning and 1.01 for direct (`off`) generation. |
| `sampling_settings` | Replaces the compact sampler controls with the complete official ABC/semantic sampling configuration. |
| `save_artifacts` | Saves reproducibility data: request, plan IDs, ABC, semantic IDs, latents, settings, timings, hashes, and audio. |
| `vae_decode=auto` | Uses full decode only on a GPU with roughly 48 GB or more, otherwise tiled decode. |

### SheetSage2 ABC recovery

`strict` remains the default and preserves upstream behavior. If melody-only
ABC export fails on a decoded note that cannot be represented on the sub-beat
grid, the other policies preserve the completed transcription:

- `snap_invalid_notes` builds a conservative 1/32-grid score from returned MIDI;
- `skip_invalid_notes` does the same but drops overlapping/invalid notes;
- `fallback_full` retries the upstream full score with harmony;
- `return_midi_only` returns MIDI and structure without requiring ABC.

Recovered MIDI-derived ABC intentionally omits chord symbols instead of
inventing harmony. Review it in an ABC editor before an important render.

## Compatibility

The current fork has been exercised on Windows with:

```text
ComfyUI       0.35
Python        3.13
PyTorch       2.11 + CUDA 13.0
Transformers  5.14
GPU           NVIDIA RTX 4090
```

Linux and other compatible NVIDIA GPUs should work, but the full matrix has not been tested. YuE2 requires substantial VRAM and a CUDA GPU with BF16 support is strongly recommended.

## Limitations

- Model files are not included in this repository.
- The YuE2 wheel must match the selected model release.
- Reference audio is used to extract a symbolic score; it does not clone the original singer.
- Vocal-only non-octave transposition changes the melody's harmonic relationship; octave shifts are the safe automatic default.
- Memory presets are starting points, not guarantees: song length, drivers, attention backend, and other loaded ComfyUI models affect VRAM use.
- YuE2 model weights are non-commercial. Review every upstream model license before use.
- `external-flash` requires a compatible optional `flash-attn` installation; the default backend does not.

## Credits and license

This repository is derived from
[`piscesbody/ComfyUI-YuE2`](https://github.com/piscesbody/ComfyUI-YuE2) and is
maintained as a compatibility-focused fork. The original node implementation,
model architecture, inference code, and weights belong to their respective
upstream projects:

- [`piscesbody/ComfyUI-YuE2`](https://github.com/piscesbody/ComfyUI-YuE2)
- [`multimodal-art-projection/YuE`](https://github.com/multimodal-art-projection/YuE)
- [`m-a-p/YuE2-3B`](https://huggingface.co/m-a-p/YuE2-3B)
- [`m-a-p/SheetSage2`](https://huggingface.co/m-a-p/SheetSage2)

Node code is distributed under the repository [`LICENSE`](./LICENSE). Model weights and bundled upstream artifacts retain their own licenses; YuE2 weights are released under **CC BY-NC 4.0**.
