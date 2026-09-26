# Dependency URLs

Every external thing Sprite Studio needs, with the URL it comes from and the
version or commit it is pinned to.

There is no `requirements.txt`, `pyproject.toml` or lockfile in this repo.
**Anything a package index can supply is deliberately not listed here** — the pip
packages are pinned in the `Dockerfile` itself, and the reasoning behind the three
load-bearing pins (`torchvision==0.24.0`, `transformers==4.57.3`,
`pillow==11.3.0`) is in that file's comments. What follows is only what no package
index can give you.

All URLs below were checked on 2026-09-25, following redirects: **all four
returned 200.**

## 1. The four external folders

These are the directories the app reads at run time. Each has an env override
(`SPRITE_MODEL_DIR`, `SPRITE_PPA_DIR`, `PIXELOE_DIR`, `SPRITE_SNAPPER_DIR`).

| Dependency | URL | Pinned | Env var |
|---|---|---|---|
| VRMBG-3.0 weights | <https://huggingface.co/briaai/VRMBG-3.0> | `59716e19a6cc97f91311edea190938219c097b76` | `SPRITE_MODEL_DIR` |
| proper-pixel-art | <https://github.com/KennethJAllen/proper-pixel-art> | `ad690b494bcbc455285ac6b7832a73e3ebd4675b` | `SPRITE_PPA_DIR` |
| PixelOE | <https://github.com/KohakuBlueleaf/PixelOE> | `1d45ba0b5c51c3d998b19043a168366a8f170eaa` | `PIXELOE_DIR` |
| spritefusion-pixel-snapper | <https://github.com/Hugo-Dz/spritefusion-pixel-snapper> | see below | `SPRITE_SNAPPER_DIR` |

**VRMBG-3.0 is not on GitHub.** It is a gated HuggingFace model repo
(`extra_gated_fields` in its README), so it cannot be cloned anonymously. The
page returns 200 to an anonymous request, but the weights need an authorised
token.

**What we need from it: 4 of its 7 files.** `pipeline.py:415` calls
`AutoModelForImageSegmentation.from_pretrained(model_dir, trust_remote_code=True)`,
which resolves everything through `config.json`'s `auto_map`:

| File | Size | Role |
|---|---|---|
| `config.json` | 211 B | `auto_map` names the two modules below |
| `vrmbg3_config.py` | 232 B | the `VRMBG3Config` class |
| `model.py` | 23 KB | the architecture — `BiRefNet`, Swin-L, 6-channel |
| `model.safetensors` | 844 MB | the weights |

Nothing else is read. Two traps in the other three files:

- **`pytorch_model.bin` is a second copy of the same weights** (844 MB) in the
  older pickle format. Transformers prefers `model.safetensors`, so it is dead
  weight — it doubles the download for nothing.
- `model.py` imports its config **relatively**
  (`from .vrmbg3_config import VRMBG3Config`), so those two files must land in
  the *same* directory. Copying `model.py` on its own will not import.

`README.md`, `.gitattributes` and `.git/` (1.7 GB) have no runtime role either.

So the model costs **845 MB, not 3.3 GB** — and the `Dockerfile` fetches exactly
those four files by revision-pinned URL with `aria2c`, rather than cloning the
repo. There is no `COPY --from=vrmbg` stage carrying the weights.

The model is also the reason two packages that this repo never imports are
required: its own `model.py` pulls in `timm` (`timm.layers`) and `einops`
(`einops.rearrange`), and it runs under `trust_remote_code=True`, so that code is
executed against whatever is installed. **No dependency scan of Sprite Studio
would ever reveal them.** They are pinned in the `Dockerfile` for exactly this
reason.

**The snapper is pinned by commit, not by clone.** The checkout on this machine
is a single local `first commit` (`b7365155bd6033f15daf4deb920aa542a806c790`)
whose SHA exists on **no reachable remote**, so no remote can serve that commit —
not via `clone`, not via `git init` + `fetch --depth 1 origin <sha>`. The
`Dockerfile` therefore clones the canonical `Hugo-Dz` repo's default branch and
rebuilds the WASM with `wasm-pack`, so **the WASM in the image is not
byte-identical** to the one this machine runs. Its `pkg/` folder is also
gitignored upstream, so a clone has no WASM at all until it is built.

## 2. Cloning what can be cloned

The three GitHub checkouts, at the exact commits this machine runs:

```bash
git clone https://github.com/KennethJAllen/proper-pixel-art
git -C proper-pixel-art checkout ad690b494bcbc455285ac6b7832a73e3ebd4675b

git clone https://github.com/KohakuBlueleaf/PixelOE
git -C PixelOE checkout 1d45ba0b5c51c3d998b19043a168366a8f170eaa

# A fresh clone lands on a different commit than the local checkout — see above.
git clone https://github.com/Hugo-Dz/spritefusion-pixel-snapper
```

The model cannot be cloned without HF credentials for the gated repo; request
access on the model page, then:

```bash
git clone https://huggingface.co/briaai/VRMBG-3.0
```

Note the snapper clone will land on a different commit than the local one and
its `pkg/` will be empty — build it with
`wasm-pack build --target web --out-dir pkg --release`.
