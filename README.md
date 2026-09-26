🐣 Please follow me for new updates: https://x.com/camenduru <br />
🔥 Please join our discord server: https://discord.gg/k5BwmmvJJU <br />
🥳 Please become my sponsor: https://github.com/sponsors/camenduru <br />
🍞 TostUI repo: https://github.com/camenduru/TostUI

#### 🍞 Tost AI - Sprite Sheet Studio

https://github.com/user-attachments/assets/7aec9682-719a-45fd-842c-abe9877205ea

Repo: <https://github.com/camenduru/TostAI-Sprite-Sheet-Studio> — **private**, so it
returns 404 to anyone without access rather than being gone. Renamed from
`Tost-Sprite-Studio` on 2026-09-25; GitHub redirects the old URL, but the
Dockerfile and `app.py`'s `APP_REPO` now carry the new one. This is also
`origin` in the local checkout.

Two tools in one server:

1. **Generator** — turn a raw video into a game-ready sprite sheet:
   **VRMBG-3.0 video matting → optional backdrop repair → tight auto-crop →
   packed grid sheet + sidecar JSON + preview player + GIF**, with an invariant
   gate that checks the result instead of trusting it.
2. **Editor** — open any existing sheet (from a run, from `walk/`, from
   `sprites/`, or any PNG + sidecar) and fix it: align the pivot, repair the
   matte, key out a colour, repaint, re-grid, re-time, re-export.

## Run it

```bash
cd C:\Users\PC\Desktop\monsters\sprite\sprite_studio
C:\Users\PC\AppData\Local\Programs\Python\Python313\python.exe app.py
```

- **http://127.0.0.1:8765** — the generator
- **http://127.0.0.1:8765/editor** — the editor

Options: `--host`, `--port`.

Requires the Python that has the ML stack (`torch`, `cv2`, `transformers`, `PIL`,
`numpy`, `fastapi`, `uvicorn`) — on this machine that is the 3.13.9 install above,
not the sandboxed runtime. `ffmpeg` is optional and only used by the helper
scripts.

### Docker

The image is self-contained: it carries the model, the three pixel ops, and a
supervisor that starts the Cloudflare tunnel and then hands PID 1 to the app.

**Build.** The three tokens arrive as **secret mounts sourced from env vars** —
never `--build-arg`, because a build-arg token is visible in
`docker history --no-trunc`. They live in `.env`:

```bash
set -a; . ./.env; set +a
docker build --progress=plain \
  --secret id=hf_token,env=HF_TOKEN \
  --secret id=gh_token,env=GITHUB_TOKEN \
  --secret id=cf_token,env=CLOUDFLARED_TOKEN \
  --build-arg CACHEBUST=$(date +%s) \
  -t TostAI-Sprite-Sheet-Studio .
```

- `HF_TOKEN` — the VRMBG-3.0 repo is gated; the model download 401s without it.
- `GITHUB_TOKEN` — the app repo is private, so an unauthenticated clone 404s.
- `CLOUDFLARED_TOKEN` — the tunnel. It is **baked into the image**, readable at
  `/etc/cloudflared/token` inside the container, so the image is not for public
  distribution. Pass `-e CLOUDFLARED_TOKEN=...` at run time to override it without
  rebuilding — that is how you rotate it.
- `CACHEBUST` — the app clone is deliberately unpinned (always latest source), and
  a cached `git clone` silently re-serves the first snapshot. Bumping this forces
  a re-clone. **Omit it and you get stale code with no warning.** The commit that
  landed is written to `.sprite_rev` inside the image.

**Run:**

```bash
docker run -d --name TostAI-Sprite-Sheet-Studio \
  --gpus all \
  -p 8765:8765 \
  --restart unless-stopped \
  TostAI-Sprite-Sheet-Studio
```

- `--gpus all` is **not optional** — without it `torch.cuda.is_available()` is
  `False` and matting falls back to CPU.
- **http://127.0.0.1:8765** — the generator
- **http://127.0.0.1:8765/editor** — the editor
- The tunnel starts automatically and serves the public hostname
  (`sprite.tost.ai`, from the tunnel's own remote-managed ingress).
- With no token the app still starts — the tunnel is skipped and a loud multi-line
  warning is logged. `-e SPRITE_NO_TUNNEL=1` skips the tunnel deliberately, even
  when a token is present.

**Check it actually came up.** `docker ps` reports *healthy* even when the tunnel
is dead, because the healthcheck probes the app, not the tunnel — a wrong
cloudflared invocation exits instantly and the supervisor restart-loops forever
with the container still green. The tunnel's own readiness endpoint is the only
honest signal:

```bash
docker exec TostAI-Sprite-Sheet-Studio sh -c 'curl -s http://127.0.0.1:20241/ready'
# {"status":200,"readyConnections":4,"connectorId":"..."}
```

`readyConnections: 4` is a healthy connector (Cloudflare runs four). To read the
connector's own log, `docker logs TostAI-Sprite-Sheet-Studio | grep 'Registered tunnel
connection'`. **Never `pgrep -a cloudflared` or `ps aux`** — the token is in its
argv and will be printed.

A from-scratch rebuild (`--no-cache`) re-downloads ~7.4 GB, including the 845 MB
model.

## The workflow

1. **Input** — type a path, use *Browse…*, or drag a file onto the drop zone
   (uploads land in `uploads/`). *Read video* shows frame count, fps, size, a
   **playable preview of the clip**, and the valid column counts.
2. **Background removal** — VRMBG-3.0, autoregressive along time. Pick the
   inference size, a frame range and a step.
3. **Matte repair** — two passes over the matted frames, both **off by default**,
   because both are destructive when the matte did not leak. Asked for, not
   assumed.
   - **Border flood-fill.** Topological: it walks near-black pixels *reachable
     from the frame border*, which includes the subject's own dark edge where the
     subject touches the border. I2 is what says whether it was needed.
     *Only regions touching the frame border* keeps that connectivity test;
     it starts **unticked** — the tonal black key is the default, because it is
     the mode that reaches what a bad matte leaves behind a closed silhouette —
     and ticking it restores the test. The tonal key reaches backdrop the
     subject closed off — inside a handle, between an arm and the body — at the
     price of eating the subject's own dark parts, and the two are not
     distinguishable from the pixels alone, which is why it is a box rather than
     a smarter default. The editor's *erase a colour* has carried the same
     switch all along (`connected`), so this is the generator catching up.
   - **Colour key.** Tonal, and the one a green screen needs: a pixel within
     *tolerance* of the picked *backdrop colour* loses its alpha. The flood fill
     cannot reach this leak by construction — a green edge hugging the subject is
     *enclosed* by the subject, and the fill only walks in from the border — which
     is why both exist. Reading a clip **seeds the picker with the clip's own
     border colour** (the walk clip's is `#12ff4d`; the textbook `#00ff00` is 82
     away from it, which at a tolerance of 64 matches nothing at all), and the
     pass only acts within `KEY_EDGE_BAND` — 6 px of transparency — so a subject
     painted in the key colour loses its contour rather than being erased whole.

     The key has the *Also clear key-coloured regions touching the frame border*
     box, and it can only ADD reach, never trade it: unticked is the 6 px band,
     ticked clears the border-connected key regions on top of the band. It began
     life as a swap — the flag replaced the band with the connectivity test — and
     that read as the box working backwards: on the walk clip the swap cleaned
     LESS (5.5% of the visible key-coloured edge left, against the band's 0.6%,
     because the band also takes the faint ramp ring the connectivity test leaves
     wherever the ring is broken by pixels beyond the tolerance), so ticking it
     re-opened leak the band had closed. A checkbox next to a leak has to mean
     "clean more". What the tick adds is the leak the band cannot see at all (a
     slab of backdrop left along a frame edge, opaque and deeper than six
     pixels), with the guarantee that an enclosed patch of the key colour — a
     subject painted like the backdrop — is still never touched at any
      tolerance, in either state. The box above it, on the flood fill, keeps its
      *Only regions touching the frame border* label and its swap semantics,
      because there the topological test IS the pass and the flag relaxes it.
      Both boxes start unticked, like both passes: matte repair is asked for,
      not assumed.

     Finally, the pass **takes the colour out of the spill it cannot match**. On
     `MiniMax_H3_00183___1_` (backdrop `#13ff0b`) the leftover the eye catches is
     the sprite's outline: `(0,26,0)` to `(0,55,0)` — hue 120°, saturation 255,
     value 23-62, alpha 195-253, 1-2 px deep along the silhouette. Those pixels are
     194-233 away from *any* green backdrop key, because the encoder and the
     upscale bled the backdrop into a black outline, and no tolerance reaches them
     without swallowing the subject: at 255 a mid-grey is 205 away and goes too.
     That is what the nine runs on that clip show — the sprite down to 62k opaque px
     from 254k, and the green still there. So the key's own channel is clamped to
     `max(the other two)`: `(0,50,0)` becomes `(0,0,0)`, the black outline it was
     always meant to be. Nothing is deleted and no alpha changes, so the silhouette
     keeps its shape. **The despill's reach is the 6 px band around full
     transparency plus every semi-transparent pixel, wherever it sits** — the
     original band-only reach was the bug behind "the box does nothing, the tail
     is still green": on the user's 2x-upscaled sheet the leftover spill sat at
     alpha 238 (p90 253) up to **21 px** from transparency (p90 12.8), so half of
     it was outside any 6 px band, in *both* checkbox modes — and since it is the
     key's channel dominance that is clamped, never alpha, a soft pixel is the one
     place despill is safe to guess: it cannot be load-bearing detail, and the
     green-creature fixture shows the body's saturation survives (band mode
     recolours 0 body px; border mode only the ring its own stronger key turned
     semi-transparent). Opaque spill deeper than the band is still left alone on
     purpose — on a subject painted like the backdrop those pixels ARE the
     subject. Over 4 frames of that clip, keyed on its own seeded backdrop:
     tolerance 32 leaves 2.0% of the sprite removed and recolours 33785 px; 64, 3.0%
     and 23699; 96, 3.6% and 18425; 255, 4.0% and 14297 — with **zero** visible
     green and no green patch left at every one of them, against a baseline of
     52807 px of rim spill (28337 visible, a 1991 px blob at the tail). The
     tolerance now trades deletion against recolouring; the despill is what takes
     the green out, which is why raising it was never the fix.

     Measured on `runs/0921-084914_MiniMax_H3_00177___1_` (124 frames, backdrop
     `#12ff4d`): 177422 px of visible key-coloured edge, 1116 left at tolerance 96
     (0.6%) and 111765 at 64. What survives is the sprite's *own* green — 1943 px
     of green rim per frame, of which about 20 are within the tolerance at all. A
     colour key cannot tell a subject painted like the backdrop from the backdrop;
     that is a stated limit, and it is why the pass is bounded and off until asked
     for.
4. **Sheet layout** — cell mode, padding, columns, texture cap.
5. **Playback** — fps, loop / ping-pong / one-shot, trim, anchor, blend.
6. **Outputs** — sheet, sidecar, preview HTML, GIF, kept frames.
7. **Verification** — the invariant gate.

Results land in `runs/<date-time>_<name>/` and are listed under *Previous runs*.
Every finished run has an **Edit this sheet →** link that opens it in the editor.

## The editor

The browser holds no pixels. It ships **op commands** to the server and receives
back **the indices of the cells that actually changed**, then re-fetches only
those cells as small PNGs. One source of truth for pixels (numpy), one for
invariants (the same `border_reachable` the pipeline uses). Re-implementing
flood fill, morphology and alpha compositing in canvas would have meant two
implementations that have to agree, and they would not.

**44 operations in 9 groups**, each with its own controls generated from the op
registry (`GET /api/editor/ops`) — with one deliberate exception, `snap_pixels`,
which builds its own panel because its output is a preview. One *control* is
shared outside the generator, though: the palette picker is a `paletteControl()`
in the page, called by the Snap Pixels panel and by the generic form whenever an
op declares an argument named `palette`, so the two panels cannot drift apart.

| group | operations |
|---|---|
| **Pixel art** | **Snap Pixels** — [spritefusion-pixel-snapper](https://github.com/Hugo-Dz/spritefusion-pixel-snapper) over the selection: colours, pixel-size override, a palette picker (auto k-means, preset console palettes, or a palette PNG), live preview, preview zoom in/out · **Pixelate (mesh)** — [proper-pixel-art](https://github.com/KennethJAllen/proper-pixel-art) true-resolution mesh recovery, one shared grid + palette across the selection, with the **same palette picker** · **Pixelize (outline)** — [PixelOE](https://github.com/KohakuBlueleaf/PixelOE) contrast-aware outline expansion + block downsample, *imposes* a grid rather than recovering one, with the **same palette picker** · **Resize (pixels)** — scale the art by a whole factor with nearest-neighbour, so pixel art stays hard-edged |
| Transform | nudge, flip H/V, rotate 90/180/270, **match content size (stop the loop pulsing)**, align pivot (de-jitter), crop to content, grow cell canvas, resample all, copy one cell into the selection |
| Alpha | threshold, remap range, scale, opacity, erode, dilate, feather |
| Matte repair | **un-premultiply RGB**, erase leaked backdrop, erase a colour (**picked from a swatch**) |
| Colour | brightness/contrast/saturation/gamma, posterize, **limit palette (k-means to N colours)**, fill/tint (**colour picked from a swatch**) |
| Frames | reverse, rotate sequence, keep a range, **drop the selection**, hold a frame, drop duplicates, **keep the best N**, insert in-betweens, reorder |
| Layout | set columns, auto columns, resize cell canvas |
| Paint | composite a patch (pencil / eraser, driven by the canvas) |
| Meta | playback metadata, verification thresholds |

Tools: select, pan, pencil, eraser, colour picker. Zoom/pan with the wheel and
middle-drag, marquee-select cells, ctrl/shift-click to extend, live animation
preview with onion skin, a clickable timeline, per-frame audio, and full
undo/redo.

**Play selection only bounds the seek, not just the playback.** With it ticked the
scrub stops being a frame index and becomes a *position in the playback order*:
for a selection of {0, 23} a slider spanning 0..23 spends 22 of its 24 stops on
frames the preview never shows, and the arrow keys walk straight through them —
reported as "if play selection only, seek should only work on the selected
frames". Every stop now lands on a played frame, the arrows step frame to frame
within the selection, and a cursor that is outside it (a ctrl-click can deselect
the frame you are standing on) snaps to the nearest selected one, which is what
pressing Play does too. Untick it and the whole document is back. The timeline
thumbnail click is deliberately *not* bounded: selecting a frame is how you put it
into the selection, so blocking it would make the selection uneditable.

Both pages are **Tost AI Sprite Studio**, with the TostUI mark inlined into the
header, and both have a dark theme: the button in the header, or `D` in the
editor. The choice is one `localStorage` key shared by the generator and the
editor, and the OS preference decides the first visit; the attribute is set by an
inline script in `<head>`, so a dark page never flashes the light palette. It is
one block of CSS-variable overrides and nothing that draws has to know which
theme is on — the canvas reads back the single colour it needs in JS (`--stage`)
on each switch rather than per frame.

The palette is **TostUI's**, read out of its `app/globals.css` and mapped onto
this editor's roles rather than invented: `--background`, `--card`, `--border`,
`--foreground`, `--destructive` and the cyan primary are TostUI's own values, and
the few roles TostUI has no value for — a faint border, the recessed surface,
success — are further steps of the same Tailwind scales it draws on. Two
deviations are deliberate, and both are forced by the contrast rule below.
TostUI's `--primary` is cyan-600 and **white on cyan-600 is 3.68:1**, so the
primary *fill* here is cyan-700; and TostUI's dark `--destructive` is a fill,
which as 11px text on a tint is 4.06:1, so the text step is red-400. TostUI's
dark theme drops its own primary to grey, which a tool that signals "selected /
active / drop here" cannot afford, so the accent keeps its hue across both
themes.

Contrast is measured, not eyeballed: every piece of text, including the status
chips and the toast, is held at WCAG 4.5:1 against its effective background in
both themes. `_probe_palette.py` is the offline pre-check — it computes the same
ratios straight from the two variable blocks, so a candidate colour is rejected
in a second instead of after a browser run. `_probe_theme.py` stays the
authority, because it reads the colours the browser actually resolved.

### Things worth knowing

- **Align pivot is the point of the tool.** A matting model's bbox moves frame to
  frame, so the subject jitters even though every frame is correct in isolation.
  Align moves each selected cell so a chosen point on the subject — bottom-centre
  is the usual one for a character — lands on the *same pixel* in every cell.
  Anchors return **integer pixels, deliberately**: a geometric centre of an
  even-width box sits on a half-pixel, and two cells whose centres are at x.5 and
  y.5 cannot both be translated onto one pixel. They would land one pixel apart
  and the "aligned" sheet would still jitter.
- **`grow` is only reachable for the centroid anchor.** For a box anchor the
  aligned extent is exactly the widest subject, which already fits the cell. The
  centroid is not bounded by the bbox, so it is the one anchor that can need a
  bigger cell — and then every cell grows, not just the selection.
- **Undo stores PNG-encoded pre-states, not raw arrays.** A 512×640 RGBA cell is
  1.3 MB raw and 5–30 KB as PNG, because sprite cells are mostly transparent.
  That turns a 124-cell align from a 31 MB undo step into ~0.3 MB — still ~100×
  smaller even though a full entry now carries the loaded reference as well. The
  stack is capped by bytes, not just by count.
- **Un-premultiply is the exact inverse of the paste bug** that this project
  already hit once: `rgb /= (a/255)`. If a sheet's semi-transparent edges look
  too dark, that is why, and this is the fix.
- **Verification is honest about what an edit destroys.** I2 (backdrop leak) is a
  property of the cell alone, so it still runs on edited cells. I1 (straight
  alpha) and I4 (subject intact) compare against the *loaded* cell, so after an
  edit they can only cover the unmodified cells — and the report says so in
  those words rather than quietly reporting a smaller number. There is also a
  reference-free "premultiply smell" line: under straight alpha `max(RGB)` is
  routinely above alpha, under premultiplied alpha it never is. It is the only
  signal that survives an edit, because it needs no reference — so it is counted
  *before* the `i in doc.dirty` skip. Two caveats it does not hide: the test is
  restricted to `alpha > 0` (a real matte leaves stray RGB under transparent
  pixels — this project's own torch sheet has `RGB=2` at `alpha=0`, which used to
  count as a hit and silenced the line on every real sheet), and an all-opaque
  sheet trivially satisfies `max(RGB) <= alpha`, so it is reported as
  premultiplied. The line is informational and says so.
- **`_shift` cannot grow an array.** `_place` exists for that. Using `_shift` to
  "pad" moves the content and then clips whatever fell outside — invisible on a
  subject that sits mid-cell, wrong on one that touches the edge.
- **An undo snapshot must be sized from the *pre*-op frame count.** Ops are free
  to change `doc.n`: `keep_range`, `dedupe` and *drop selected frames* all shrink
  the sheet, `interpolate` and *hold a frame* grow it. Deriving the snapshot's
  index range from `doc.n` **after** the op drops the tail cells, and `_restore`
  then rebuilds `range(entry["n"])` and raises `KeyError` on the first index past
  the new count — so a `keep_range` was silently un-undoable. Growing ops only
  escaped it because a `if i in pre` filter happened to discard the surplus.
- **Don't min-max-normalise two factors and multiply them.** Min-max maps the
  *worst* frame on each axis to exactly 0, so the product collapses to 0 for
  every frame the moment the two factors peak on different frames — the metric
  stops discriminating while still looking like it works. Measured on a fixture
  of 11 crisp frames and 1 smeared one, min-max × min-max scored all 12 frames
  0.0. *Keep the best N* scales each axis by its max instead, so the number stays
  readable ("0.53 of the best frame on this axis") and cannot collapse.
- **A refusal has to reach `#status`, and it arrives via the `catch`.** `api()`
  throws on `{ok:false, error}`, so a guard written as `if (!res.ok)` above the
  call is unreachable — the first version of the layout form had exactly that
  bug, and the reason (a texture-cap overflow, a column count that does not
  divide) flashed past in a 2-second toast while the form kept showing the value
  that had just been rejected.
- **A frame-count change is what the status line should say.** For a full op the
  "changed" index list is the *pre-op* range, so a `keep_range` that took 124
  frames down to 10 used to report "124 cells" — which reads as though nothing
  was removed. It now reports `124 -> 10 frames`.
- **`_set_cells` is atomic, and it has to be.** It used to assign `doc.cells`
  first and call `set_layout` afterwards. When the re-grid was *refused* — a
  shrink whose new frame count has no column count that fits under the texture
  cap — the document was left holding 122 cells against a 31×4 layout:
  `n != cols*rows`, `verify` reporting "122 cells (31x4)", and `save` writing a
  sheet whose sidecar claimed more frames than it had. Every frame-pruning op
  routes through it, so one refused shrink was all it took. `save_doc` now also
  refuses to write a grid that is not full, whatever the reason.
- **`orig` is part of the undo state.** It is the loaded reference that I1/I4
  verify against, and the frame ops rewrite it alongside the cells. A full
  restore that put the cells back but not the reference left `len(orig) < n`, so
  the next op to index `orig` on the tail frames raised `IndexError` — and until
  it did, I1/I4 were reporting "no reference, not applicable" for the whole
  document. `_restore` now also *recomputes* `dirty` by comparing against the
  restored reference instead of blanket-marking every cell, so a full undo no
  longer blinds the verifier.
- **Saving must not overwrite the sheet it was loaded from.** The output folder
  is a free-text field and the name defaults to the document's own name, so
  typing the source folder — the obvious move — composes the source filename
  exactly: opening `walk/<n>_sheet.png` and typing `walk/` replaced the sheet the
  whole edit was derived from, and its sidecar with it. `src_sheet`/`src_sidecar`
  were already being tracked for this kind of check and were simply never
  consulted. `save_doc` now compares the composed paths against both and refuses,
  naming the file and the opt-in; the panel has an **allow overwriting** box that
  is off by default, so replacing the original stays possible but has to be
  asked for. The default destination is a fresh `runs/editor_<time>_<name>/`.
- **The brush size is in screen pixels, and has to be.** Read as *sheet* pixels
  the default 6 meant a 6-px mark on a 15872-px-wide sheet — at the fitted zoom
  (0.054) that is **0.32 px on screen**. The stroke landed, the cell hash
  changed, the journal recorded `paint`, and the user saw nothing at all, which
  is exactly how it was reported ("eraser and pencil is not working"). The brush
  radius is now `(size / 2) / zoom`, so "6" means 6 px to the person holding the
  mouse at any zoom, and there is a **brush cursor ring** on hover so the number
  has a visible referent. Anything sized in document units needs this treatment:
  a size the user cannot see is indistinguishable from a broken tool.
- **A keyboard shortcut has to survive a focused toolbar box.** The handler used
  to bail out for *any* `<input>`, and the toolbar is full of them — so setting a
  brush size (the first thing anyone does before drawing) left focus in that box
  and `B` for the pencil and `E` for the eraser did nothing at all. That is the
  other half of "eraser and pencil is not working", and it is why the tool looked
  broken to the person using it while every driven test passed: a driven test
  clicks the tool button, and clicking the button is what fixes the focus. A
  number/colour/range box cannot contain a letter, so a letter pressed there is
  unambiguously a shortcut; anything that takes free text (a path, a name) still
  swallows every key, as it must. `Escape`/`Enter` hand the keyboard back.
- **A restart invalidates every document id an open page is holding.** The
  registry is in memory, so `GET /api/editor/<id>/cell/0.png` starts returning
  404. The page keeps its cached cell images and still looks perfectly healthy —
  toolbar, sheet, timeline, playback — while every single write dies as an
  unhandled rejection. `commitStroke` had no `catch` at all, so a failed paint
  said nothing anywhere and the tools simply appeared to be broken. The page now
  confirms the loss against the server (`docIsGone`, so a network blip cannot
  trigger it), says so in the status line — which outlives the 2-second toast,
  and the user has just lost unsaved work — and reopens the sheet. If the sheet
  cannot be reopened, it says *that* instead: an earlier version wrote the
  optimistic "Reopening it." over the specific "no such sheet: `<path>`" it had
  just failed with, leaving no document open and a status line claiming
  everything was fine. `openSheet` returns whether a document is actually
  loaded, which is what makes the honest wording possible.
- **Shift-select is a linear frame range, on the sheet and in the timeline.**
  They used to differ: the canvas drew the *grid rectangle* between the two cells
  and the timeline drew the *frame range*, and those are the same thing only
  while the two cells share a row. On the walk sheet (31 columns) cell 4 is row 0
  col 4 and cell 99 is row 3 col 6, so shift-clicking 4 then 99 gave **12 cells
  on the canvas and 96 in the timeline** — reported as "when user shift select 4
  to 99 shift select should select all frames between 4 and 99 also 4 and 99".
  Both surfaces now call one `selectRange`, which includes both ends. A rectangle
  is still available and still the only way to rubber-band from *inside* the
  sheet: that is the marquee drag. The check that should have caught this had the
  right assertion (`sel == range(a, b+1)`) and a fixture that could not
  distinguish the two meanings, because it picked cells that share a row.
  `selectRange` also clamps to the document, and the anchor is reset when a
  different sheet opens, so a stale anchor from a longer sheet cannot range from
  an index that is not there.
- **"Only the selected frames" is a save-time view, never an edit.** The obvious
  implementation — `keep_range` on the live document — writes a correct sheet and
  silently destroys every frame that was not selected, with no undo entry. The
  export builds a **throwaway document** from the sorted selection instead and
  routes it through `_set_cells`, so a subset gets the same atomic re-grid as any
  frame-pruning op: the column count is kept when it still divides, an auto count
  is chosen when it does not, and a count with no workable grid under the texture
  cap is refused rather than written with blank cells. The open document is
  untouched — same frames, same revision, same dirty set, same undo history —
  which is what makes "export a subset, change the selection, export again"
  possible. The sidecar's `edited.ops` gains an `export: 55 of 124 frames` marker,
  so provenance survives without adding a key an engine's loader might not expect.
- **The result panel is a starting point, not a dead end.** It names every
  artifact that was written (sheet, sidecar, preview, gif) and links each one, and
  it carries an **edit** action that reopens the file that was just written as the
  document you are editing. After a subset export that is the whole point: you
  wrote 55 frames because you want to keep working on those 55. The button names
  the count — *edit the 55 saved frames* — so it cannot be mistaken for "reopen
  the sheet I came from", which is exactly how it was misread once.
- **An option's label must not promise what the option is not doing.** The
  subset box's label used to carry the count whenever anything was selected,
  whether or not the box was ticked: an unchecked box reading "only the selected
  55 of 100 frames" was taken as a statement about what Save would do. It wrote
  all 100, and edit duly reopened 100 — reported as "when i click edit not 55
  frames full 100 frames returned". The count now appears only when the option is
  **on**, and a line beside the Save button states the outcome in words in both
  states: *this save writes all 100 frames* / *this save writes the 55 selected
  frames, not the other 45*. Neither the export nor the reopen was ever wrong; the
  wording was.
- **The clip you are about to matte has to be something you can watch.** The
  input panel used to show a still of frame 1, which cannot show a loop, a hitch,
  a dropped frame or a bad cut — the exact things worth checking before a run that
  costs minutes per frame. It is a `<video>` now, with the still as its poster and
  as the fallback when the browser cannot decode a codec the server happily
  reads. The bytes come from `/api/video`, which speaks `Range`: a media element
  opens with `Range: bytes=0-` and answers a seek by asking for the bytes it
  needs, so a server that only ever sends a 200 from byte zero plays once and then
  re-downloads the whole clip on every scrub. It streams rather than reading the
  file into memory (the clip can be hundreds of megabytes and this process is
  also holding the model), answers a range past the end with a 416, and serves
  only known video extensions — the path is the user's own, which the tool already
  reads to decode frames, but that is no reason to hand a browser every `.py` and
  `.json` on the disk through the same door.
- **The preview player is a page in its own right, and nothing tested it.** It
  was a 1100 px column pinned to the top of an otherwise empty document, so a
  566x640 cell sat in the corner of the window — and inside the generator's own
  preview panel it was mostly cut off. Reported as "after generate preview page
  too short". The player now gives the sheet the whole window (`height:100%`, the
  controls beside it, under it on a narrow one) and the zoom slider multiplies
  *the size that fits*, so 1.0 is "as big as the window allows" instead of "100%
  of the source pixels" — a 566x640 cell at 1:1 does not fit a 430 px panel at
  all, which is the size a 1.0 slider used to hand the user.
- **One field doing two jobs in the player's animation loop.** `state.last` was
  both the previous animation timestamp and the last frame index, so the
  end-of-frames test compared a *frame position* against a *millisecond
  timestamp* — never true. Playback walked past the end of the sheet, reported as
  "preview play button not stopping at max frame", and drew nothing once it was
  there: `drawImage` with a source rectangle outside the sheet paints no pixels
  and raises nothing. Recorded from inside the page over 92 ticks, on a sheet
  that has 8 frames: frames 0..35, blank canvas on 70 of them. The timestamp
  (`state.prev`) and the bound (`CFG.count`) are now separate fields, and what
  happens at the bound is the sidecar's `play_mode` — `loop` wraps, `pingpong`
  reverses, `once` stops and puts the button back to *Play*.
- **A cell's URL has to change when the cell does.** Re-loading `cell/7.png`
  after an edit is not a request: the browser answers a URL it has already
  decoded from the bitmap it is still holding — whatever `Cache-Control` says —
  and hands the page back the **pre-edit** cell. Every write reloads the cells it
  changed through exactly that URL, so the pencil, the eraser and *Clear* all
  landed, hashed, journalled and verified correctly on the server while the
  sheet, the frame preview and the timeline went on painting the old pixels. From
  the chair that is the tool doing nothing at all, and it is how it was reported:
  "pencil eraser and clear frame not working". It is also invisible to every
  check this suite had, because they all read the cell back over HTTP (a `fetch`,
  a hash of the response) rather than looking at the page. The URL now carries
  `?rev=<cell_rev>`, so it is a fix rather than a cache-buster: same revision,
  same image, and the revision is what `S.imgRev` already compares against. The
  driver grew four checks for the layer it never looked at — the sheet canvas,
  the preview panel and the timeline thumbnail must show the stroke, and the
  cell's `src` must carry its revision. The last one is the deterministic guard:
  reverting the URL makes it fail every run, where the pixel checks only fail on
  the runs the cache actually answers stale.
- **Snap Pixels drives the snapper server-side, and the UI is per-op on purpose.**
  spritefusion-pixel-snapper ships as a Rust/WASM `process_image`. The pixels the
  editor edits live in numpy on the server, so the module is run through a small
  Node bridge (`snapper_runner.mjs` + `pixel_snapper.py`), one process per batch,
  rather than re-implemented in canvas — the same "one source of truth for
  pixels" rule as every other op. The panel is the one place the registry's
  generated controls are not enough: the point of the op is *seeing* the grid the
  snapper picks (its cut count is content-dependent, so 640÷10 is not reliably
  64), which needs a preview and a preview zoom. The palette is a picker rather
  than a text box: auto k-means, the spritefusion web tool's preset console
  palettes (NES, SNES, Game Boy, PICO-8, …), or a palette imported from a PNG,
  each shown as a swatch and a colour count — the same options as the upstream
  web page. The op appears only while a
  frame is selected and acts on the selection. A cell size is a property of the
  document, so the snap keeps it: the grid result is scaled back up to the cell
  with nearest-neighbour, the frame and the sheet keep their size, the layout is
  never rewritten, and the frames the selection leaves out are left untouched.
  The snapper auto-detects a pixel size *per image* and quantizes each frame to
  its own palette, which would let consecutive frames land on different grids and
  drift between colours. Both decisions are instead fitted once over a sample of
  the selection and applied to every frame (the median detected size, one
  k-means palette), so a selection is temporally much steadier. It is not a hard
  guarantee: the snapper places its cuts on detected edges, so its grid is
  content-elastic and can still shift by a pixel as the subject moves. For an
  animation that must not flicker at all, use Pixelate (mesh) below.
- **Pixelate (mesh) brings in proper-pixel-art, and shares its decisions.**
  Where the snapper works from grid projections, proper-pixel-art recovers the
  grid from geometry — Canny edges, morphological closing, a probabilistic Hough
  transform, an outlier-trimmed median pixel width, then a homogenised mesh — and
  collapses each mesh cell to its dominant colour. It is a Python package rather
  than a WASM module, fetched from its own checkout (`SPRITE_PPA_DIR`) the same
  way the snapper is, and driven through `proper_pixel.py`. The animation path is
  the one used here: the mesh and the palette are fitted *once* over a sample of
  the selection and then applied to every frame, with a single fixed cell map, so
  a multi-frame edit cannot jitter between two grids or two palettes — solving
  each frame alone is exactly the failure the package's video mode exists to
  avoid. Because the mesh is fixed, a part of the frame that does not change comes
  out byte-identical on every frame (the test suite pins this). Like the snapper,
  the cell size is kept: the true-resolution result is scaled back up to it
  (nearest), so the sheet's layout survives. It takes the same **palette picker**
  the snapper has — *"add same palette from snap pixel to pixelate (mesh)"* — and
  the picker is the same control, extracted into `paletteControl()` in the page
  and called by both panels rather than copied, so a preset chosen in one is the
  preset chosen in the other. A named palette *replaces* the fitted one: the
  auto-quantiser is switched off (the mesh's own representative colours are
  mapped onto the named palette instead), because leaving it on would quantise
  twice and could merge away colours the user asked for. The `colours` spinner
  therefore does nothing while a palette is named, which the op-layer suite pins
  by running twice with different counts and requiring identical output.

  **Its two defaults are `colours 0` (keep all) and `mesh-detection upscale 8`.**
  Both are registry defaults, and `coerce_args` fills a missing key from the
  declared `d`, so the honest test is to run the op with **no arguments at all**
  and measure the result — pinning `arg["d"] == 0` would freeze a number and not
  an effect. Measured on the suite's own 192×192 noisy fixture: the defaults
  resolve a **21×20** grid in 30 colours, where the old `16 / 2` pair gave **6×6**
  in 5 colours, and naming `0` and `8` explicitly is byte-identical to omitting
  them (so the declared default really is the value the op receives). The finer
  grid is the point of the higher upscale, and it is asserted as a *direction* —
  more cells than the old upscale resolved — rather than as those exact numbers,
  so it does not freeze a fixture detail. `upscale`'s `max` moved 4 → 8 with it:
  a default above the old ceiling would have rendered
  `<input type=number min=1 max=4 value=8>`, a form the page cannot honour.
  `coerce_args` does not clamp, so nothing would have *failed* — the control would
  simply have been wrong, which is why the max is now asserted to admit its own
  default. Cost is real: ~1.9 s against ~0.1 s on that fixture, so 8 is the
  practical ceiling rather than a starting point to go beyond.
- **Pixelize (outline) brings in PixelOE, and *imposes* a grid rather than
  recovering one.** The two ops above and this one are easy to confuse, so the
  distinction is the point: the snapper and proper-pixel-art both assume the
  image *was* pixel art at some resolution and try to find that grid back.
  PixelOE does not. It makes a grid, in two stages — a contrast-aware outline
  expansion (the LAB luminance is locally median/min/max filtered into a
  per-pixel weight, the image is rebuilt as `erode * w + dilate * (1 - w)` so
  lines thicken where contrast is high, then a closing/opening cleans it up),
  and a contrast-based downscale where each `pixel_size` block picks the
  luminance nearest its centre, its median, mean, min or max and the chroma by
  median — so a block becomes one colour chosen for contrast instead of an
  average. That first stage is the whole idea: without it a one-pixel outline is
  averaged away before the grid that would have preserved it exists. It lives in
  its own checkout (`PIXELOE_DIR`), is driven through `pixeloe_bridge.py`, and
  runs on the **torch** backend only — the Slang compute-shader backends need
  `slangpy`, which is not a studio dependency, so the bridge does not ask for
  them.

  **Two things about this integration are load-bearing, and both cost time to
  find.** First, **the pixel size must divide the cell.** PixelOE
  replicate-pads an image whose dimensions are not a multiple of `pixel_size`,
  and that padding is fed *into* the outline expansion and the colour match, so
  a padded run genuinely differs from an unpadded one and cropping the padding
  back off does **not** recover the unpadded result — a 64×64 cell at pixel size
  6 comes back 66×66 and is simply a different image. The bridge therefore
  refuses a size that does not divide the cell and lists the sizes that do,
  which keeps the cell size exact and the output size invariant true (the op and
  the bridge both refuse; the bridge's shape backstop catches a pad that slipped
  past the guard). This is why `pixel_size` is a **factor of the cell size** and
  not a free number: the default of **2** divides every even cell size the editor
  meets, so the default works on a 64×64 frame out of the box. An earlier default
  of 6 refused on the suite's own fixture, and a default of 8 was the first fix —
  8 also divides a 64×64 cell, but it is a coarse starting point, so the default
  came down to 2 once the request was for a finer block. Second, **PixelOE is
  RGB-only**
  — it has no alpha channel and every stage works on luminance and chromaticity
  — so alpha is stripped before the call and put back afterwards, byte for byte,
  and the op is described as what it is rather than as a transparency tool. The
  colour *under* a transparent pixel is invented by the algorithm; that is inert
  for straight-alpha rendering, but it is not a claim the op makes good on.

  `thickness` of 0 turns the outline stage off; `contrast` is the paper's
  downsampler and `k_centroid` / `lanczos` are the package's others (the rest
  are plain interpolation modes, and `nearest` is what one of those gives).
  **The defaults are the plain path.** Out of the box the op is `pixel_size 2`,
  `thickness 0`, `k_centroid`, `colours 0`, `dither none`, `sharpen none`,
  `sharpen amount 0`, colour match on: a block downsample that keeps every
  colour and does *not* expand outlines or sharpen. That ordering matters,
  because the two stages that make PixelOE distinctive are also the two that
  change the art most — outline expansion deliberately thickens edges and the
  unsharp/laplacian pass deliberately overshoots them, so both are opt-in and
  the help text says so. The suite pins them as a **set**, not as a list of
  values: every declared default is asserted, the sharpen amount's own bounds are
  asserted to admit its default, and the whole set is run on the fixture to prove
  it is a runnable combination (an earlier cut had a default of `sharpen
  unsharp` + `sharpen amount 2`, which the off-state checks then had to be
  rewritten around — the "off" spelling is the one that stays stable).
  **Every "off" is a named option, not a blank one.** `sharpen` offers
  `none / unsharp / laplacian` and `dither` offers `none / ordered /
  error_diffusion`, each with `none` first as its declared default. The first cut
  used an empty string for off, which is the trap: the page builds one `<option>`
  per entry with `textContent = op`, so `""` renders as an **unlabelled row** in
  the dropdown — it works, and it reads as a bug, which is how it was reported.
  The op still maps both `""` and `"none"` to "off", so a stale page cannot turn
  sharpening on by accident, and the suite pins that the two spellings agree
  *and* that a real mode changes the image — otherwise the equality would also
  pass with sharpen wired to nothing.
  `colours` above 1 enables k-means quantisation and then a dither; a named
  count is a **target, not a guarantee**, because the colour match that follows
  quantisation can put a shade back — the op's help says so, and the suite pins
  the honest version (it lands on a small palette, rather than asserting an
  exact count that the package does not promise). It also takes the **palette
  picker** the other two ops have — auto / console preset / PNG import — and the
  same rule applies: **a named palette replaces quantising**, `colours` is
  ignored, and the pixelized result is mapped onto the named palette, because
  leaving the quantiser on would quantise twice and could merge away colours the
  user asked for. No page code was needed for it: the picker is offered to any
  `text` arg named `palette` by `opNode`, so declaring the arg is the whole
  integration, and the same control therefore appears in all three panels. The batch goes through in one
  call, so two identical frames come out byte-identical and a multi-frame edit
  cannot jitter — the same rule the two mesh ops follow. Measured cost is small:
  ~25 ms for one 64×64 frame, ~37 ms for a batch of three, so this is not a slow
  op and has no preview panel of its own; it uses the registry's generated
  controls, like Pixelate (mesh).
- **Per-frame audio ships with the sheet, not buried in the app.** A clip is
  attached to a frame (the Audio panel; the target is the selection, else the
  current frame) and staged under `uploads/` while the document is open. On save
  the clip is copied to `audio/` beside the sheet — `save_doc` refuses to write a
  sidecar that points at a file that is not there — and the sidecar gains an
  `audio` list, `{frame, file, name, volume}`, with `file` relative to the JSON.
  So an engine loads the JSON, the sheet and the `audio/` folder together and
  plays the clip whose `frame` it is on; the private upload staging path never
  reaches the artifact. Reopening the saved sheet resolves the clips from beside
  it, and exporting a subset keeps only the clips on exported frames, renumbered
  to their new indices. No `audio` key is written when there is none, so a sheet
  with no sound has the same sidecar it always had.
- **The sidecar JSON is the document, and opening goes through it.** It names the
  sheet, the grid, the name and the clips, so opening resolves the PNG from it
  (the `sheet` field, then `<stem>_sheet.png`, then `<stem>.png`, and it says what
  it tried when none of them is there). Opening a path that points at the PNG
  still works — the sidecar is then found beside it — but opening the JSON is what
  guarantees a saved sheet's sounds come back with it, instead of depending on a
  directory scan picking the right JSON out of a folder. Pointing the open box at
  a `.json` used to reach PIL with it and fail with `cannot identify image file`.
  Pointing it at a **folder** works too: the folder is resolved to the sidecar in
  it (the one named after the folder first, then any JSON whose grid and sheet
  both check out), because typing the folder a sheet lives in is at least as
  natural as naming the JSON inside it. A folder that holds a sheet but no
  sidecar is refused rather than guessed at — the grid would have to be invented.
  **A sidecar is searched for even when a grid is supplied.** The columns and rows
  boxes that used to stand in for a missing sidecar are gone from the panel, but
  the API still takes them, and the search used to be gated on them being absent —
  so supplying a grid silently skipped the sidecar, and the sheet opened with none
  of its clips. Measured on a real run sheet: the PNG alone returned its 2 clips,
  the PNG with its own 11×3 grid supplied returned 0. An explicit grid is a
  fallback for a PNG that has no sidecar, not a request to ignore the one beside
  it.
- **The picker lists the folders you have opened, not every sheet on the server.**
  The editor is used by giving a path, and a run writes a fresh folder every time,
  so the last ten folders a document was opened from are worth more than a scan of
  the whole library. Newest first, one entry per folder (compared
  case-insensitively, because the filesystem is), capped at ten, kept in
  `localStorage` so it survives a reload; the row is labelled with the folder, and
  clicking it reopens that folder. `/api/sheets` is still there and still covered,
  and the row click falls back to the exact file that was opened when the folder
  alone will not resolve — a sheet opened by PNG with an explicit grid has no
  sidecar in its folder to find.
- **Clips are attached with a file dialog; the folder library is gone.** The Audio
  panel's *Add sound…* takes files, uploads them, and the server stages them
  exactly as before. The panel that used to sit above it — point the editor at a
  folder of clips, list them, drag one onto a frame — was removed, and the
  drag-and-drop went with it rather than being left in place unreachable: every
  drop target was gated on the library's own drag state (`ALIB.dragging`), so
  without the list there was no drag source and no drop could ever fire. The
  routes behind it (`/api/editor/audio_lib` and `/audio/from_path`) are still
  there and still covered by the smoke suite, so restoring the panel would be a UI
  change and nothing else — but nothing in the editor calls them now. Every panel
  in the left column except Audio starts folded: the sheet picker is used once a
  session and the palette is long, while clips are attached repeatedly, so the
  panel that gets used is the one left open.
  A frame that has a sound is marked **on the cell itself**, in
  the corner the frame number does not use (the number is drawn top-left, the
  note top-right), so "does this frame make a sound" is answerable on the sheet
  and not only in the Audio panel — the timeline's badge is at the bottom of a
  46px thumbnail and says nothing about the frame you are looking at. Both marks
  appear only when the cell is big enough on screen to read them, from the same
  threshold the frame numbers already use.
- **The player has its own fps and speed; the document has its own fps.** Two
  pairs, and they do not move each other. The **Grid** panel's `fps` is the
  document's — Apply metadata writes it into the sidecar as `"fps"`, and that is
  the number the export and the preview player both read. The **Playback** panel's
  `player fps` and `speed` are the player's own: they set how fast the preview
  plays and reach nothing else, and neither is in the JSON (asserted, not just
  labelled — if someone wires the speed multiplier into the sidecar the label
  becomes a lie and nothing else would notice). They used to be one control: the
  playback loop read the Grid panel's `#fps`, so changing the document's frame
  rate silently changed how the preview played, and there was no way to preview a
  24 fps sheet at 12 without editing the document's own rate. The player's fps
  starts at the document's on open — a 12 fps sheet previews at 12 rather than at
  a hardcoded 24 — and stops following the moment the box is typed in; opening
  another sheet hands it back. Ownership is marked on the first keystroke rather
  than by comparing values, so typing the number the document already has still
  counts as claiming it. Speed is a multiplier on the player's fps, not a third
  rate. Guarded by driving the real playback loop rather than reading the boxes:
  with the player at 240 fps and the document at 1, `once` mode plays the
  124-frame sheet through inside a second; with the player at 1 and the document
  at 240 it does not. A boolean, not a frame count, because a count depends on how
  long the sleep actually took and wraps at 124.
- **Apply metadata writes the JSON — it is the one panel that writes a file.**
  Every other op is a change to the in-memory document: nothing persists until
  Save, and Save rewrites the sheet PNG, the GIF and the preview player along with
  the JSON. So metadata that only lived in memory was the surprise. Apply metadata
  does the narrow thing instead — `set_meta`, then
  `POST /api/editor/{did}/save_meta`, which updates the sidecar the document was
  loaded from, in place and atomically. It is a **targeted update, not a
  regeneration**: only the keys the panel owns (`name`, `fps`, `play_mode`,
  `anchor`, `blend`, `animations`, `note`) are written, so `source`, `matte`,
  `crop`, `sheet`, `edited` and `audio` survive verbatim. Regenerating would
  replace `edited.ops` with this session's journal and quietly cost the user the
  provenance of their own sheet. It refuses when the JSON and the document no
  longer describe the same sheet — a grid change (the PNG is not rewritten, so the
  sidecar would claim a geometry the PNG on disk does not have) or a changed clip
  list (the new clip has not been copied beside the sheet) — and both refusals
  point at Save. Refusals reach the panel: the status line reads
  `metadata NOT written: …` and the document keeps the metadata it was given.
- **A non-2xx response is an error, whatever the body says.** `api()` used to
  decide only on the body: throw when it sees `{ok: false, error}`. A route that
  is not there answers `404` with FastAPI's `{"detail": "Not Found"}`, which has
  neither key — so the check walked past it, the caller got an object with no
  fields, read `undefined` out of it and reported success. That is not
  hypothetical: a test server started before the `save_meta` route existed made
  Apply metadata announce it had written a file it had never touched, and only the
  assertion that the status line *names the path* caught it. Every route in this
  app answers 200 when it means yes, so the HTTP status is now part of the check.
- **A checkbox's words are part of the checkbox.** Every row like this was a
  `<div class="chk"><input type="checkbox"><span>words</span></div>`, and in a
  `<div>` the words are not part of the control: clicking them does nothing at
  all. Reported as *"allow overwriting the sheet this document was loaded from
  when checked not working"* — and the box itself was fine, measured with a real
  click: ticked it saves over the source, unticked it refuses. What was broken is
  that the text is what a user aims at, so the box never got ticked, the save
  refused, and the opt-in looked dead. They are `<label class="chk">` now, which
  makes the whole row toggle the input with no handler, and the op registry's
  `bool` controls are built the same way (Nudge's *wrap around* had it too). The
  audio panel's *play with preview* already used a `<label>`, which is how the
  inconsistency survived: one row was right and six were not. Guarded by clicking
  the **words** with a real mouse and asserting the box flips — every earlier
  interaction with these controls set `.checked` in JavaScript, which proves the
  handler reads the box and says nothing about whether a mouse can reach it.
- **The overwrite box is permission, so it decides the destination only when no
  destination was named.** Ticked with the output folder left blank (its default),
  the save writes the files the document was loaded from; ticked with a folder the
  user typed, that folder still wins. Both halves matter: without the first, a
  blank folder sent the save to a brand-new `runs/editor_<stamp>/` and the loaded
  JSON was never touched — *"overwriting not working fix taht"* — and without the
  second, an explicit folder would be silently ignored and the original replaced
  when a copy was asked for. `save_doc` takes an `out_dir_named` flag from the
  route, and the two `overwrite-*` mutation hooks are a pair because each one
  alone is satisfied by a wrong implementation.
- **A row behind a `display:none` ancestor cannot be opened, only revealed.**
  Snap Pixels' checkbox lives in the Pixel art group, which the page hides until a
  document is open and a frame is selected. `display:none` ignores `open`, so a
  zero-rect row means one of three different things — folded, hidden, or outside
  the scroll viewport — and the fix differs for each. Ask which ancestor is
  `display:none`; `_probe_chk.py` prints exactly that.

## What it will refuse to do, and why

These are not bugs. Each one is a real failure mode that produces a sheet which
loads fine and then misbehaves in an engine:

- **A grid that is not full.** `columns × rows` must equal the frame count. Unity's
  Flipbook node and the Particle System's Texture Sheet Animation both assume a
  packed grid and will play the empty cells as blank frames. So `columns` must be
  a divisor of the frame count, and the UI only offers divisors. The editor
  enforces the same rule on every re-grid, and re-derives a valid grid when an
  operation changes the frame count.
- **A sheet over the 32768 px 2D texture limit** (the cap this tool ships with;
  D3D12, Unity and Godot stop at 16384, so half of that range is only safe on
  hardware that reports the larger figure). The error names the smallest cell or
  the column count that would fit rather than just failing. A subset export
  re-derives the grid for the frames it is given, and it keeps the document's own
  column count only when the sheet that count makes is *legal* as well as
  divisible — the sheet just written may be the illegal thing, since a document
  opened before the cap was raised still describes a grid that is over its own
  cap. That matters because the refusal it replaced looked like a bug and was not:
  *58 frames have no column count that fits* is true for 58 frames of 566x640
  under 16384 (the closest grid, 29 x 2 = 16414 x 1280, is 30 px over, and 1 x 58,
  2 x 29 and 58 x 1 are all worse), so only a bigger cap could save that export.
- **A crop that would clip the subject.** The window is derived from the measured
  subject bbox and asserted to contain it.
- **Padding past the frame edge.** A crop cannot reach outside the source, so
  `cell_w`/`cell_h` are clamped to the source size. Without this, a character
  whose feet sit on the last row silently gains transparent rows and its anchor
  shifts.
- **A 90°/270° rotation of a subset of non-square cells.** It would change the
  cell's aspect and produce a ragged sheet, so it is refused unless the whole
  sheet rotates (which swaps the grid's cell dimensions).
- **A dedupe that would collapse the sheet to one frame.** That is not an
  animation; the error says so.
- **Dropping every frame.** *Drop selected frames* takes the selection as its
  input, but for every other operation an empty selection means "all cells". Read
  that way here it would delete the animation, so an empty selection — and a
  selection that would leave fewer than 2 frames — is refused with that
  reasoning spelled out. Use *keep a frame range* if you mean "delete everything
  outside this window".
- **Keeping fewer than 2 frames.** Same rule, from the other direction.
- **Writing over the sheet it was loaded from.** Saving into the source folder
  under the source name would destroy the original matte the edit came from, with
  no undo and no visible symptom until an engine reads the new sheet. The refusal
  names the file it would have written and the `overwrite` flag that means it on
  purpose; the UI's box for that is unchecked by default.

## Verification

The gate reads only the finished sheet and the frames, and checks five invariants —
the fifth only bites when the colour key was on:

| | check |
|---|---|
| I1 | **straight alpha** — where alpha > 0, cell RGB equals the frame's RGB |
| I2 | **no backdrop leak** — no near-black pixel reachable from the cell border is left opaque |
| I3 | **full grid** — size, fullness, under the texture cap |
| I4 | **subject not eroded** — bright pixels keep their alpha |
| I5 | **no key spill on the rim** — no rim pixel whose key channel still dominates (the rim is the despill's own reach: band plus soft alpha) |

Two thresholds are load-bearing and documented in `pipeline.py`:

- **I2 uses `alpha >= 200`, not `alpha > 0`.** A soft anti-aliased edge
  legitimately has near-black pixels at low alpha. Measured on a VRMBG matte, 86%
  of the `alpha > 0` hits were at alpha ≤ 32 (median 4, none at 255) — that is
  fringe, not leak. A real blob is ~93% at alpha ≥ 200.
  I2's own criterion is *reachable from the cell border*, the same one the flood
  fill uses while its box is ticked. Untick it and the pass reaches inside the
  silhouette, where I2 does not look — by design, since that black may be the
  subject's own outline. The cleared-pixel count in the log is the only statement
  about what was actually removed.
- **I4 tolerates a small hole.** A matting model always drops a few edge pixels —
  typically a 1-px seam of ~15 px. It fails on a hole ≥ 24 px or ≥ 0.5% of the
  subject. The raw counts are always printed, so a passing run still shows them.

**I5 is the fifth check, and it is a real invariant when the colour key is on.** It
counts rim pixels whose *key channel still dominates* — green, for a green key,
within `KEY_EDGE_BAND` px of transparency — and fails the cell if there are any.
It is deliberately not a distance from the picked colour: that is precisely the
measurement that cannot see the leak. The 00183 outline is 194-233 away from every
green key, so a tolerance-based check reports zero while the sprite is visibly
green to the eye, and raising the tolerance till it matches deletes the subject
(the run history on that clip goes 254k → 62k opaque px and still shows green).
Channel dominance is what the eye catches, and after the despill the pass runs it
is an invariant: the key's channel is clamped to `max(the others)` everywhere on
the rim. With the key off the count is informational, because failing every run
over a backdrop nobody asked to key is the same mistake as defaulting `do_repair`
to on. The evidence that the repair *ran* is the stage's own cleared-pixel and
despilled-pixel counts, and a key that matched nothing says so in as many words
instead of reporting success.

`test_verify.py <run_dir>` mutation-tests the gate: it plants a premultiplied
sheet, a backdrop blob, a 30×30 hole, a 1-px seam, an empty cell and a wrong
frame count, and asserts each is (or is not) flagged. **6/6 expected — on a run
whose pristine sheet passes.** Pass it a defective run dir and you get 5/6: the
baseline already fails I4 (`runs/0920-154626_…` reports 116 cells damaged, 14116
bright px lost, holes to 100 px), so the planted 1-px seam sits on top of existing
erosion, merges with it into a hole over the 24-px tolerance and is flagged. The
suite exits 1 either way, so check the pristine verdict before reading the 5/6 as
a verifier bug.

`test_matte_repair.py` covers both repair passes on fixtures built to contain the
failures their guards exist to prevent. **67 checks, 0 failures.** The colour-key
fixture has a backdrop-coloured rim around a neutral subject *and* an opaque
backdrop-coloured blob inside it; run it with `band=10**9` and the blob is erased,
so the band is observed to be load-bearing rather than assumed. The flood-fill
fixture has a near-black leak touching the frame border *and* a near-black patch
enclosed by the subject: the default clears exactly the leak and the patch
survives, unticking the box clears both and the difference is exactly that patch.
Its third fixture separates the key's two guards: a slab of backdrop left along
the frame edge, 12 px wide and 20 deep, where the band takes the outer shell and
leaves the core while the connectivity test takes the whole slab, and an enclosed
key-coloured blob that neither may touch. A fourth is the spill: an opaque
`(0,50,0)` outline on the silhouette that the key cannot reach at any tolerance,
a dark patch of the same colour deeper than the band that must survive, and a
red-tinted rim pixel that must not be touched — plus that the clamped channel is
whatever the *picked* colour's channel is, with a blue key on a blue slab.
It also checks the boundaries that are not taste — `key_rgb` sends a form typo to
the default rather than to black (a black key would quietly delete every dark
pixel on the rim), and a frame with no transparency at all is returned untouched
instead of being erased whole.

### Erase a colour — the colour comes from a picker, not three numbers

*"matte repair erase a color add a color picker"*. The op used to declare three
int arguments (`r`, `g`, `b`), which the generator dutifully turned into three
spin boxes — so keying a backdrop meant knowing its numbers, and nobody can name
their green screen by eye (the walk clip's is `#13ff38`, not `#00ff00`). It now
declares **one `color` argument**, which the generator already knew how to render
as `<input type="color">`: the type existed and nothing used it. The swatch opens
the system picker, and the wire carries one hex string.

**Fill / tint was the last op still on channel boxes**, and it is worth a note
because of *why* it survived: it had no callers and no tests, so nothing failed
while `erase_color` moved and it stayed behind. It now declares the same single
`color` argument, defaulting to `#ff0000` — the `(255, 0, 0)` the three boxes
defaulted to, so a saved recipe that calls `fill` with no arguments does not
silently change colour. The colour is parsed even in `alpha_only` mode, where it
is unused: a value the op cannot read is refused rather than ignored, so a typo
can never look like a setting that had no effect.

The page has one picker, not two. Section 16p does not check "Fill / tint has a
colour input" — it asks the page's own registry which ops declare a `color`
argument and requires **every** one of them to render exactly one picker showing
the declared default. A new op that declares a colour is covered without anyone
remembering to add it, and an op that quietly goes back to channel boxes fails.
That is the shape this class of request should have taken from the start: the
first two ops were converted one at a time, by hand, and this is what stops the
third from being missed.

An unreadable value is **refused**, not defaulted. The editor's ops refuse bad
input with a reason (`set_columns` on a non-divisor, `match_size` on a selection
with no subject); the fallback for a colour would be black, and black is the one
value a keyer must never guess — it would take out every dark pixel on the
subject's own outline and still report success. `pipeline.key_rgb` guards the same
failure from the other side: it falls back to the backdrop green rather than to
black, because over there a typo must not stop a batch, while here the user is
looking at the frame and can fix it. `hex_rgb` also accepts `#abc` shorthand and
the hash-less form — not because the picker produces them (it always emits
`#rrggbb`) but because the route is reachable by hand and by any future saved
recipe, and rejecting a colour that is unambiguous would be arbitrary.

Three hooks, one per way this can go wrong: `hex-rgb-swapped` reads red and blue
back to front (invisible on any grey fixture — only a colour whose channels
differ shows it, and a backdrop is exactly that), `hex-rgb-falls-back-to-black`
replaces the refusal with a default, and `erase-color-ignores-picker` keys black
whatever the control says — which the user cannot see at all, because black *is*
the default value.

`hex_palette` is the same reader for a list — `#ff0000,00ff00` — because the
palette picker on Snap Pixels and Pixelate (mesh) writes a comma-separated hex
list into a hidden input. It follows the same rule and for the same reason: an
entry that cannot be read is refused, never dropped, because a shortened palette
is not the palette the user named and the run would still report success. Blank
entries *are* skipped, since a trailing comma is punctuation rather than a
colour. The refusal message itself is generic — `colour 'oops' is not a hex
colour like #13ff38` — because two ops now raise it and an op-specific tail
("so there is nothing to erase") would be wrong on the other one; the rationale
it used to carry lives in the docstring instead.

### Match content size — *"some art inside cells are little small some little big this couse a jump in the loop"*

One button, and the loop stops pulsing. Select every frame, press **Match content
size** under Transform, and each frame's subject is scaled to the same size. The
cell size, the grid and the frame count are untouched — only the art inside the
cells changes — so nothing downstream has to be re-aligned or re-exported at a
different resolution.

The part that is not obvious is the **pivot**, and it is the whole reason this is
not a one-line `cv2.resize`. Scaling every frame about the *cell* would fix the
size jump and introduce a position jump: a character standing on the floor of its
cell would rise and fall as its height changed, and an alignment the user had
already run with **Align pivot** would be quietly undone. So the op scales about
a point *on the subject* — the `anchor` control, defaulting to the feet — and
then measures that anchor again on the result and translates it back if the
resample moved it. `scale_about` places its output on an integer offset, so a
fractional factor leaves the scaled pivot on a half-pixel and the soft edge an
interpolating filter leaves can round it the other way; without the second
measurement two frames of a four-frame test fixture drift one pixel and the loop
jitters. A translation cannot change any size, so this costs nothing the op is
trying to fix.

The reference size is the **median** of the selection, and the choice is the
point: a matte that leaked one stray opaque pixel inflates that frame's box, so a
mean drags the whole set toward the bad frame and a max makes every good frame
grow to match it. The median moves less, and the op reports the factor range it
needed so an outlier is visible rather than contagious. If a frame's number looks
wrong, raise the **alpha threshold** — a stray opaque pixel is what inflates a
box. `max` is offered for the case where nothing may be shrunk, `first` for
matching a frame the user is looking at.

What it refuses to hide: the status line reports the size actually achieved
(`now 28` when the target was 27 and an interpolating filter widened the soft
edge — `resample: nearest` lands on 27 exactly), how many frames were **clipped**
at the cell edge, and how many had **no subject** and were left alone. A subject
already taller than its cell cannot be matched without losing the top of it, and
that is said out loud rather than silently done. A selection with nothing
measurable in it is refused with a reason.

Three mutation hooks, one per property, because each can fail alone:
`match-size-pivot-ignored` scales about the cell centre (sizes still come out
right — only the positions fail), `match-size-anchor-not-restored` drops the
post-scale correction (only *some* frames drift, which reads as "close enough"
until the loop is played), and `match-size-scope-ignored` acts on the whole
document instead of the selection (invisible in the one case the user described —
select all — and wrong in every other one). `match-size-scope-ignored` was a
**no-op** in its first version: it rewrote only the second of the op's two
`for i in idxs:` loops, and the loop body opens with `if i not in boxes:
continue`, so the extra iterations all skipped themselves. It reported MISSED,
which reads as "the suite has a hole" when the truth was "the mutant was never a
mutant". A hook has to be shown to alter behaviour before its being caught means
anything.

`test_editor.py` covers the editor's op layer — **462 checks, 0 failures** —
including the shift/place geometry, the un-premultiply inverse, the topological
colour key, the align solver, undo/redo across a frame-count change, the loaded
reference surviving an undo, the frame scoring, the reference-free premultiply
smell (including the stray-transparent-RGB regression), save refusing to clobber
its own source, the ticked save writing back to the files it was loaded from,
the subset export leaving the document it came from untouched,
`save_meta` writing the metadata into the sidecar it was loaded from without
regenerating the rest of it, an attached clip's volume being changeable *after*
it is attached and reaching the sidecar as the number it was set to, the
nearest-neighbour resize being an exact inverse of itself (a cell doubled and
halved again is byte-identical, and no colour is invented on the way), and that
every one of the 44 registered operations executes (or refuses for a stated
reason).

It also carries **mutation hooks**, because a guard that has never been observed
to fail is not a guard. `python test_editor.py <name>` breaks one guard and the
suite must then fail — all thirty-eight are currently caught:

| mutation | what it reverts to | caught by |
|---|---|---|
| `undo-naive` | the redo stack gets the pre-op entry, so redo rewinds twice | `redo re-applies the exact post-state` (+1) |
| `pad-shift` | `pad_cell` uses `_shift`, which cannot grow an array | `the corner block survived padding whole, not clipped` (+1) |
| `pivot-fractional` | a geometric half-pixel pivot, rounded per cell | `pivots are integer pixels, not half-pixels` (+7) |
| `erase-tonal` | a tonal colour key instead of a topological one | `the enclosed dark patch survived` |
| `verify-ignore-leak` | I2 stops failing | `verify FAILS on an opaque border-connected black backdrop` (+1) |
| `store-post-n` | the undo snapshot is sized from the post-op frame count | `undo restored the frame count` — raises `KeyError: 5` |
| `setcells-nonatomic` | `_set_cells` skips the rollback when a re-grid is refused | `and the grid still matches the frame count` (+3) |
| `restore-drop-orig` | `_restore` leaves the loaded reference alone | `the reference is back to 16 before the next op` (+2) |
| `smell-counts-transparent` | the smell counter counts `alpha == 0` pixels too | `stray RGB in a fully transparent pixel does not silence the line` (reports 3, not 4) |
| `save-clobbers-source` | `save_doc` skips the source-path comparison | `saving back over the loaded sheet is refused` (+1) |
| `export-mutates-doc` | the subset is applied to the live document | `the open document still has all its frames` (+3) |
| `export-unsorted` | the frame order is left as the set iterated | `the subset is sorted and de-duplicated` (+3) |
| `export-regrids-whole` | the whole-document shortcut is dropped | `selecting every frame returns the document itself` |
| `open-folder-ignored` | `load_doc` does not resolve a folder to the sidecar in it | `a document opens from the folder that holds the sidecar` — raises `PermissionError` reading the directory as an image |
| `grid-skips-sidecar` | an explicit grid suppresses sidecar discovery again | `giving columns and rows does not stop the sidecar being found` (+1) |
| `set-meta-ignores-fps` | `set_meta` drops `fps` from its key loop | `set_meta writes fps onto the document` (+7) |
| `save-meta-drops-provenance` | `save_meta` regenerates the file instead of updating it | `a key the panel does not own is carried through, not regenerated` (+2) |
| `overwrite-ignores-source` | the redirect to the loaded files is dropped | `ticked with a blank output folder, the sheet goes back to the file it came from` (+2) |
| `overwrite-ignores-named-folder` | `and not out_dir_named` is dropped, so a ticked save redirects even when a folder was named | `ticked with a folder named, that folder is still where it goes` (+1) |
| `volume-not-clamped` | `set_audio_volume` stores whatever it is handed | `a volume above full is clamped, not left out of range` (+2) |
| `match-size-pivot-ignored` | `match_size` scales about the cell centre, not the subject's anchor | `every subject is on the pixel it was on` (+2) |
| `match-size-anchor-not-restored` | the post-scale anchor correction is dropped | `every subject is on the pixel it was on` (+1) |
| `match-size-scope-ignored` | `match_size` measures and rescales the whole document, not the selection | `frames outside the selection are left exactly as they were` (+1) |
| `hex-rgb-swapped` | `hex_rgb` reads red and blue back to front | `hex parses to the channels a keyer needs` (+3) |
| `hex-rgb-falls-back-to-black` | an unreadable colour becomes black instead of being refused | `an unreadable colour is refused rather than falling back to black` (+7) |
| `erase-color-ignores-picker` | `erase_color` keys black whatever the colour control says | `the colour the picker names is the colour that gets erased` (+2) |
| `hex-palette-drops-bad-entry` | `hex_palette` skips an entry it cannot read instead of refusing | `an unreadable palette is refused: '#ff0000,oops'` |
| `pixelate-mesh-ignores-palette` | the parsed palette is never applied to the mesh output | `every colour in the result comes from the named palette` (+2) |
| `pixelate-mesh-quantises-twice` | the auto quantiser is left on underneath the named palette | `the colours spinner is ignored when a palette is named` |
| `mesh-defaults-reverted` | `colours` goes back to 16 and `upscale` to 2 (with its old max of 4) | `pixelate (mesh) defaults to 0 colours (keep all) and upscale 8` (+4) |
| `fill-ignores-picker` | `fill` tints red whatever the colour control says | `the colour the picker names is the colour that lands` (+3) |
| `resize-blends` | `_resize_nn` becomes a BILINEAR `Image.resize` | `scaling up invents no colour` — reports `(230,230,230,191)`, `(228,228,228,48)` and `(232,216,216,255)`, none of them on the source art (+3) |
| `resize-drops-remainder` | the divisibility guard is removed, so the stride truncates and still reports success | `a factor the cell does not divide is refused, not truncated` (+3) |
| `pixeloe-pads` | the bridge's divisibility guard is removed, so PixelOE replicate-pads a cell it cannot divide and reports success on a different image | `a pixel size that does not divide the cell is refused with the sizes that fit, not by the shape backstop` — the backstop fires with `PixelOE returned 66x66 for a 64x64 cell at pixel size 33` |
| `pixeloe-drops-alpha` | the alpha carry-through is replaced with a flat opaque `255`, so the RGB result is still blocky pixel art of the right size but every transparent pixel is lost | `alpha is preserved byte for byte` — `max\|delta\|=255` |
| `pixeloe-quantises-twice` | the quantiser is left on underneath a named palette, so the result is quantised to PixelOE's own fitted palette and then mapped onto the user's | `the colours spinner is ignored when a palette is named` — caught by exactly that one check, because every "colours come from the palette" check passes under it (both paths end by mapping onto the user's) |

Note that `pixeloe-quantises-twice` does **not** depend on where the `sharpen`
default sits: it mutates the `colors=0 if pal is not None else want,` line only,
and the check that catches it names a palette and varies `colors`, which is a
different control from the sharpen pair. That is deliberate — when the defaults
moved to the off state, the sharpen off-state checks were rewritten, and a
mutant whose only catcher was one of those rewritten lines would have gone
MISSED without anything being wrong with the guard.

`store-post-n`, `setcells-nonatomic`, `restore-drop-orig`,
`smell-counts-transparent`, `save-clobbers-source`, `open-folder-ignored`,
`grid-skips-sidecar`, `set-meta-ignores-fps`, `save-meta-drops-provenance`,
the two `overwrite-*` hooks, the three `export-*` hooks, the three
`match-size-*` hooks, the three colour hooks, the three `hex-palette-*` /
`pixelate-mesh-*` hooks, `fill-ignores-picker` and the three `pixeloe-*` hooks are
derived from the live
source
with `inspect.getsource` and a targeted replacement, so a mutant cannot drift
away from the implementation it is testing; if the function changes shape the
hook exits 2 rather than silently testing nothing.

The two bridge-level `pixeloe-*` hooks patch the module object in `sys.modules`,
not a fresh
import, because the op does `import pixeloe_bridge` *inside* the function body —
rebinding an attribute on a different module object would replace nothing the op
ever looks at, and the sweep would report MISSED (a hole) instead of CAUGHT. The
third, `pixeloe-quantises-twice`, mutates the *op* instead and is
`inspect.getsource`-derived like the rest. `pixeloe-pads` is also why the
op-level refusal test treats "refused by the
shape backstop" as a **failure**, not a pass: under the mutant the guard is gone
and only the backstop catches the pad, which is a worse refusal (the user is not
told which sizes fit) — waving it through would make section 32 pass under a
mutant that removes the guard it claims to test.

`mesh-defaults-reverted` is the exception, and deliberately so: the thing it
reverts is a *registry default*, not a line of code, so it edits
`OPS["pixelate_mesh"]["args"]` in place instead of rewriting source text. That is
enough to be a real mutant because `coerce_args` fills a missing key from
`spec["args"]` and the ops route serialises the same list — so the page renders
the old defaults and the op receives them, exactly as if the source had been
reverted. It fails five labelled checks rather than one, because a default is
observable from several directions at once (the declared value, the colour count,
the colour ceiling, the byte-equality against an explicit `0/8`, and the grid
resolution). It also has to verify it found both args, since a rename would
otherwise leave it mutating nothing and reporting MISSED.

The two `resize-*` hooks are hand-written rather than derived from source, and
each carries the same "did it apply" guard for the same reason. `resize-blends`
rebinds `_resize_nn` to a BILINEAR `Image.resize`, so it first reads the op's own
source and exits 2 unless the op still calls `_resize_nn` by name;
`resize-drops-remainder` replaces the registry entry, so it exits 2 unless
`resize_pixels` is in the registry at all. Without that, a rename would leave the
hook rebinding a name nothing reads, the suite would pass, and the sweep would
report MISSED — which reads as a hole in the suite when the truth is that the
hook stopped applying. `resize-blends` is also the mutant that proves section 30
is about exactness and not about a number: a size assertion passes against a
blended resize, and this one fails it with the invented colours spelled out
(`(230,230,230,191)`, `(228,228,228,48)`, `(232,216,216,255)`).

`export-mutates-doc` is the reason the sweep tool exists. Its regex required
`return sub, idx` to follow `_set_cells(sub, ...)`, and when the per-frame audio
feature inserted the clip carry-over between those two lines the regex stopped
matching. The hook printed "did not apply", which read as noise rather than as a
failure, and it tested nothing for a whole session while this README went on
claiming it was caught. It now anchors on the `_set_cells(sub, ...)` call alone,
and `sweep_editor_mutants.py` runs every hook and classifies it by exit code —
**0 MISSED** (the suite passed, so the guard is not covered), **1 CAUGHT**,
**2 BROKEN** (the hook could not apply, so it tested nothing). A hook that cannot
apply is not a pass. The sweep reads the hook names out of the suite rather than
from a list here, so a new hook is swept without anyone remembering to add it.

The two `overwrite-*` hooks are a pair on purpose, and each one on its own would
be satisfied by a wrong implementation. `overwrite-ignores-source` alone passes
if the save *always* redirects to the loaded files — which would silently ignore
a folder the user typed and replace their original when they asked for a copy
somewhere else. `overwrite-ignores-named-folder` alone passes if it *never*
redirects, which is the reported bug. Only both together pin the actual rule:
the box decides the destination only when the caller named none.

Two of those are worth calling out because of how they were got wrong first.
`set-meta-ignores-fps` patches `E.OPS["set_meta"]["fn"]` rather than
`E.op_set_meta`: the `@op` decorator stores the function in the registry at
import time and `run_op` dispatches through it, so rebinding the module
attribute does nothing at all — the suite passes and the sweep reports the
mutant as MISSED, which reads like a weak guard rather than an unapplied
mutation. And `save-meta-drops-provenance` drops only the keys the panel does
not own; its first version emptied the sidecar, which made `load_doc` raise and
the suite die with a traceback instead of a `FAIL` line naming the property.

`store-post-n`, `restore-drop-orig` and `open-folder-ignored` are caught by an
uncaught exception (`KeyError: 5`, `IndexError`, `PermissionError`) rather than
by a named assertion — the suite exits 1 with a traceback, which is a failure but
a cruder one. The other twenty-eight fail on
a labelled check.

`pixelate-mesh-quantises-twice` is worth a line of its own, because it is caught
by **exactly one** check. Leaving the auto quantiser on underneath a named
palette is invisible to every assertion about *which* colours came out — both
paths end by mapping onto the user's palette, so the colours are from the palette
either way. The only thing that sees it is the pair of runs with different
`colours` counts that must come out identical. One check is a thin margin, and
that is the point: it is the only observable consequence of the decision, so it
is the check that had to exist.
`smoke_editor_api.py` is the HTTP integration pass — **209 checks, 0 failures** —
opening a real 124-frame sheet, running real ops, verifying, undoing, redoing,
saving, exporting a subset, and then checking the files on disk. It covers every
way to open a document — the PNG, the sidecar JSON, and the folder that holds
them — including the exact payload the browser sends. Its clobber test runs on a
**copy** of the sheet in a temp dir rather than the real
asset: the op-layer mutation reverts exactly this guard, and pointing that at the
real walk sheet would let a mutation run destroy it. Its section 12b is the audio
library: the listing (subfolders, hidden files, non-audio extensions, a quoted
path, a blank one, a missing one, a file given where a folder was), the attach by
path, that the staged copy is byte-identical to the library file, that a clip
from the library ships beside the sheet exactly like an uploaded one, and every
refusal including the 64 MB cap. It also opens the **folder** a run wrote and
checks the clips come back, and that a folder holding a sheet but no sidecar is
refused rather than guessed at. `SPRITE_URL` overrides the base URL, so a change
to a route can be tested against a second server instead of restarting the one
that is already holding open documents.

Its section 13 is **Match content size over the real route**: it writes the
reported sheet to disk (four 64×64 frames whose subjects are 30, 24, 20 and 38 px
tall — an 18 px spread), opens it through `/api/editor/open`, runs the op, and
then measures the **served cell PNGs** rather than trusting anything the server
says about itself. It checks the spread collapses to zero, that every subject is
still on the pixel it was on, that the layout and frame count are untouched, that
the status line's claimed size is the size the pixels actually have, that undo
restores sizes *and* positions, that a partial selection leaves the other frames
byte-identical, that blank frames are skipped and counted, and that a selection
with nothing measurable is refused with no undo entry left behind. Two controls
are observable from outside and both are checked: `anchor: top-left` holds a
different point (so the pivot is wired, not fixed) and `resample: nearest` lands
on the reference size exactly where the interpolating default came back one pixel
over.

One trap is worth recording, because the first version of this section fell into
it. The anchor was measured as "the bbox's middle column, last opaque row", which
is *not* what `bottom-center` means — it is `(x0 + w // 2, y1 + 1)`. Those two
agree or differ by one depending on whether the subject's width is even or odd,
and scaling changes the parity, so the check reported a one-pixel drift on two
frames and looked like a real bug in the op. It now calls the editor's own
`anchor_point`. **A check that restates the rule it is supposed to verify can
only ever test itself.**

Its section 13b is the colour picker over the route, which is the layer where a
changed payload actually shows. The op layer calls the op directly and the
browser suite drives the page's own form, so neither would notice a route that
dropped or mangled the value on the way through. It asserts the registry serves
one `color` control, then keys a green band off a real sheet and checks that the
colour the picker named is the colour that went — and that an unreadable value is
refused *at the route* with no undo entry left behind, rather than silently keyed
as black.

Its section 13c does the same for the **palette on Pixelate (mesh)**: it opens a
noisy upscaled sheet through the route, runs the op with a four-colour palette,
and reads the colours out of the **served cell PNGs** — every opaque colour must
come from the palette and nothing else, the status line must say the palette came
from the picker, and the unselected frame must stay byte-identical. One fixture
detail is load-bearing and cost a run to find: `_sheet_raw` writes 2×2 of 64×64,
and the mesh detector finds nothing to measure in a source that small — it
returns a degenerate 1×1 grid and a fully transparent result, so the check read
"no colours at all" rather than "the wrong colours". The section names
`pixel_width` instead of leaving it on auto, which is the control the op offers
for exactly that case and keeps the mesh out of the way of what is being tested.
A real input is a full-size AI image, which is what auto is for.

Its section 13d does the same for **Fill / tint**, which is the op that had no
tests of any kind until this change. It checks the registry serves one `color`
control defaulting to `#ff0000`, fills a frame through the route with
`#13ff38`, and reads the **served cell PNGs** back: the pixels are the colour the
picker named, the transparent region is still transparent (so it is a tint and
not a fill of the whole cell), and only the selected frames are reported changed.
It also drives the refusal over the route, so a bad colour cannot be swallowed
between the op layer and the caller.

`editor_drive.py` drives the real page over the Chrome DevTools Protocol —
**441 checks** — because the op layer and the HTTP smoke both post
JSON directly and neither touches the DOM. Real `Input.dispatchMouseEvent` input
(rather than a JS-constructed `PointerEvent`, which carries no active pointer and
makes `setPointerCapture` throw) drives the marquee, the shift/ctrl-click
selection, the pencil stroke, the timeline, the playback loop, a real save
through the save panel, the subset export, the edit-and-continue round trip, the
save refusal, and the Snap Pixels panel (hidden until a frame is selected; its
defaults; its live preview; that applying shrinks the cell to the previewed size;
that the viewport zooms in; and that undo restores it). It is the layer that
found the unreachable-refusal bug, the
marquee-on-a-cell bug, the `orig`-desync `IndexError`, the shift-select range
bug, and the two focus-and-restart bugs the other two suites cannot see at all —
both other suites open a fresh document against a live server, so a page holding
a document that server has forgotten is outside what they can express. It asserts
undo restores a cell's *exact PNG bytes* by hashing the server response.

Section 16e used to drive the audio library — a real folder of clips on disk,
listed by the panel, dragged onto a frame in the timeline and onto a cell of the
sheet. It went when the panel did: every drop target was gated on the library's
own drag state, so removing the panel left the targets with no drag source and
they were removed with it. **Both are back**, because removing them turned out to
be wrong — reported as *"Audio Librarry disapeard"*. The argument for the removal
was that *Add sound* is a file dialog and needs no folder; the counter-argument is
the one that actually applies here, which is that picking the hundredth clip out of
a file dialog is the thing a folder listing exists to avoid.

The rebuild fixes the reason it was fragile in the first place. The drag now carries
the clip's path in the `DataTransfer` under a private MIME type
(`application/x-sprite-clip`), and the drop targets accept *that* — so they decide
from the drag's own payload instead of consulting a module-level "a library drag is
in progress" flag. A drop target that reads the payload cannot be orphaned by the
panel that feeds it, which is exactly how the last one died.

Section 16l is the guard for the *gesture* half of **Match content size**, which
neither of the other suites can reach. It asserts the palette node exists under
Transform, that the form generated one control per argument in registry order
with the declared defaults (so the one-click case needs no setup), then writes the
pulsing sheet to disk, opens it through the page, presses **All**, and presses the
node's own **Apply** — the exact sequence the user described. The sizes are
measured from the **client's own decoded cells** via a `cellBox` helper, which is
the only measurement in any of the three suites that reads what is actually on
screen rather than re-fetching from the server. It also checks the frame preview
really redrew (frame 0's subject shrinks from 30 rows to 27, so the ink count must
drop — a surface that kept stale pixels after an edit is a bug this project has
had before), that the layout readout is unchanged, and that undo restores the
pixels exactly.

Section 16n pins **"when undo do not change the zoom and position"**. Undo and
redo both used to end in `fitView()`, and `fitView` is a pure function of the
sheet's pixel size and the viewport — so with the sheet unchanged it recomputes
the view that is already there, and the only thing it can actually alter is a
zoom or pan the user set by hand. Since undo is pressed *while looking at a
pixel*, that made hand-zoomed work impossible. The fix is `refitView(was)`: it
re-fits only when the sheet's own pixel size moved (a re-grid, a cell resize, a
rotate) and otherwise just redraws.

The check has two halves on purpose, and that is the interesting part. "Undo does
not move the view" alone is satisfied by *never* touching the view, which would
strand the sheet off screen after undoing a re-grid — so the second half requires
a re-fit when the sheet's pixel size really does move, stated as the two
properties a user would notice (the whole sheet is on screen, and it is centred)
rather than by recomputing `fitView`'s formula. It is the *distinction* that is
pinned, not the absence of movement. It also first asserts that the hand-set view
is off centre, because "the view did not change" would otherwise also pass if the
view had happened to already be the fitted one. Both directions were shown to
fail: reverting to `fitView()` breaks the zoom-preservation half, and replacing
`refitView` with a bare `draw()` breaks the re-fit half.

Section 16o is the guard for *"add same palette from snap pixel to pixelate
(mesh)"*, and it is the only suite that can see the thing that was actually
asked for. The op layer proves the parsing and the mapping, and the HTTP smoke
proves the route carries the value; neither can see whether the second panel
renders the **same** control or a lookalike that writes a different value. So it
compares both panels' pickers — same classes, same swatch, same hidden value,
same starting name — opens the mesh one, picks **PICO-8**, and checks that the
op's own form value is the 16-colour palette. Then it runs the op and reads the
**client's decoded pixels** back with a `cellColours` helper: every colour on
screen must be in the palette that was picked, and no others. Unwiring the
control makes eight of those checks fail, and the stray colours they report
(`488ED4`, `5096DC`, `589EE4`) are exactly the un-palettised mesh output.

Section 16p is the guard for **Fill / tint**, and it is deliberately not a third
hand-written copy of "the control is a picker". It asks the page's own registry
(`OPSCHEMA`) which ops declare a `color` argument and requires **every** one of
them to render exactly one `<input type="color">` — so a new op that declares a
colour is covered without anyone remembering to add it here, and an op that
quietly goes back to channel boxes fails. Then it sets the fill picker to
`#13ff38`, applies it, and requires **every opaque pixel on screen** to be that
colour, read from the client's decoded cells. Reverting `fill` to channel boxes
fails it, and so does the op-layer section 29 and the HTTP section 13d — the same
regression caught at all three layers.

One check in that section is split in two on purpose, and the first version got
it wrong. It asserted that every colour picker "shows the colour the registry
declares" — but by the time 16p runs, section 16m has already set Erase a
colour's picker to `#e6e6e6` and applied it, and a control keeps what the user
chose. The check reported the page as broken when the page was right, and the
real fault was that the assertion ignored its own section's side effects. A
declared default is a property of a **freshly built** page, so it is now read
after a reload (with the remembered sheet cleared first, or `boot()` reopens a
fixture the run has already deleted and puts a 404 in the console that section 17
would rightly fail on).

Section 16q is the guard for *"when play buton active pallet drop downs not
opening"*, and it is the one section in this file whose first draft measured the
wrong thing twice.

The panel is `position:fixed`, measured off the button's rect, and it is closed
by a capture-phase `window` scroll listener. That listener used to close on *any*
scroll outside the panel. The only thing in the page that scrolls by itself is
`markTimelineCurrent()` — a `scrollIntoView()` on the timeline strip, which runs
only while `S.playing`. So the click **did** open the panel and the strip's own
auto-scroll shut it again before it could be seen, which is exactly why the
report names the play button. The fix narrows the listener to a single
`t.contains(btn)`: the button moves when one of its own ancestors scrolls, and
nothing else can move it.

That one clause is the whole test, and the viewport is not a special case of it —
a viewport scroll reports `document` as its target and `document.contains(btn)`
is true, so it is already covered. The first version of the fix spelled out
`t === document || t === document.scrollingElement || t.contains(btn)`; the
probe then showed the **second clause could never fire** (the target is
`document`, while `document.scrollingElement` is `<html>`), and the first was
redundant with the third — which the comment above it already said. Belt-and-
braces clauses that cannot fire are worse than none, so they went. Section 16q
now measures the facts that make the single clause sufficient rather than
asserting them: that the viewport **cannot** scroll as shipped
(`body{overflow:hidden}` plus `.app{height:100vh}`, so `scrollHeight ==
clientHeight` and `scrollTop` stays 0), and that when the condition is forced, a
document scroll really does arrive with target `document` and really does close
the panel.

Both halves are asserted, because either one alone is satisfied by a broken
listener. *Always close* passes the half that requires the panel to go when its
own column scrolls; *never close* passes the half that requires it to survive the
strip. So the section opens the picker, scrolls the ops column, and requires the
button to actually move (425 → 365 px) and the panel to close — then starts
playback, opens the picker, waits for the strip to really auto-scroll
(`scrollLeft > 0`, and the page must see a scroll whose target is `strip`), and
requires the panel to still be open with the button unmoved. Mutant `Q` reverts
the listener and fails **only** the playback check; the other seventeen checks in
the section still pass, which is what shows the two halves are independent. (The
document-scroll checks are not the discriminator, and should not be: they assert
that the panel *closes*, which a listener that closes on everything also does.)

Two things had to be true before any of that meant anything, and the scratch
probe that preceded this section got both wrong. **The button has to be a
laid-out control**: the Operations panel is folded at boot, and
`updateSnapGate()` hides the whole Pixel art group unless a frame is selected, so
without both the picker has a zero rect — the probe was opening a dropdown
attached to a button no user could have clicked, and `button's scrollable
ancestor: none found`. The section now asserts the gate is closed with nothing
selected, that selecting a frame lays the picker out, and that the ops column
then really overflows (1336 > 967 px), because a scroll of a column that does not
overflow cannot move anything. **And a scroll event is not delivered when you
assign `scrollTop`**: it is queued for the rendering step, so forcing layout with
`document.body.offsetHeight` is not enough, and the first draft read "still open"
for a listener that had not run yet. Hence the `settle()` helper — two real
`requestAnimationFrame`s — next to `open_by_name`.

There is also a **second close path**, and confusing the two is how the section's
first draft reported a failure that was not one:

```js
document.addEventListener("click", () => { panel.hidden = true; });
```

That is the ordinary click-outside-to-close. The palette button calls
`e.stopPropagation()` so its own click does not reach it — but a click on Play
does. So the order in 16q is load-bearing: playback is started **first** and the
picker opened **second**. Opening the picker and then pressing Play measures the
click-outside path, not the scroll path, and reads as the reported bug while
being nothing of the kind.

Section 16r is the guard for **Resize (pixels)** — the request *"add resize
function"*, answered with the pixel-art one: nearest-neighbour, whole factors.
The three size controls that already existed (`set_cell`, `resample`,
`match_size`) all *filter*, so scaling pixel art up softened it and the user's
own wording for the gap was "nearest-neighbour for antialiazed?".

`test_editor.py` section 30 pins the arithmetic on numpy and the HTTP smoke pins
the route, but "nearest-neighbour" is a claim about pixels **on screen**, and the
two things only a browser has are the palette node the user presses and the
client's own decoded cells. So the section loads a 4-frame 64×64 fixture built to
punish any filter — a one-pixel checkerboard of blue and yellow, the worst case
there is — applies the op through the real Apply button, and then reads the
page's decoded pixels three ways:

- **shape** — every 2×2 block of the result is a single colour, which is what
  replication means and what no interpolating resize can satisfy;
- **colour set** — nothing appears that was not already on screen, so a filter
  that invented the blends between blue and yellow fails here;
- **byte-exactness** — `up x2` then `down x2` returns the original pixels by
  hash, so the two are exact inverses rather than near-misses.

It also asserts the two halves of the wiring that a `full` op needs and that
neither other suite can see. First, the client really re-decodes: the payload
saying `cell_w: 128` while the cell cache still holds a 64-pixel image is exactly
the failure this is here for, so the decoded size is measured, not the layout
number. Second, the view **refits** — and that check found a real gap. The op was
missing from the list of ops that call `fitView()` after they run, so a ×2 left
the sheet at the old zoom with half of it off screen. The check had to be made
observable before it meant anything: `fitView` is a pure function of the sheet
size and the viewport, so a sheet that already fits is refitted to exactly where
it already was. The section therefore forces a zoom the sheet does not fit in
(asserting it really does overflow), applies the op, and requires the scale to
have dropped and the whole sheet to be on screen again. Mutant `T` removes just
the name from that list and fails that one check, which is what makes it a guard
rather than a description.

Section 16m is the guard for the picker itself, which is a *page* claim and so
belongs here: the op layer calls the op directly and the HTTP smoke posts JSON, so
neither can see whether the control the user actually gets is a colour picker or
three spin boxes. It asserts the node is in Matte repair, that the form renders
four controls where it used to render six, that the first is
`<input type="color">` defaulting to `#000000`, and then sets the swatch, presses
Apply, and measures the result on the client's own decoded cells: keying the
subject's own colour has to leave only the red marker pixel behind. That last part
is what makes it a colour key rather than "erase everything". It has to untick
*only border-connected regions* for the fixture — the block sits inside the frame
without touching its edge, so the topological guard protects it, which is the op
working rather than a nuisance.

Section 16j is the guard, and it is the only section in any of the three suites
that drives a **real HTML5 drag**. That needs `Input.setInterceptDrags`, a real
press on the row, and a real mouse that travels all the way to the target: the
browser then starts the drag itself and `Input.dragIntercepted` hands back the
payload the page's own `dragstart` built, which is fed to `Input.dispatchDragEvent`.
Two things cost real time and both look identical to a dead drop handler, so they
are worth stating: **`dragEnter` is required** (`dragOver` + `drop` alone produces
no drop at all — measured, 0 drops), and **the mouse has to reach the target**
(Chromium tracks the drag's position from those moves, so a mouse that stops short
delivers the drop somewhere else). It then checks the clip landed on the frame it
was dropped on rather than on the current one, that a drop onto a sheet cell maps
through the same `screenToSheet`/`cellAt` pair the pointer tools use, that a
foreign drag carrying a clip that *would* attach is still turned away, and that a
folder which is not there lists nothing and says why.

**The third thing, and the one that actually broke this section: a drag that
hovers near the edge of a scrollable container makes Chromium autoscroll it.**
The strip is `overflow-x:auto` and six frames do not fit in it — `scrollWidth` 311
against a `clientWidth` of 240 — so frame 4's *rect* runs past the scrollport and
its centre lands about 9px inside the right edge. A drag dwelling there made the
strip run to its maximum, `scrollLeft` 71 of 71, which slid the aimed-at frame 71px
away, ended the drag with `dragend` and **no `drop` at all**, and left the document
untouched. From the outside that is indistinguishable from a dead handler, and it
was diagnosed as one: this file used to carry a "known environment caveat" claiming
that a `--headless=new` Chrome refuses to deliver the drop. That was wrong. The
evidence that killed it was an injected drag source carrying the same private MIME
and dragged to the same point, which landed and attached — the gesture works fine
in isolation. 16j now scrolls the target frame into view before aiming at it, which
is also what a user does before dropping onto a frame they cannot see, and it
records what the drag actually did so the next failure of this shape explains
itself instead of being re-derived by hand:

```
dragenter:CANVAS[4]@[1539,465] sl=0   -> dragover:CANVAS[4]@[1539,465] sl=0
  -> dragenter:CANVAS[5]@[1539,465] sl=71 -> dragleave ... -> dragend
```

Two checks guard the mechanism rather than the symptom, because "the drop landed"
can be satisfied by a drop that landed somewhere else: the aim point has to
hit-test to the frame it means to, and the strip's `scrollLeft` has to be unchanged
across the drag.

Section 16j also guards **which listing the panel is showing**, because two
listings can be in flight at once. `boot()` starts `restoreLibDir()` without
awaiting it, so the remembered folder's listing can still be arriving when the user
types a folder and clicks Go; if that first response lands second it replaces the
listing they asked for with the one they did not, and `LIB.items` then hands out
one folder's paths while the panel shows another's. That is not hypothetical — 16j
read a row path belonging to the *previous run's* folder while the panel showed
this run's, and it survived three runs because both fixture folders hold the same
clip names, which is the one shape of fixture that cannot tell two listings apart.
`loadLib` now carries a request token and only the newest request may render, or
clear the panel, or drop the remembered key. The driver proves it by **holding the
first response open** for longer than the whole sequence takes, so the wrong order
happens by construction rather than by luck, and it does that twice: once with a
stale listing that succeeds, once with one that fails.

A crashed run used to poison the next one. `editor_drive.py` makes its fixture
folders with a tracked `mkdtemp` and removes them on exit, however it exits — an
`atexit` hook rather than a `finally`, because the sections are 3400 lines of one
function and an exception in the middle of one has to clean up too. Without it, a
run that died inside 16j left its folder on disk, and because the page remembers
the folder a sheet was opened from, the *next* run's very first check ("a fresh
profile shows an empty folder picker") failed against a folder this suite had made.
For a profile that was already dirtied that way, `_probe_reset_profile.py` clears
`sprite.recentFolders`, `sprite.lastSheet` and `sprite.audioLibDir` and leaves the
theme alone.

One more trap worth knowing about, because it is the same shape as the superseded
listing: `audio_payload()` sorts by `(frame, id)`, so `audio[length-1]` is the
**highest** frame and not the newest clip. The sheet-drop check used to read the
frame off the end of the list, which answered the question correctly only while the
timeline drop above it was failing to attach anything at frame 4. Fixing the drop
made that check fail with `(4, 0)` — the sheet's clip was there, at frame 0, with
the frame-4 clip sitting behind it. It looks up the frame now.

And one check in this section was measuring a behaviour the page has never had.
`restoreLibDir` removes the remembered key when a folder will not open and
deliberately **leaves the path in the box**, "so the failure is visible while the
user is looking at it" — the same pair the sheet's own failed reopen does in
`boot()`, which removes `sprite.lastSheet` and leaves the path in `sheetPath`. The
check asserted the box was emptied, so it could never pass, and its failure was
lumped into the same "environment caveat" as the drags. It asserts the visible half
now: the key is gone, the path stays.

What remains of the audio behaviour elsewhere is 16d (attach through *Add sound*,
listed, marked on its frame, removed) and 16f below (clips surviving a save and a
reopen).

That "marked on its frame" was only half true for a whole session, and the mutant
sweep is what said so. 16d checked the **timeline** mark — the thumbnail's title
and its badge — and the README's mutant table claimed a second check on the
**sheet canvas** that did not exist: it went with the old audio-library section and
was never restored when the panel came back. Mutant `K` (delete the sheet mark)
therefore came back MISSED, 439 checks 0 failures. The check is back, and it is
written so it can fail: it forces a zoom where the corner marks are legible (and
asserts `legible`, since the page draws them only above a size threshold), reads
the page's own canvas for the mark's exact cyan `#0e7490`, and compares the frame
that has the clip against one that does not — an absolute count of cyan could be
satisfied by anything else on the canvas. `K` fails it now. One trap worth
recording: `__d` is **not** injected this early in the run (16b reloads the page
and the helpers come back at 16g), so the section sets the view through `S.view`
directly and restores it after, the way 16c and 16d already do.

Section 16g is the folder picker's own guard, because the three rules that make it
a shortlist rather than a history are the whole feature and each can fail alone. It
builds eleven folders, each holding a sheet and a sidecar, opens them through the
real path box, and checks that the newest is on top, that reopening an older one
moves it back up instead of listing it twice, that the list stops at ten and drops
the oldest, that clicking a row opens that folder (the *last* row, so the check
cannot pass by reopening the document that happens to be open), and that the list
survives a reload. The reload check compares the *set* rather than the order, since
boot() reopens the last sheet and that legitimately moves its folder to the top.

Section 16h is the guard for *"when i click apply metadata it should write in
current json"*. It plants a sidecar carrying keys the panel does not own
(`source`, `matte`, `edited`), opens the folder, types an fps, clicks the real
button, and then reads the file **from disk in Python** — the page's own report is
not evidence that a file changed. It checks that the fps and name landed, that the
note was rewritten from them, that the unowned keys survived, that the sheet PNG
is byte-identical, and that the status line names the path. Then it re-grids and
clicks again: the refusal has to reach the panel (`metadata NOT written: …`) and
the file must be left alone rather than half-written. The whole thing runs on a
throwaway copy — this section's point is that a button writes a file, and the
sheet it is opened from in the earlier sections is a real asset in the user's
`walk/` folder.

Section 16h is also where the *handing back* half of the player's fps lives: 8c
takes the box over, 16h opens a sheet and checks it goes back to following the
document. That pairing is deliberate — the hand-back needs an open, and an open in
8c would throw away the edits section 10 asserts on.

Section 16i is the guard for *"allow overwriting the sheet this document was
loaded from when checked not working"*, and it is the clearest case in this
project of a harness being blind in exactly one place. It walks every `.chk` row
in the page, **clicks its words with a real mouse**, and asserts the box flips;
then it asserts each row is a `<label>`. Both fail against a `<div>` row. The
reason nothing had caught it is worth stating plainly: every earlier interaction
with those controls — in the driver and in the sections above — set `.checked` in
JavaScript, and a JS assignment proves the handler reads the box while saying
nothing about whether a mouse can reach it. The section scrolls each row into
view before clicking, because the columns scroll internally while the document
does not, and an un-scrolled click below the fold lands on nothing — which would
read as "labels are broken" for the wrong reason.

One row still would not click, and the reason is worth keeping because it looks
exactly like the bug this section exists to find. Snap Pixels' own checkbox sits
in the **Pixel art** group, and `updateSnapGate` sets that group's
`display: none` until a document is open *and* a frame is selected. A
`display:none` ancestor ignores `open`, so no amount of unfolding could ever lay
that row out: its rect stays all zeros and the click lands on the panel header.
That is the page working as designed — the panel's own hint says to select
something first — not a dead label. So the section names the gate rather than
skipping the row: it asserts the *only* unreachable rows are the gated ones, that
selecting a frame releases them, and only then clicks. "It is behind a gate" must
not become a general excuse for any row that will not click, which is why the
gate's membership is asserted too. The failure detail (`words_diag`) reports
whether an ancestor is `display:none` for the same reason: without that field,
"the label does not work" is the only conclusion the output supports, and it is
the wrong one.

Section 14 is where the save *destination* is pinned, and it is the other half of
the same report — *"overwriting not working fix taht"*. The gesture that exposes
it is the ordinary one: open a run's sheet, tick the box, and leave the output
folder **blank**, which is its default ("blank = runs/editor_…"). A blank folder
made `app.py` invent `runs/editor_<stamp>_<name>/`, so the ticked save wrote a
brand-new pair there and the files it was loaded from were never touched. Measured
on the running server before the fix: `dir=runs/editor_0921-181431_smoke12`, and
both loaded files byte-identical afterwards. The section now clears the folder
box, renames the document so the derived output name cannot coincide with the
loaded file, saves, and then checks the **bytes on disk** rather than the panel's
report — plus that nothing was written under the new name.

The rule it pins is two-sided, because the box is worded as permission
("allow…") and the refusal it disables is about the destination. So a folder the
user *did* name still wins, and section 14 asserts that too: ticked, with a
folder given, the copy lands in that folder and the original is left alone.
Redirecting there would silently ignore an explicit instruction and replace the
original when a copy was asked for. `test_editor.py`'s section 27 pins both
halves at the op layer, and the two `overwrite-*` hooks exist as a pair for the
same reason — each alone is satisfied by a wrong implementation.

This one took three reports, and the reason is a trap worth writing down: **the
page is re-read from disk on every request, but `app.py` and `editor.py` are
loaded once at process start.** A server left running from before the fix keeps
serving the old save path while the browser shows the new page, so the fix looks
absent no matter how many times the page is reloaded. The footer's `BUILD` chip
catches a stale *page*; `GET /api/editor/version` catches a stale *process* — it
names the modules written since this process loaded them, and the footer badge
turns on and says so. The symptom is otherwise indistinguishable from "the fix
does not work", and the other way to tell them apart is to probe a route that
only the new `app.py` has — `POST /api/editor/<id>/save_meta` answers
`{"ok":false,"error":"unknown document"}` when the route exists and
`{"detail":"Not Found"}` when it does not. Three times this cost real time in
this project, the stale server was one a session had started itself, on the
default port, and then forgotten — and it happened a fourth time with the op
below, which is why the rule is now: **after changing `editor.py`, restart the
server before believing anything the page shows.**

The badge prints `started` as the server's own start time, and it was reporting
`min(_BOOT_STAMPS)` — the mtime of the *oldest watched file*. That reads as a
start time only when the two files were written together; on this machine they
were not, so a server booted minutes earlier announced a start two days in the
past, on the one badge a user consults to decide whether their server is current.
It is now `time.time()` captured at import, and `smoke_editor_api.py` pins the
invariant that catches it: a server with nothing stale must have started no
earlier than the newest file it loaded. Reverting the fix makes that check fail,
which is how it was shown to be load-bearing rather than merely plausible.

Section 16f is the guard for a reported bug — *"when i load folder audios are not
loaded"*. It builds a run's output folder on disk in the shape `save_doc` writes:
a sheet, a sidecar naming it, and the clip in `audio/` beside them. Then it types
the **folder** into the open box and checks the document, the clip, the Audio
panel's row, the timeline's badge, and that the box rewrites itself to the
sidecar the folder resolved to. The sidecar in that fixture is deliberately not
named after the folder, so it exercises the fallback in `_sidecar_in_folder`
rather than the fast path — which is the shape a real run has.

It also measures the pencil's mark in **screen pixels**, because "the bytes
changed" and "the user can see it" are different claims: at the fitted zoom the
old brush left an 11-px-thick mark that was 0.59 px on screen, which every
byte-level assertion happily passed. Reverting `brushRadius` to sheet pixels makes
that check fail, so the regression is pinned. And it then reads the **page's own
pixels** — the sheet canvas, the preview panel and the timeline thumbnail —
because "the server has it" and "the page shows it" came apart too: the cell URL
was constant, so a reload could be answered from the browser's decoded copy of
the cell and the edit never reached the screen.

`mutate_editor_page.py` mutation-tests this suite by editing the served page and
re-running it — the only way to test a guard that lives in the browser and is
only reachable through real input. **Twenty mutants**:

| mutant | what it reverts to | caught by |
|---|---|---|
| `A` | the reopen reports success unconditionally | `and the status admits the reopen failed instead of claiming it` |
| `B` | the `unknown document` trigger is removed, so the page never recovers | the section-16 wait times out |
| `C` | the canvas shift-click is a grid rectangle again | `96 frames, not the 12-cell rectangle between them` (+4) |
| `D` | a new sheet does not reset the shift-select anchor | `the shift-select anchor was reset with the new document` |
| `E` | the shift-select range is not clamped to the document | `an anchor past the end of the document cannot select a cell that is not there` |
| `F` | the subset box is decorative: nothing is sent | `and the panel reports how many frames it actually wrote` (+1) |
| `G` | no edit action on the save result | `and offers an edit action on the result` |
| `H` | the export label ignores whether the box is ticked | `with the box off, the label talks about the selection` |
| `I` | the save hint stops following the box | `and the hint switches to the subset` |
| `J` | the Operations palette is open by default again | `the left column is the picker, the palette and the audio panel, folded except Audio` |
| `K` | a frame with a sound is no longer marked on the sheet | `and the frame with a sound is marked on the sheet while the one without is not` — the check this row named did not exist until the sweep found it (see below) |
| `L` | the playback loop reads the document's fps instead of the player's | both 8c rate pairs invert — `at 240 player fps the preview plays the whole sheet through` (+3) |
| `M` | Apply metadata stops calling the save route | `the fps typed into the panel is in the JSON on disk` (+3) |
| `N` | the overwrite checkbox's words stop being part of the control | `every checkbox row is a <label>` (+2) |
| `O` | a dropped clip attaches to the current frame instead of the one under the pointer | `and it landed on the frame it was dropped on, not on frame 0` |
| `P` | any drag is accepted as a library clip | `a drag carrying only text/plain attaches nothing, even when it names a clip that would attach` |
| `Q` | the palette dropdown closes on any outside scroll again | `the picker stays OPEN through the strip's auto-scroll (the reported bug)` |
| `R` | a superseded audio-library listing renders anyway | `a listing that is no longer the one asked for is dropped` (+2) |
| `S` | a superseded request that *failed* clears the panel anyway | `and a superseded request that fails leaves the panel and the note alone` |
| `T` | a cell-size op is left out of the refit list | `the view refitted, rather than leaving the doubled sheet at the zoom it was at` |

Every mutant is derived from the live `editor.html` text, and the script exits 2
rather than silently mutating nothing if the source has drifted. It restores the
page in a `finally`, so a crashed run cannot leave a mutant on disk.

The classifier used to conflate two different failures. `wait_for` **raises**
`TimeoutError` instead of recording a FAIL, so a mutant that makes the page never
reach a state a section waits for kills the driver before its summary — and
because "CAUGHT" was gated on the summary line, that was reported as **BROKEN**,
which in this file means "the hook could not apply, so it tested nothing". The
truth was the opposite: the mutation had been detected, and the driver said so by
dying. "Actually ran" now also accepts a driver that started and then died
(`"Traceback" in out and "  PASS  " in out` — the PASS lines are the evidence it
got as far as running checks), so BROKEN stays reserved for a run that produced
no result at all. A full sweep before that change read 13 CAUGHT / 5 BROKEN /
1 MISSED. Its pre-flight had passed, so the anchors had not drifted and the five
were not "the hook could not apply" — and re-running exactly those five under the
fixed classifier returned **all five CAUGHT** (B, D, F, G, M), with labelled
failures. So that whole column was a reporting artefact of the classifier.

**`K` was the one real finding, and it is worth the sweep existing.** It came back
MISSED: 439 checks, 0 failures, with the mark deleted from the page. The reason is
that the check the row named — "frame 0, which has a sound, is marked on the
sheet" — **did not exist anywhere in the suite**. The `♪` assertions were all
about the *timeline* (its thumbnail title and its badge); the sheet canvas, which
is the surface the user is actually looking at while working on a frame, had no
check at all. It went with the old audio-library section and was never restored
when the panel came back, so for a whole session the table claimed a guard that
was not there — the same failure mode as `export-mutates-doc`, found the same way.

The check is back in 16d, next to the timeline one, and it is written to be
falsifiable: it forces a zoom where the corner marks are legible (asserting
`legible`, since the page draws them only above a size threshold), reads the
page's own canvas for the mark's exact cyan, and compares the frame that has the
clip against one that does not — an absolute count could be satisfied by anything
else cyan on the canvas. `K` is caught by it: 441 checks, 1 failure, with the
detail `{'legible': True, 'marked': 0, 'plain': 0}`.

`M` is the one the op layer structurally cannot cover: `test_editor.py` proves
`save_meta` does the right thing to a file, and only a real click can prove the
button reaches it. `L` is the same shape for the playback loop, and `N` for the
checkbox labels — a JS `.checked = true` passes just as happily against a row
whose words are dead.

The sweep takes `--url`/`--port` through to `editor_drive.py`. It used to drive
the default port, which is a server someone else may be using — and one started
before the latest `app.py` changes will fail a section on *every* mutant, so the
sweep reports "CAUGHT" for a reason that has nothing to do with the mutation.

**That pass-through had a second edge, and it produced a false CAUGHT of exactly
the kind the paragraph above warns about.** `sys.argv[1:]` was forwarded whole, so
the mutant *tags* went to the driver along with the flags. `editor_drive.py` takes
only `--url`/`--port`, so `python mutate_editor_page.py Q` handed it a bare `Q`,
argparse exited 2, and the mutant was reported CAUGHT in about a second instead of
a couple of minutes — before a single check had run. The tags are now filtered out
of the pass-through (`PASSTHRU`), and a run that reports **no result at all** is
classified `BROKEN` rather than CAUGHT: an argparse error, a missing server and a
crash all exit non-zero, so reading the exit code alone cannot tell a strong
mutant from a suite that never started. The same trap is why `sweep_editor_mutants.py`
classifies its own exit code 2 separately.

Two things about that restore are worth stating, because both were wrong. The
page is **replaced atomically** (`os.replace`) rather than truncate-then-written:
the server reads `editor.html` on every request, so a plain write leaves a window
in which a page fetch gets half a mutant — a page that behaves like nothing else
and cannot be reproduced from either version. And it is written with
`newline=""`, because the default text mode rewrites line endings on Windows, so
the "restored (N bytes)" line used to print a number that was not the number
taken before the run. It now says `byte-exact`, and that is checked rather than
asserted.

**And fixing those two introduced a third, which is the more instructive one.**
Reading with `newline=""` stops the universal-newline translation, so the page
text now keeps its `\r\n` — while the mutant anchors are written with plain `\n`.
Four of the eleven anchors (B, H, I, J) span a line break, so they could no longer
match; and because the pre-flight exits on the *first* miss, anchor B took the
whole sweep down. It failed loudly, but it read as "B has drifted" rather than
"four anchors can never match and nothing has been tested since". The anchors are
now converted to the page's own line endings before matching, the pre-flight
names *every* drifted anchor instead of only the first, and a replacement that
changes nothing is refused outright — otherwise the pristine page runs and
reports `MISSED`, which reads like a weak mutant instead of a broken harness. The
lesson is not about line endings: **it is that a harness fix has to be followed by
a full sweep, because three defects here were found one per run, each introduced
by fixing the last.**

`ui_drive.py` grew three checks from this work. The newest is the colour key: the
controls have to be in the form, the key has to start off, the tolerance has to
match the server's default, and **reading a clip has to seed the picker with the
clip's own border colour** — compared against what `/api/probe` itself reports, so
the check is about the route and the page agreeing rather than about a hard-coded
hex. One of the others is the input panel itself: the
clip has to load its metadata, play, and land a mid-clip seek (which is what
proves the route answers ranges), and the panel has to be holding a *video*
rather than the old still. The other is a comparison between the form the page
shows and `/api/defaults` — `ui.html`'s markup is not the last word, because the
page overwrites every field from the server on load, and a process started before
a default changed silently re-applies the old one. That is how a checkbox that
was already unchecked on disk kept coming back ticked, and how a build whose
constant said 32768 still refused a save at 16384; both were reported as editor
bugs. It also stops printing the page's text on a cp1252 console, which used to
kill the run with a `UnicodeEncodeError` about an arrow *before* it printed its
checks table.

`preview_drive.py` drives the *generated player page*, the one surface nothing
covered: the byte-level checks look at the sheet, the op-layer tests at the
document, and `ui_drive.py` only asserted that the generator embedded an
`<iframe>`. It needs no model and no video — it writes its own 8-frame sheet of
solid, distinct colours, so every assertion is about pixels the test chose — and
it checks the layout (the cell fills the stage on one axis without overflowing
it, at three window sizes) and the playback, 31 checks in all. The playback check
records a trace instead of sampling: `draw` is wrapped from inside the page, so
every animation tick lands in `window.__s` with its frame index, the alpha of the
cell's centre pixel and the button's label. Sampling from the outside can miss
the tick the old bug went blank on; a trace cannot — reverting the loop makes it
fail with 14 problems, including `frames 0..35` on an 8-frame sheet.

It also sets its own preconditions rather than inheriting them: the tool is
explicitly reset to Select at the start of each interactive section, because the
pencil selected in one section persists and a "click" would then paint a stroke
instead of selecting a cell.

```bash
# start the server, then a headless Chrome with a debug port
chrome --headless=new --remote-debugging-port=9222 about:blank
python editor_drive.py --url http://127.0.0.1:8765 --port 9222
python preview_drive.py --url http://127.0.0.1:8765 --port 9222
```

The CDP client drops its `Origin` header (`suppress_origin`), because Chrome
refuses the WebSocket handshake with a 403 unless that origin was passed to
`--remote-allow-origins` — so the launch line above, on its own, used to fail
before the page ever loaded.

`smoke_editor_api.py` defaults to `http://127.0.0.1:8765` and takes
`SPRITE_URL` to point somewhere else (`SPRITE_URL=http://127.0.0.1:8766 ...`),
which is how a change to a route gets tested without restarting the server that
is already running.

The server has to be *restarted* after a change to `pipeline.py` (or any other
Python file), not just reloaded in the browser: `ui.html` and `editor.html` are
served from disk on every request, so front-end edits appear immediately, while
`DEFAULT_CFG` and the texture cap live in the running process. Two of the reports
here — a default box that came back checked and a save refused for exceeding a
16384 px cap — were exactly that, with the code already fixed on disk: the page
takes its defaults from `/api/defaults`, so the stale process re-applied them.

## Files

| file | role |
|---|---|
| `pipeline.py` | every generator stage, importable and testable without the web layer |
| `editor.py` | the editor's document model, op registry and 44 operations |
| `pixel_snapper.py` | server-side bridge to the spritefusion-pixel-snapper WASM |
| `snapper_runner.mjs` | the Node entry point that loads the WASM and processes a batch |
| `proper_pixel.py` | server-side bridge to the proper-pixel-art mesh pixelator |
| `pixeloe_bridge.py` | server-side bridge to the PixelOE outline-expansion pixelizer (torch backend; alpha carried outside, pixel size required to divide the cell) |
| `app.py` | FastAPI server, job manager, model cache, editor routes |
| `ui.html` | the generator front-end (offline, no CDN) |
| `editor.html` | the editor front-end (offline, no CDN) |
| `drive.py` | run a generator job from the CLI and print the result |
| `test_verify.py` | mutation test for the sheet verifier |
| `test_config_types.py` | regression test for config type coercion |
| `test_matte_repair.py` | both repair passes, including what the rim band and the connectivity test protect |
| `test_editor.py` | op-layer checks + mutation hooks |
| `sweep_editor_mutants.py` | runs every mutation hook and classifies it MISSED / CAUGHT / BROKEN, so a hook that stopped applying cannot read as a pass |
| `smoke_editor_api.py` | HTTP integration smoke for the editor |
| `editor_drive.py` | drives the real page over CDP (real mouse input) |
| `mutate_editor_page.py` | mutation-tests the driver by breaking the served page |
| `preview_drive.py` | drives the generated preview player over CDP (no model needed) |
| `_probe_theme.py` | contrast/theme probe: both pages, both themes, every text element measured |
| `_probe_palette.py` | the same WCAG ratios computed offline from the two variable blocks |
| `_add_logo.py` | one-shot: inlines the TostUI mark into both headers and renames the pages |
| `_probe_folder_audio.py` | drives the real page opening a saved folder and prints what arrived — the reproduction tool for "when i load folder audios are not loaded" |
| `_probe_open_routes.py` | walks all five ways to open a document (folder, sidecar, PNG, library row, boot) and reports clips + panel rows per route |
| `_probe_overwrite.py` | the overwrite opt-in over HTTP: unticked must refuse, ticked must replace the exact file — the server half of "when checked not working" |
| `_probe_overwrite_page.py` | the same question through the real page, with a real mouse: what `elementFromPoint` says is at the box, whether a click on the box toggles it, whether a click on its **words** does, and whether the source PNG actually changed |
| `_probe_reset_profile.py` | clears `sprite.recentFolders`, `sprite.lastSheet` and `sprite.audioLibDir` so the next driver run starts from the fresh profile its first checks assume. The driver removes its own fixture folders on exit now, so this is for a profile dirtied before that fix — or by a probe |
| `_probe_overwrite_live.py` | the gesture that was actually reported, over HTTP, on a throwaway copy: open a run's sheet, tick the box, leave the output folder **blank**, save — then report which files changed and whether the sheet went to the folder it was loaded from. Runs against any server, so it can be pointed at a stale one on purpose |
| `_probe_chk.py` | dumps the ancestor chain of the first `.chk` rows with each ancestor's `open`, `display` and rect — the tool that showed a zero-rect row was `display:none` rather than folded |

## Notes

- **A known quirk, deliberately left alone: the name box is re-rendered by every
  op.** `applyDoc` writes `#metaName` (and the rest of the meta panel) from the
  document on each op, so a name typed into it is discarded the moment you run
  anything. The save then writes `<document name>_sheet.png`, which is how the
  overwrite test in `editor_drive.py` came to write `copy_sheet_sheet.png` and
  look like a broken overwrite. The driver re-types the name before saving, which
  is what a user does anyway. Whether the field *should* follow the document is a
  behaviour question rather than a bug — `set_meta` is a real op and the panel is
  its form — so it is noted here instead of changed.

- **The model is loaded once and kept warm.** Loading costs 2–3.5 s and cudnn
  autotuning makes the first ~25 frames 5–10× slower (~580 ms/frame vs 261 ms), so
  the first job of a server session is much slower than the rest. Jobs run one at
  a time on purpose — the GPU is a single shared resource.
- **"Real-time" is not on the table.** VRMBG-3.0 measures ~492 ms/frame at 1024
  and ~243 ms at 640 on an RTX 3090. Treat this as a batch tool.
- **Licence.** VRMBG-3.0 is not open source: non-commercial use only, commercial
  use requires an agreement with BRIA AI. The model card references a `LICENSE`
  file that is not present in the published repo. Check before shipping anything
  built with it.
