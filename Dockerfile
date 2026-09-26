# syntax=docker/dockerfile:1.10
#
# 1.10, not 1.7: `env=` on a secret mount (used below to turn HF_TOKEN
# into an env var for one RUN) is rejected by older frontends with
# "unexpected key 'env' in 'env=HF_TOKEN'". Verified against 1.7 before bumping.

# ===========================================================================
# Sprite Studio -- self-contained image (clone all repos, download all models)
#
# Written in the style of a RunPod worker Dockerfile: ubuntu base, single stage,
# linear, everything fetched during the build. No named build contexts, no local
# folders -- it builds on any machine with network access.
#
# Tokens come from the ENVIRONMENT. Two sources, and the system one wins:
#
#   1. your shell / OS environment -- nothing to do, this is the default path
#   2. the gitignored `.env`, loaded only if (1) did not provide the var
#
#   set -a; . ./.env; set +a     # cheap; a var already set is NOT overwritten
#
# `.env` guards every assignment as NAME=${NAME:-...}, so sourcing it is safe
# even when the system already has the value. A plain NAME=... in that file
# would do the opposite and silently discard your system value -- measured
# 2026-09-25, see the comment at the top of `.env`.
#
#   docker build \
#     --secret id=hf_token,env=HF_TOKEN \
#     --build-arg CACHEBUST=$(date +%s) \
#     -t TostAI-Sprite-Sheet-Studio .
#
# Docker itself reads neither `.env` nor your OS environment: the `env=NAME` on
# each `--secret` is what lifts the value out of the process environment the
# build is running in. So the `set -a` line is required whenever the values are
# only in `.env`.
#
# No `-f`: this file IS the Dockerfile. It was `Dockerfile.standalone` while the
# multi-stage `Dockerfile` still existed; that one is gone.
#
# CACHEBUST IS NOT OPTIONAL IN PRACTICE. The app is cloned from its own repo, and
# BuildKit caches that clone under a key that ignores what the branch points at
# now -- so without the flag a rebuild silently re-serves the first snapshot,
# forever. Measured 2026-09-25: the editLink path fix was pushed, the rebuild
# reported success, and the image still contained the old code. The flag
# invalidates only the clone and the four cheap layers after it; the ~6-minute
# apt, pip and model layers stay cached, so a rebuild costs about 15 seconds.
#
# Rule of thumb: if the image id does not change after a rebuild, NOTHING was
# rebuilt. Confirm with `docker image ls --no-trunc TostAI-Sprite-Sheet-Studio`.
#
# A RUNNING CONTAINER CAN ALSO UPDATE ITSELF, with no rebuild: the studio's
# Update button (POST /api/update) fetches the latest source and re-execs the
# server. That path is a dev convenience -- the files it writes live in the
# container and die with it. This build is the durable one.
#
# The one secret comes from the environment. `--secret id=...,env=NAME`
# takes the value out of the caller's environment, and
# `--mount=type=secret,...,env=NAME`
# in the Dockerfile exposes it to that ONE RUN as $NAME. Nothing is written to
# disk, nothing survives into the next layer, and nothing is recorded in the
# image.
#
# Docker never forwards env vars into a build on its own -- the --secret
# flag above is what wires HF_TOKEN from your shell into
# the build. Without it the build stops at "secret hf_token: not found".
#
# Only the model repo is private (gated): the app's own repository
# `camenduru/TostAI-Sprite-Sheet-Studio` is public, so it clones with no token.
# The app is CLONED from that repo rather than copied out of the build context,
# so the image is reproducible from the repository alone.
#
# ---------------------------------------------------------------------------
# THREE DEVIATIONS FROM THE PATTERN, each forced. Do not "fix" them.
#
# 1. THE MODEL NEEDS AN AUTH HEADER. The usual
#      aria2c -c -x 16 -s 16 -k 1M <resolve-url>
#    works for public repos because the resolve URL 302s to a CDN that serves
#    ranges. briaai/VRMBG-3.0 is GATED, so the same request returns 401 before
#    it ever redirects. The aria2c lines below carry
#    --header="Authorization: Bearer ..." and the token arrives as a secret
#    mount, NOT as an ARG or ENV.
#
#    That distinction is load-bearing, and measured rather than assumed. On
#    docker 29.8.0 / buildx 0.37.1, a token passed as `--build-arg` shows up
#    THREE times in `docker history --no-trunc` (once as a bare
#    `ARG HF_TOKEN=...` entry), i.e. anyone who can run `docker history` on the
#    image can read it. The same token passed as `--secret ... ,env=...` shows
#    up ZERO times, is absent from `.Config.Env`, and is gone from the next
#    layer. BuildKit also warns outright: "SecretsUsedInArgOrEnv: Do not use
#    ARG or ENV instructions for sensitive data". So: no ARG, no ENV.
#
# 2. THE SNAPPER CANNOT BE CLONED AT ITS PINNED COMMIT. `b7365155` exists on no
#    reachable remote, so this clones the canonical repo's default branch and
#    the resulting WASM is NOT byte-identical to the one the studio runs today.
#    There is no way around this from a clone; only the local folder has it.
#
# 3. NO CUDA TOOLKIT INSTALL. The usual `cuda_*.run --silent --toolkit` is a
#    ~4 GB download that inference does not need: the pip torch wheels bring
#    their own CUDA runtime (nvidia-cuda-runtime-cu13, nvidia-cudnn-cu13, ...)
#    and only the HOST driver is required. Add the toolkit yourself if you ever
#    need to compile custom kernels.
#
# The paths are the ones the app expects: the app itself at /app/sprite_studio
# with /app/walk and /app/sprites as siblings, and /opt/models/... plus
# /opt/vendor/... for the four external folders. That shape is load-bearing --
# app.py computes WORKSPACE as the *parent* of its own directory, and the
# editor's asset browser lists WORKSPACE/walk and WORKSPACE/sprites.
# ===========================================================================
FROM ubuntu:22.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=True
ENV PYTHONDONTWRITEBYTECODE=True
ENV PATH="/home/camenduru/.local/bin:/usr/local/bin:${PATH}"

# ---------------------------------------------------------------------------
# System packages, node, and a non-root user
#
# node comes from NodeSource rather than apt: ubuntu 22.04 ships node 12, and
# pixel_snapper.py needs a modern one to run the wasm_bindgen module.
#
# git-lfs is included because the checkouts declare LFS filters; without it a
# clone can fail on the smudge step.
# ---------------------------------------------------------------------------
RUN apt update -y && apt install -y \
        software-properties-common build-essential ca-certificates \
        libegl1 libgl1 libglib2.0-0 \
        python-is-python3 python3-pip python3-dev \
        sudo nano aria2 curl wget git git-lfs unzip xz-utils ffmpeg xvfb && \
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt install -y nodejs && \
    adduser --disabled-password --gecos '' camenduru && \
    adduser camenduru sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    mkdir -p /app /opt/models /opt/vendor && \
    chown -R camenduru:camenduru /app /opt/models /opt/vendor /home && \
    chmod -R 777 /app /opt/models /opt/vendor /home && \
    rm -rf /var/lib/apt/lists/*

USER camenduru

# ---------------------------------------------------------------------------
# Python packages
#
# --extra-index-url, not --index-url: the PyTorch index hosts the CUDA wheels
# but not every transitive dependency, so PyPI has to stay reachable. Both
# 2.9.0+cu130 and 0.24.0+cu130 sort ABOVE their plain PyPI builds, so pip picks
# the CUDA ones.
#
# torchvision 0.24.0 is the only release whose metadata requires torch==2.9.0.
# transformers must stay 4.x -- VRMBG-3.0's model.py runs under
# trust_remote_code=True against whatever is installed. pillow 11.3.0 is what
# proper_pixel.py's shim is written for. einops + timm are imported by the
# model's own model.py and declared nowhere.
#
# THREE PINS ARE CEILINGS, NOT PREFERENCES. ubuntu:22.04 ships Python 3.10, and
# these three refuse to install on anything below 3.11:
#
#     numpy   2.3.4  needs >=3.11  ->  2.2.6   last release that allows 3.10
#     kornia  0.8.3  needs >=3.11  ->  0.8.2
#     av     18.1.0  needs >=3.11  ->  17.1.0
#
# The original numbers came from the host, which runs Python 3.13, where all
# three resolve fine. On 3.10 pip fails outright with "No matching distribution
# found" -- it is not a preference that can be ignored. Checked every pin against
# 3.10 with the PyPI JSON API rather than fixing them one build at a time.
#
# The alternative is to move the base to ubuntu:24.04 (Python 3.12) and keep the
# original pins; torch 2.9.0+cu130 does ship cp312 wheels. Not done here, to keep
# the base image as chosen.
#
# The torch install is its own layer on purpose: it is ~2.5 GB, and a change to
# any pin below must not force a re-download of it.
# ---------------------------------------------------------------------------
RUN python -m pip install --user --no-cache-dir --upgrade pip && \
    python -m pip install --user --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
        "torch==2.9.0" "torchvision==0.24.0"

RUN python -m pip install --user --no-cache-dir \
        "numpy==2.2.6" \
        "pillow==11.3.0" \
        "transformers==4.57.3" \
        "timm==1.0.29" \
        "einops==0.8.2" \
        "kornia==0.8.2" \
        "opencv-python-headless==5.0.0.93" \
        "fastapi==0.141.1" \
        "uvicorn==0.52.3" \
        "websocket-client==1.9.0" \
        "PyYAML==6.0.3" \
        "tqdm==4.67.1" \
        "av==17.1.0" \
        "safetensors==0.6.2"

# ---------------------------------------------------------------------------
# The two clonable checkouts, pinned by full SHA
#
# `git init` + `fetch --depth 1 origin <sha>` rather than `clone --depth 1`:
# a shallow clone has no history, so checking out an older pinned commit fails.
# Fetching the exact SHA at depth 1 is precise and cheap, against 536 MB and
# 303 MB for the full trees (mostly .git and image assets).
#
# Only the package folder is kept. Each bridge puts its checkout root on
# sys.path, and PixelOE is a src layout, so `src` is what goes on it.
# ---------------------------------------------------------------------------
RUN git init -q /tmp/ppa && \
    git -C /tmp/ppa remote add origin https://github.com/KennethJAllen/proper-pixel-art && \
    git -C /tmp/ppa fetch -q --depth 1 origin ad690b494bcbc455285ac6b7832a73e3ebd4675b && \
    git -C /tmp/ppa checkout -q FETCH_HEAD && \
    mkdir -p /opt/vendor/proper-pixel-art && \
    cp -r /tmp/ppa/proper_pixel_art /opt/vendor/proper-pixel-art/ && \
    rm -rf /tmp/ppa

RUN git init -q /tmp/pixeloe && \
    git -C /tmp/pixeloe remote add origin https://github.com/KohakuBlueleaf/PixelOE && \
    git -C /tmp/pixeloe fetch -q --depth 1 origin 1d45ba0b5c51c3d998b19043a168366a8f170eaa && \
    git -C /tmp/pixeloe checkout -q FETCH_HEAD && \
    mkdir -p /opt/vendor/PixelOE/src && \
    cp -r /tmp/pixeloe/src/pixeloe /opt/vendor/PixelOE/src/ && \
    rm -rf /tmp/pixeloe

# ---------------------------------------------------------------------------
# The snapper WASM -- cloned, then compiled
#
# `pkg/` is gitignored upstream, so a clone never contains it; the Rust
# toolchain is required. It is installed, used, and then removed inside this one
# RUN, so none of it survives into the layer -- the same discipline as deleting
# a downloaded CUDA installer in the same RUN.
# ---------------------------------------------------------------------------
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --profile minimal --default-toolchain 1.98.1 && \
    . "$HOME/.cargo/env" && \
    mkdir -p "$HOME/.local/bin" && \
    curl -sSfL \
      "https://github.com/rustwasm/wasm-pack/releases/download/v0.15.0/wasm-pack-v0.15.0-x86_64-unknown-linux-musl.tar.gz" \
      | tar -xz -C "$HOME/.local/bin" --strip-components=1 \
          "wasm-pack-v0.15.0-x86_64-unknown-linux-musl/wasm-pack" && \
    git clone --depth 1 https://github.com/Hugo-Dz/spritefusion-pixel-snapper /tmp/snapper && \
    cd /tmp/snapper && \
    wasm-pack build --target web --out-dir pkg --release && \
    test -f pkg/spritefusion_pixel_snapper.js && \
    test -f pkg/spritefusion_pixel_snapper_bg.wasm && \
    mkdir -p /opt/vendor/spritefusion-pixel-snapper && \
    cp -r /tmp/snapper/pkg /opt/vendor/spritefusion-pixel-snapper/pkg && \
    cd / && rm -rf /tmp/snapper "$HOME/.rustup" "$HOME/.cargo" \
                      "$HOME/.local/bin/wasm-pack" "$HOME/.cache"

# ---------------------------------------------------------------------------
# The matting weights, downloaded from HuggingFace
#
# Four of the repo's seven files. pytorch_model.bin is the same weights in the
# older pickle format -- 845 MB for nothing, because transformers prefers
# safetensors. README.md / .gitattributes / .git are not read at runtime.
#
# -x/-s are 4 for the small files (16 connections against 211 bytes is silly)
# and 16 for the 845 MB one, where the parallelism actually pays.
#
# The revision is pinned in the URL. `resolve/main/` would silently track
# whatever the default branch happens to be; `resolve/<sha>/` matches the pin.
#
# If aria2c ever fails here with "only one auth mechanism allowed", the CDN is
# rejecting the forwarded Authorization header alongside its own signed query
# string -- swap this for huggingface_hub, which is already installed as a
# transformers dependency:
#   snapshot_download(repo_id=..., revision=..., local_dir=..., allow_patterns=[...])
#
# WHY THE retry() WRAPPER EXISTS -- measured 2026-09-25, not defensive habit.
# A --no-cache build died on the *first* file, config.json, after 0.5 s with:
#   aria2c errorCode=1 ... SSL/TLS handshake failure:
#   The TLS connection was non-properly terminated.
# That reads like a broken URL or a rejected token. It is neither. Re-probing
# the identical URL with the identical --header and the identical aria2c flags
# in a fresh ubuntu:22.04 fetched all four files cleanly (211 B / 232 B / 23 KB
# and the full 884915688 B safetensors), so the failure was a transient TLS
# reset -- the kind a long `--no-cache` build is exposed to because every layer
# re-downloads. Without the wrapper a single blip throws away the whole build,
# which by then has already paid for apt + torch + ~1 GB of CUDA wheels.
#
# -c (continue) means a retry resumes rather than restarts, so the 845 MB file
# does not go back to zero. Six attempts with linear backoff (5 s..25 s) is well
# under the cost of one rebuild.
# ---------------------------------------------------------------------------
ARG VRMBG_REV=59716e19a6cc97f91311edea190938219c097b76

# HF_TOKEN arrives from the caller's environment via
# `--secret id=hf_token,env=HF_TOKEN` and is exposed to this RUN only. It is not
# an ARG and not an ENV, so it never reaches `docker history` or the image
# config. The `${HF_TOKEN}` below is expanded by the shell at build time, not by
# the Dockerfile parser, which is why the instruction text in the image history
# contains the literal placeholder instead of the token.
#
# TWO guards, and they cover DIFFERENT failures -- measured, because the split
# is not obvious and each one looks redundant until you test it:
#   * `required=true` fires only when the --secret flag is not passed at all:
#     "ERROR: failed to build: secret hf_token: not found".
#   * it does NOT fire when the flag IS passed but HF_TOKEN is unset or empty.
#     BuildKit hands the RUN an empty value and the build would sail on into an
#     unauthenticated download; the -z check below is the only thing that stops
#     it. Verified with --no-cache on this same mount (same
#     mechanism): unset token -> "HF_TOKEN is empty", exit 1.
# Delete either guard and you lose a real failure mode.
#
# A third case worth knowing: BuildKit resolves secrets AFTER the cache lookup,
# so a CACHED download layer builds fine with no token at all. That is correct
# -- the layer already contains the model -- but it means a cached build proves
# nothing about your token.
RUN --mount=type=secret,id=hf_token,env=HF_TOKEN,required=true \
    set -eu; \
    if [ -z "${HF_TOKEN:-}" ]; then echo "HF_TOKEN is empty" >&2; exit 1; fi; \
    BASE="https://huggingface.co/briaai/VRMBG-3.0/resolve/${VRMBG_REV}"; \
    DEST=/opt/models/VRMBG-3.0; \
    mkdir -p "$DEST"; \
    retry() { \
        n=1; \
        while [ "$n" -le 6 ]; do \
            if "$@"; then return 0; fi; \
            echo "  attempt $n/6 failed, retrying in $((n * 5))s" >&2; \
            sleep $((n * 5)); \
            n=$((n + 1)); \
        done; \
        echo "  gave up after 6 attempts" >&2; \
        return 1; \
    }; \
    for f in config.json vrmbg3_config.py model.py; do \
        retry aria2c --console-log-level=error -c -x 4 -s 4 -k 1M \
               --header="Authorization: Bearer ${HF_TOKEN}" \
               "${BASE}/${f}" -d "$DEST" -o "${f}"; \
    done; \
    retry aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
           --header="Authorization: Bearer ${HF_TOKEN}" \
           "${BASE}/model.safetensors" -d "$DEST" -o model.safetensors; \
    ls -l "$DEST"

# ---------------------------------------------------------------------------
# Runtime environment
#
# PIXELOE_BACKEND=torch: PixelOE defaults to "auto" and would otherwise probe
# for slangpy, which is deliberately not installed.
#
# *_OFFLINE: the weights are baked in, so nothing should reach the Hub at run
# time. Set AFTER the download above for exactly that reason.
# ---------------------------------------------------------------------------
ENV SPRITE_MODEL_DIR=/opt/models/VRMBG-3.0 \
    SPRITE_PPA_DIR=/opt/vendor/proper-pixel-art \
    PIXELOE_DIR=/opt/vendor/PixelOE \
    SPRITE_SNAPPER_DIR=/opt/vendor/spritefusion-pixel-snapper \
    PIXELOE_BACKEND=torch \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# ---------------------------------------------------------------------------
# The app -- cloned from its own repository (public, no token needed)
#
# Plain `git clone` of the public repo. No token in the URL, so there is no
# credential to scrub afterwards (the `.git` removal below stays: the image
# does not need the history).
#
# What lands in the image is the default branch's HEAD *as of the
# build* -- which only holds if CACHEBUST was passed, per the header. The
# resolved commit is written to .sprite_rev so the running app can report what
# it is rather than guessing.
#
# GITHUB_TOKEN used to be consumed here when the repo was private, same
# mechanism as HF_TOKEN. Not needed anymore -- public clone below.
#
# `docker_selfcheck.py` is copied from the build context rather than cloned,
# because it is newer than anything committed to the repo. So the context needs
# that ONE file -- it does not need the rest of the app.
# ---------------------------------------------------------------------------
# ALWAYS THE LATEST SOURCE -- no pinned SHA (user instruction, 2026-09-25).
#
# The hazard a pin was guarding against is real and unchanged: BuildKit caches
# this layer under a key that does not include "what the branch points at now",
# so a plain `git clone` of a branch is re-served from cache forever and the
# image silently freezes at the first snapshot. Measured 2026-09-25 -- the
# editLink fix was pushed, the rebuild reported success, re-exported the SAME
# image id, and the image still contained the old code.
#
# So the freshness has to come from a cache-buster instead of a pin. CACHEBUST
# is declared immediately above the clone, which means
#
#   --build-arg CACHEBUST=$(date +%s)
#
# invalidates THIS layer plus the four cheap ones after it (COPY, mkdir,
# self-check, WORKDIR) and nothing before it. The ~6-minute apt, pip and model
# layers stay cached, so the rebuild costs about 15 seconds.
#
# Omit the flag and you get the cached clone -- i.e. stale code, silently. The
# tell is unchanged, and worth repeating: if the image id does not move, nothing
# was rebuilt. Check what actually landed with:
#
#   docker image ls --no-trunc TostAI-Sprite-Sheet-Studio
#   docker run --rm --entrypoint sh TostAI-Sprite-Sheet-Studio -c \
#     'sed -n "/function editLink/,/^}/p" /app/sprite_studio/ui.html'
#
# The resolved commit is recorded in .sprite_rev so the running app can report
# what it is (GET /api/update) instead of guessing.
ARG CACHEBUST=0
RUN set -eu; \
    git clone --depth 1 \
      "https://github.com/camenduru/TostAI-Sprite-Sheet-Studio.git" \
      /app/sprite_studio; \
    git -C /app/sprite_studio rev-parse HEAD > /app/sprite_studio/.sprite_rev; \
    rm -rf /app/sprite_studio/.git; \
    test -f /app/sprite_studio/app.py; \
    echo "cloned camenduru/TostAI-Sprite-Sheet-Studio at $(cat /app/sprite_studio/.sprite_rev)"

# --chown IS LOAD-BEARING, not tidiness. `USER camenduru` is set above, and the
# `chown -R`/`chmod -R` that fix up /app happen BEFORE this COPY -- so without
# it this one file lands as root:root 755 and the running app (uid 1000) cannot
# overwrite it. That is invisible until something tries to: the studio's Update
# button copies the repo's tree over this directory and died with
#   PermissionError: [Errno 13] Permission denied:
#   '/app/sprite_studio/docker_selfcheck.py'
# returning a 500 that the dialog could not even parse. Every other file in the
# app dir comes from the git clone and is already camenduru-owned; this COPY was
# the only root-owned path in it.
COPY --chown=camenduru:camenduru docker_selfcheck.py /app/sprite_studio/docker_selfcheck.py

# app.py computes WORKSPACE as the *parent* of its own folder, and the editor's
# asset browser lists WORKSPACE/walk and WORKSPACE/sprites.
RUN mkdir -p /app/walk /app/sprites /app/sprite_studio/runs /app/sprite_studio/uploads

# ---------------------------------------------------------------------------
# Build-time proof. Imports every optional dependency the way the app does,
# asserts the four external folders resolved, and loads the model for real
# (BiRefNet, ~220M params). A truncated download or a bad clone fails the build
# here rather than 500ing on the first palette click.
# ---------------------------------------------------------------------------
RUN python /app/sprite_studio/docker_selfcheck.py

# Must come after the clone: `CMD python app.py` is relative to this directory,
# and without it the container starts in /app and dies looking for /app/app.py.
WORKDIR /app/sprite_studio

EXPOSE 8765

# --host 0.0.0.0 because app.py defaults to 127.0.0.1, which inside a container
# is only reachable from inside the container.
CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "8765"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/api/defaults', timeout=4)"
