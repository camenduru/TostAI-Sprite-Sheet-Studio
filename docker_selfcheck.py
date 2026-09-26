#!/usr/bin/env python
"""Build-time proof for the Sprite Studio image.

Run by the Dockerfile after the matting model and the three vendored checkouts
have been placed. Imports every optional dependency exactly the way the app does,
asserts the four external folders resolved, and loads the model for real -- the
difference between a build that fails loudly and an image that starts fine and
then 500s the moment someone presses a palette node.

Path-agnostic on purpose: the app directory is derived from this file's own
location and the model directory from SPRITE_MODEL_DIR, so the same script works
for any layout (`Dockerfile` uses /app/sprite_studio + /opt/models/VRMBG-3.0;
other images may use /content/...).

Kept as a file rather than an inline heredoc because several targets run it, and
a heredoc would have to be duplicated in each.
"""
import os
import sys

APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)
os.chdir(APP_DIR)
print("app dir      ", APP_DIR)

import torch, torchvision, transformers, timm, einops, kornia
import cv2, numpy, PIL, fastapi, uvicorn, yaml, av, tqdm

print("python       ", sys.version.split()[0])
print("torch        ", torch.__version__, "| cuda", torch.version.cuda,
      "| device available at build time:", torch.cuda.is_available())
print("torchvision  ", torchvision.__version__)
print("transformers ", transformers.__version__)
print("timm         ", timm.__version__)
print("opencv       ", cv2.__version__)
print("pillow       ", PIL.__version__)
print("numpy        ", numpy.__version__)

# The pins the model and the shim depend on. A silent drift here is what makes
# trust_remote_code load fail at run time, so fail the build instead.
assert torch.__version__.startswith("2.9.0"), torch.__version__
assert torchvision.__version__.startswith("0.24.0"), torchvision.__version__
assert transformers.__version__.startswith("4."), transformers.__version__
assert PIL.__version__.startswith("11."), PIL.__version__

import pipeline as P
import editor as E

model_dir = P.DEFAULT_CFG["model_dir"]
print("model_dir    ", model_dir)
assert os.path.isdir(model_dir), "model_dir does not exist: " + model_dir

# All four, not just the two the loader obviously needs: model.py imports its
# config relatively, so a set missing vrmbg3_config.py fails at load time.
for name in ("config.json", "vrmbg3_config.py", "model.py", "model.safetensors"):
    path = os.path.join(model_dir, name)
    assert os.path.isfile(path), "model file missing: " + name
    print("  %-20s %10d bytes" % (name, os.path.getsize(path)))

# Existence is NOT enough, and this was learned the hard way: the original check
# only asserted config.json and model.safetensors were present, so a build could
# go green while `trust_remote_code=True` was still unable to execute the model.
# Load it for real -- it is a few seconds and it is the actual contract.
from transformers import AutoModelForImageSegmentation

_net = AutoModelForImageSegmentation.from_pretrained(
    model_dir, trust_remote_code=True)
_params = sum(p.numel() for p in _net.parameters())
print("model loads  ", type(_net).__name__, "%.1fM params" % (_params / 1e6))
assert _params > 200e6, "suspiciously small model: %d params" % _params
del _net

# The three editor ops, each probed the way its palette node probes it.
import proper_pixel, pixeloe_bridge, pixel_snapper

for name, fn in (("proper-pixel-art", proper_pixel.available),
                 ("PixelOE",          pixeloe_bridge.available),
                 ("pixel-snapper",    pixel_snapper.available)):
    ok = fn()
    print("%-16s %s" % (name, "available" if ok else "NOT AVAILABLE"))
    assert ok, "%s did not resolve -- check its *_DIR env var" % name

print("build self-check OK")
