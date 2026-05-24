---
title: AI Virtual Interior Designer
emoji: 🏡
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.15.0
app_file: app/gradio_app.py
pinned: false
license: mit
hardware: zero-a10g
---

# AI-Virtual-Stager 🏡

A tool to transform empty room photos into virtual staged interiors using Stable Diffusion and ControlNet.

**Powered by ZeroGPU** — runs Stable Diffusion on NVIDIA A10G for free!

## Project Structure
- `data/`: Raw and processed room photos.
- `models/`: ControlNet weights.
- `src/`: Core logic for image processing and generation.
- `app/`: Gradio interface powered by ZeroGPU.

## Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Run the app: `python app/gradio_app.py`
