---
title: AI Virtual Interior Designer
emoji: 🏡
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.0.0
app_file: app/gradio_app.py
pinned: false
license: mit
---

# AI-Virtual-Stager 🏡

A tool to transform empty room photos into virtual staged interiors using Stable Diffusion and ControlNet.

## Project Structure
- `data/`: Raw and processed room photos.
- `models/`: ControlNet weights.
- `src/`: Core logic for image processing and generation.
- `app/`: Streamlit dashboard for the user interface.

## Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Run the app: `streamlit run app/main.py`
