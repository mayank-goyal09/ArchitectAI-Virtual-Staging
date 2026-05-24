import os
import cv2
import numpy as np
import gradio as gr
from PIL import Image
from huggingface_hub import InferenceClient
import io

# === HF Inference API Client ===
# Uses HF's free serverless GPU infrastructure — no local GPU needed!
HF_TOKEN = os.environ.get("HF_TOKEN", None)
client = InferenceClient(token=HF_TOKEN)

print("✅ Connected to Hugging Face Inference API!")


# === IMAGE PROCESSING (runs locally on CPU — lightweight) ===
def get_canny_skeleton(pil_image):
    """Extract structural edges from the room image."""
    img_np = np.array(pil_image)
    if len(img_np.shape) == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    elif img_np.shape[2] == 4:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
    else:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    gray = cv2.equalizeHist(cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY))
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    return Image.fromarray(cv2.cvtColor(dilated, cv2.COLOR_GRAY2RGB))


# === INFERENCE VIA HF API ===
def ai_interior_designer(input_img, custom_prompt, creativity_level, style_strength):
    if input_img is None:
        raise gr.Error("Please upload an image of your empty room first!")
    if not custom_prompt or not custom_prompt.strip():
        raise gr.Error("Please describe your design vision!")

    # 1. Resize and extract structure
    original = input_img.convert("RGB").resize((512, 512))
    skeleton = get_canny_skeleton(original)

    # 2. Build the full prompt
    full_prompt = f"{custom_prompt}, professional interior design, highly detailed, photorealistic, 8k"

    print(f"🎨 Sending to HF API: '{full_prompt}'...")

    # 3. Call HF Inference API — image-to-image using instruct-pix2pix
    # This model takes an image + instruction and modifies it (perfect for staging!)
    try:
        result_image = client.image_to_image(
            image=original,
            prompt=full_prompt,
            model="timbrooks/instruct-pix2pix",
            guidance_scale=10.0,
            image_guidance_scale=float(style_strength) * 2.5,
        )
        print("✅ Room design complete!")
        return result_image, skeleton

    except Exception as e:
        error_msg = str(e)
        print(f"❌ API Error: {error_msg}")

        if "rate limit" in error_msg.lower() or "429" in error_msg:
            raise gr.Error("⏳ Rate limit reached. Please wait 30 seconds and try again.")
        elif "401" in error_msg or "token" in error_msg.lower():
            raise gr.Error("🔑 HF Token needed. Add HF_TOKEN in Space Settings → Variables.")
        else:
            raise gr.Error(f"API Error: {error_msg}")


# === GRADIO UI ===
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🏗️ AI Virtual Interior Designer
    Transform empty rooms into professionally staged spaces using AI.
    
    *Powered by Hugging Face Inference API — no GPU required!*
    """)

    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="pil", label="Upload Your Empty Room")
            prompt = gr.Textbox(
                placeholder="e.g., Add modern Scandinavian furniture with light wood and plants",
                label="Describe Your Vision",
                value="Add modern minimalist furniture, warm lighting, indoor plants"
            )
            with gr.Accordion("Advanced Settings", open=False):
                creativity = gr.Slider(0.1, 1.0, value=0.8, label="Creativity Level")
                strictness = gr.Slider(0.1, 1.0, value=0.6, label="Style Strength")
            btn = gr.Button("🎨 Design My Room", variant="primary")

        with gr.Column(scale=1):
            output_img = gr.Image(label="Your Professionally Staged Room")
            skeleton_out = gr.Image(label="Detected Room Structure")

    btn.click(
        fn=ai_interior_designer,
        inputs=[input_img, prompt, creativity, strictness],
        outputs=[output_img, skeleton_out]
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)
