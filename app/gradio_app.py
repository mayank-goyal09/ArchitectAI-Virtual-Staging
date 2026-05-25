import os
import cv2
import numpy as np
import gradio as gr
from PIL import Image
from huggingface_hub import InferenceClient
import io

# === HF Inference API Client ===
# Uses HF's free serverless GPU infrastructure — no local GPU needed!
import base64
import requests

HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or None

# Proactively load token from .env if not present in environment
if not HF_TOKEN:
    try:
        if os.path.exists(".env"):
            with open(".env") as f:
                for line in f:
                    if "HUGGINGFACE_TOKEN" in line or "HF_TOKEN" in line:
                        parts = line.split("=")
                        if len(parts) >= 2:
                            # Strip quotes, whitespace and trailing comments
                            val = parts[1].split("#")[0].strip().strip('"').strip("'")
                            if val and val != "your_token_here":
                                HF_TOKEN = val
                                break
    except Exception as e:
        print(f"⚠️ Error reading .env file: {e}")

if HF_TOKEN:
    safe_token_display = f"...{HF_TOKEN[-4:]}" if len(HF_TOKEN) > 4 else "loaded"
    print(f"🔑 Hugging Face Token loaded successfully ({safe_token_display})")
else:
    print("⚠️ No Hugging Face Token found. Requests may be rate-limited or fail.")

# Initialize the InferenceClient (without provider="hf-inference" to avoid strict validation where possible)
client = InferenceClient(token=HF_TOKEN)
print("✅ Connected to Hugging Face Inference API!")


def call_hf_api_direct(image_bytes, prompt, model_id, token):
    """Direct HTTP POST request to Hugging Face Inference API, bypassing huggingface_hub client-side bugs."""
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
        
    print(f"🌐 Direct API call to: {url}")
    
    # Try Format 1: Standard JSON payload with base64 encoded image
    try:
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")
        payload = {
            "inputs": encoded_image,
            "parameters": {
                "prompt": prompt,
            }
        }
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code == 200 and len(response.content) > 100:
            return Image.open(io.BytesIO(response.content))
        else:
            print(f"Format 1 failed with status {response.status_code}: {response.text[:200]}")
    except Exception as e:
        print(f"Format 1 exception: {e}")

    # Try Format 2: Raw binary image bytes as body, prompt in parameters query param
    try:
        response = requests.post(
            url,
            headers={**headers, "Content-Type": "image/jpeg"},
            params={"prompt": prompt},
            data=image_bytes,
            timeout=30
        )
        if response.status_code == 200 and len(response.content) > 100:
            return Image.open(io.BytesIO(response.content))
        else:
            print(f"Format 2 failed with status {response.status_code}: {response.text[:200]}")
    except Exception as e:
        print(f"Format 2 exception: {e}")

    # Try Format 3: Multipart form data
    try:
        response = requests.post(
            url,
            headers=headers,
            files={"image": ("image.jpg", image_bytes, "image/jpeg")},
            data={"prompt": prompt},
            timeout=30
        )
        if response.status_code == 200 and len(response.content) > 100:
            return Image.open(io.BytesIO(response.content))
        else:
            print(f"Format 3 failed with status {response.status_code}: {response.text[:200]}")
            if response.status_code != 200:
                raise Exception(f"HF API returned status {response.status_code}: {response.text[:200]}")
    except Exception as e:
        print(f"Format 3 exception: {e}")
        raise e

    raise Exception("All direct HF API transmission formats failed.")
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

    # Convert PIL.Image to JPEG bytes for robust API transmission
    buffer = io.BytesIO()
    original.save(buffer, format="JPEG")
    image_bytes = buffer.getvalue()

    # Define models
    model_to_use = "Qwen/Qwen-Image-Edit-2511"
    fallback_model = "timbrooks/instruct-pix2pix"

    try:
        # First attempt: Try standard high-level client with selected model
        print(f"Attempting design using {model_to_use} via huggingface_hub client...")
        result_image = client.image_to_image(
            image=image_bytes,
            prompt=full_prompt,
            model=model_to_use,
        )
        print("✅ Room design complete (high-level client)!")
        return result_image, skeleton
    except Exception as client_err:
        client_err_str = str(client_err)
        print(f"⚠️ High-level client failed for {model_to_use}: {client_err_str}")
        
        # Second attempt: Try direct HTTP request for the selected model
        try:
            print(f"Attempting design using {model_to_use} via direct HTTP request...")
            result_image = call_hf_api_direct(image_bytes, full_prompt, model_to_use, HF_TOKEN)
            print("✅ Room design complete (direct HTTP)!")
            return result_image, skeleton
        except Exception as direct_err:
            direct_err_str = str(direct_err)
            print(f"⚠️ Direct HTTP failed for {model_to_use}: {direct_err_str}")
            
            # Third attempt: Fallback to the highly compatible free serverless model "timbrooks/instruct-pix2pix"
            print(f"🔄 Falling back to highly compatible model: {fallback_model}...")
            try:
                # We try standard high-level first for the fallback model
                try:
                    result_image = client.image_to_image(
                        image=image_bytes,
                        prompt=full_prompt,
                        model=fallback_model,
                    )
                    print("✅ Room design complete (fallback via high-level client)!")
                    return result_image, skeleton
                except Exception:
                    # Fallback to direct HTTP for the fallback model
                    result_image = call_hf_api_direct(image_bytes, full_prompt, fallback_model, HF_TOKEN)
                    print("✅ Room design complete (fallback via direct HTTP)!")
                    return result_image, skeleton
            except Exception as fallback_err:
                fallback_err_str = str(fallback_err)
                print(f"❌ All attempts failed. Fallback error: {fallback_err_str}")
                
                # Format a user-friendly error message based on failure reason
                if "rate limit" in fallback_err_str.lower() or "429" in fallback_err_str:
                    raise gr.Error("⏳ Rate limit reached. Please wait 30 seconds and try again.")
                elif "401" in fallback_err_str or "token" in fallback_err_str.lower() or "unauthorized" in fallback_err_str.lower():
                    raise gr.Error("🔑 HF Token needed or invalid. Check your HUGGINGFACE_TOKEN in the .env file.")
                else:
                    raise gr.Error(f"API Error: {fallback_err_str}")


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
    demo.launch(server_name="0.0.0.0", server_port=port, share=False, ssr_mode=False)
