#!/usr/bin/env python3
# Text-to-video: 9:16, 4–10s, cinematic
# Usage:
#   export REPLICATE_API_TOKEN=your_token_here
#   python generate_village_1816.py
#
# Optional: Override the default model with an environment variable
#   MODEL_ID="runwayml/gen3a-turbo" python generate_village_1816.py


import os
import sys
import replicate
from pathlib import Path
import requests

# --- CONFIGURATION ---
OUT_PATH = Path("village_1816.mp4")
FPS = 24
DURATION_SECONDS = 4 # Default 4s, increase to 10s if model/account supports it

# Choose a model. Luma Dream Machine is excellent for cinematic quality.
# You can override with MODEL_ID env var (e.g., "cogvideo/synthia", "runwayml/gen3a-turbo").
MODEL_ID = os.getenv("MODEL_ID", "luma/dream-machine-v1-5")

# Refined prompt for cinematic quality and motion
PROMPT = (
    "Wide establishing shot of a 19th-century European village at dawn, 1816. "
    "Heavy ash-gray clouds fully block the sun. Frost clings to tiled rooftops and dirt roads. "
    "Villagers in wool coats stand still, looking upward, cold breath visible. "
    "Muted color palette with desaturated blues and grays. Volumetric fog drifting through the streets. "
    "Cinematic depth of field, subtle camera push-in. Ominous, unnatural silence. "
    "Photoreal, UE5-style realism, 9:16 portrait."
)

# Negative prompt to avoid common AI artifacts and unwanted elements
NEGATIVE_PROMPT = (
    "no text, no captions, no watermark, no bloom, no lens flares, "
    "no dramatic sunlight, no bright colors, no crowds, no movement from the villagers"
)

def run_text_to_video(model_id: str, prompt: str, negative_prompt: str, duration: int) -> Path:
    print(f"-> Using model: {model_id}")
    print(f"-> Generating {duration}s video at {FPS} FPS...")

    try:
        if model_id.startswith("luma/dream-machine"):
            # Luma API signature
            output = replicate.run(
                model_id,
                input={
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "duration": duration,
                    "aspect_ratio": "9:16",
                    "output_format": "mp4",
                    "fps": FPS,
                },
            )
        elif model_id.startswith("cogvideo/synthia"):
            # CogVideo API signature (uses frame count)
            num_frames = duration * FPS
            output = replicate.run(
                model_id,
                input={
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "num_frames": num_frames,
                    "fps": FPS,
                    "image_size": "854x480", # 9:16 portrait
                    "guidance_scale": 7.5,
                    "num_inference_steps": 50,
                },
            )
        elif model_id.startswith("runwayml/gen3a-turbo"):
            # Runway API signature
            output = replicate.run(
                model_id,
                input={
                    "prompt": prompt,
                    "watermark": False,
                    "duration": duration,
                    "aspect_ratio": "9:16",
                }
            )
        else:
            # Fallback for other models (may not support all parameters)
            print("WARNING: Falling back to basic prompt generation for this model.")
            output = replicate.run(model_id, input={"prompt": prompt})

        # The API returns a URL
        video_url = str(output)
        print(f"-> Generation complete. Downloading from: {video_url}")

        # Stream the download
        with requests.get(video_url, stream=True) as r:
            r.raise_for_status()
            with open(OUT_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        return OUT_PATH

    except replicate.exceptions.ReplicateError as e:
        print(f"ERROR: Replicate API call failed: {e}")
        print("Please check your model, parameters, and API token.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    if not os.getenv("REPLICATE_API_TOKEN"):
        print("ERROR: REPLICATE_API_TOKEN environment variable not set.")
        print("Please set it to your token from https://replicate.com/account/api-tokens")
        sys.exit(1)
        
    print("--- Starting Video Generation ---")
    print("Prompt:", PROMPT[:100] + "...")
    print("-" * 30)
    
    out_path = run_text_to_video(MODEL_ID, PROMPT, NEGATIVE_PROMPT, DURATION_SECONDS)
    
    print("-" * 30)
    print("✅ Success! Video saved to:", out_path.resolve())
