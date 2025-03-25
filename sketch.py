import json
from urllib import request
import re
import sys

if len(sys.argv) != 3:
    print("Usage: python sketch.py {input_name.png} {job_id}")
    sys.exit(1)

input_name = sys.argv[1]
output_name = input_name.split(".")[0]
job_id = sys.argv[2]

print(f"Input image: {output_name}")

prompt_text = """
{
  "3": {
    "inputs": {
      "seed": 79465383210919,
      "steps": 20,
      "cfg": 8,
      "sampler_name": "euler_ancestral",
      "scheduler": "normal",
      "denoise": 0.44,
      "model": [
        "4",
        0
      ],
      "positive": [
        "6",
        0
      ],
      "negative": [
        "7",
        0
      ],
      "latent_image": [
        "11",
        0
      ]
    },
    "class_type": "KSampler",
    "_meta": {
      "title": "KSampler"
    }
  },
  "4": {
    "inputs": {
      "ckpt_name": "cool_model.safetensors"
    },
    "class_type": "CheckpointLoaderSimple",
    "_meta": {
      "title": "Load Checkpoint"
    }
  },
  "6": {
    "inputs": {
      "text": "A highly detailed hand-drawn pencil sketch, intricate line art, monochrome, delicate and precise ink strokes, cross-hatching, shading with fine lines, elegant contours, vintage etching style, traditional illustration, minimalistic yet expressive, black and white artwork, rough yet refined sketch, concept art, comic-style inking.\n",
      "clip": [
        "4",
        1
      ]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "CLIP Text Encode (Prompt)"
    }
  },
  "7": {
    "inputs": {
      "text": "embedding:verybadimagenegative_v1.3, blurry, low resolution, bad anatomy, deformed, ugly, extra limbs, extra fingers, distorted face, overexposed, extra eyes, harsh lighting, oversaturated, bad composition, unnatural colors.\n",
      "clip": [
        "4",
        1
      ]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "CLIP Text Encode (Prompt)"
    }
  },
  "8": {
    "inputs": {
      "samples": [
        "3",
        0
      ],
      "vae": [
        "4",
        2
      ]
    },
    "class_type": "VAEDecode",
    "_meta": {
      "title": "VAE Decode"
    }
  },
  "10": {
    "inputs": {
      "image": \""""+input_name+""""
    },
    "class_type": "LoadImage",
    "_meta": {
      "title": "Load Image"
    }
  },
  "11": {
    "inputs": {
      "pixels": [
        "10",
        0
      ],
      "vae": [
        "4",
        2
      ]
    },
    "class_type": "VAEEncode",
    "_meta": {
      "title": "VAE Encode"
    }
  },
  "17": {
    "inputs": {
      "filename_prefix": \"sketch_"""+output_name+"""\",
      "images": [
        "8",
        0
      ]
    },
    "class_type": "SaveImage",
    "_meta": {
      "title": "Save Image"
    }
  }
}
"""

def escape_json_control_chars(text):    
    return re.sub(r'[\x00-\x1F]', '', text)

def queue_prompt(prompt):
    p = {"prompt": prompt, "job_id": job_id}  # Pass job_id separately
    data = json.dumps(p).encode('utf-8')
    req = request.Request("http://127.0.0.1:8188/prompt", data=data)
    request.urlopen(req)
    
prompt_text = escape_json_control_chars(prompt_text)
print(prompt_text)

prompt = json.loads(prompt_text)

# You can modify parameters here
# For example, change the positive prompt
# prompt["6"]["inputs"]["text"] = "A hand-drawn pencil sketch"

# Change the seed
# prompt["3"]["inputs"]["seed"] = 42

# Change the image input if needed
# Note: You need to make sure the image is already uploaded to ComfyUI or accessible by it
# prompt["10"]["inputs"]["image"] = "your_new_image.jpg"

queue_prompt(prompt)