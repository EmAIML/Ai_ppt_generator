# utils/bedrock_utils.py
import boto3
import json
import base64
import os
import logging

# Use certifi certificate bundle to avoid Windows/Anaconda SSL issues
# boto3.client accepts 'verify' param which can be a path to a CA bundle
REGION = os.environ.get('AWS_REGION', 'ap-south-1')
TEXT_MODEL = os.environ.get('BEDROCK_TEXT_MODEL', 'anthropic.claude-3-haiku-20240307-v1')
IMAGE_MODEL = os.environ.get('BEDROCK_IMAGE_MODEL', 'amazon.titan-image-generator-v1')

# Create bedrock client with explicit cert bundle
bedrock = boto3.client('bedrock-runtime', region_name=REGION,)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _read_body(resp):
    """
    Utility to decode response body bytes safely.
    """
    body_bytes = resp['body'].read()
    try:
        return body_bytes.decode('utf-8')
    except Exception:
        # fallback: try latin1
        return body_bytes.decode('latin1', errors='ignore')


# ---------- TEXT ----------
def generate_text(prompt, max_tokens=800):
    """
    Call Bedrock text model and return the model text output (string).
    Tries to handle a few common bedrock response shapes.
    """
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens
    }

    try:
        resp = bedrock.invoke_model(modelId=TEXT_MODEL, body=json.dumps(body), contentType='application/json')
    except Exception as e:
        log.exception("Bedrock invoke_model (text) failed")
        raise

    raw = _read_body(resp)

    # Try parse as JSON
    try:
        parsed = json.loads(raw)
    except Exception:
        # Not JSON — return raw text
        return raw

    # parsed is JSON — handle common formats:
    # 1) { "content": [ { "type": "output_text", "text": "..." } ] }
    if isinstance(parsed, dict):
        # Anthropic-like
        if 'content' in parsed and isinstance(parsed['content'], list):
            first = parsed['content'][0]
            # some shapes: {'type': 'output_text', 'text': '...'}
            if isinstance(first, dict):
                if 'text' in first:
                    return first['text']
                # sometimes nested differently
                if 'content' in first:
                    return first['content']
        # Amazon-style: {"output": "text..."} or other keys
        if 'output' in parsed and isinstance(parsed['output'], str):
            return parsed['output']
        # if there's a top-level 'message' or similar
        for key in ('message', 'text', 'response'):
            if key in parsed and isinstance(parsed[key], str):
                return parsed[key]

    # If none matched, return stringified JSON
    return json.dumps(parsed)


# ---------- IMAGE ----------
def generate_image(prompt, width=1024, height=1024, num_images=1):
    """
    Call Bedrock image model. Return raw image bytes for the first image.
    Handles a few common response formats (base64 string, dict with b64_json, etc.)
    """
    # A flexible image-generation request body to fit various Titan/stability shapes
    body = {
        "taskType": "TEXT_IMAGE",
        "textToImageParams": {
            "text": prompt,
            "style": "flat infographic",
            "negativeText": "",
            "cfgScale": 7.0,
            "seed": 0
        },
        "imageGenerationConfig": {
            "numberOfImages": num_images,
            "quality": "standard",
            "width": width,
            "height": height
        }
    }

    try:
        resp = bedrock.invoke_model(modelId=IMAGE_MODEL, body=json.dumps(body), contentType='application/json')
    except Exception as e:
        log.exception("Bedrock invoke_model (image) failed")
        raise

    raw = _read_body(resp)

    # Try parse JSON
    try:
        parsed = json.loads(raw)
    except Exception:
        # not JSON — maybe raw base64
        try:
            return base64.b64decode(raw)
        except Exception:
            raise ValueError("Image response is neither JSON nor base64")

    # parsed is JSON — common shapes:
    # { "images": ["<base64>"] }
    if isinstance(parsed, dict):
        # common key 'images'
        if 'images' in parsed and isinstance(parsed['images'], list) and parsed['images']:
            b64 = parsed['images'][0]
            if isinstance(b64, dict) and 'b64_json' in b64:
                b64 = b64['b64_json']
            return base64.b64decode(b64)
        # some models return 'artifacts'
        if 'artifacts' in parsed and isinstance(parsed['artifacts'], list) and parsed['artifacts']:
            art = parsed['artifacts'][0]
            # artifact could be dict with 'base64' or 'b64_json' or 'data'
            if isinstance(art, dict):
                for k in ('b64_json', 'base64', 'data'):
                    if k in art:
                        return base64.b64decode(art[k])
            if isinstance(art, str):
                return base64.b64decode(art)
    # fallback: try to find a base64-looking string anywhere
    def find_b64(obj):
        if isinstance(obj, str):
            # rough check: long string with base64 chars
            if len(obj) > 200 and all(c.isalnum() or c in '+/=\n' for c in obj):
                return obj
        if isinstance(obj, dict):
            for v in obj.values():
                res = find_b64(v)
                if res:
                    return res
        if isinstance(obj, list):
            for it in obj:
                res = find_b64(it)
                if res:
                    return res
        return None

    candidate = find_b64(parsed)
    if candidate:
        return base64.b64decode(candidate)

    raise ValueError("Could not parse image bytes from model response")
