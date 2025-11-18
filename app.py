# app.py
from flask import Flask, request, render_template, send_file, jsonify
import json
import os
from utils.bedrock_utils import generate_text, generate_image
from utils.ppt_utils import create_ppt
import uuid

app = Flask(__name__)
os.makedirs("output", exist_ok=True)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate():
    topic = request.form.get('topic', '').strip()
    slides = int(request.form.get('slides', 5))

    if not topic:
        return "Please provide a topic", 400

    # Create outline — force strict JSON-only output from the model
    prompt = f"""
You are an expert slide designer. Create exactly {slides} slide objects for the topic below.
Return ONLY a valid JSON array (no extra text) with this structure:

[
  {{
    "title": "Slide title",
    "bullets": ["short bullet 1", "short bullet 2"]
  }}
  ...
]

Topic: {topic}
Keep each title short (6-8 words) and each bullets array max 5 items.
"""
    try:
        outline_text = generate_text(prompt)
    except Exception as e:
        app.logger.exception("Text generation failed")
        return f"Text generation failed: {e}", 500

    # parse JSON robustly (in case model returned surrounding text)
    try:
        data = json.loads(outline_text)
    except Exception:
        # try to extract first JSON array found in text
        start = outline_text.find('[')
        end = outline_text.rfind(']')
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(outline_text[start:end+1])
            except Exception as e:
                app.logger.exception("Failed to parse JSON from model output")
                return f"Failed to parse model output as JSON: {e}", 500
        else:
            app.logger.error("No JSON array found in model output")
            return "Model did not return a JSON array. Inspect logs.", 500

    # Validate data is a list
    if not isinstance(data, list):
        return "Model output is not a list of slides", 500

    # Generate 1 image per slide
    for idx, s in enumerate(data):
        title = s.get('title', f'Slide {idx+1}')
        img_prompt = s.get('image_prompt') or f"{title} — educational infographic, flat, simple, labels, no text overlay"
        try:
            img_bytes = generate_image(img_prompt)
        except Exception as e:
            app.logger.exception("Image generation failed for slide %s", title)
            # continue without image instead of failing entire job
            s['image_path'] = None
            continue

        img_path = f"output/{uuid.uuid4().hex}.png"
        try:
            with open(img_path, 'wb') as f:
                f.write(img_bytes)
            s['image_path'] = img_path
        except Exception as e:
            app.logger.exception("Failed saving image to disk")
            s['image_path'] = None

    # Create PPT
    try:
        ppt_path = create_ppt(data)
    except Exception as e:
        app.logger.exception("Failed to create PPT")
        return f"Failed to create PPT: {e}", 500

    return send_file(ppt_path, as_attachment=True)


if __name__ == '__main__':
    # production: remove debug=True
    app.run(host='0.0.0.0', port=5000, debug=True)
