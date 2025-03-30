import os
import time
import re
import requests
from flask import Flask, request, render_template, send_file, jsonify
import fitz  # PyMuPDF
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Hugging Face API Configuration
HF_API_KEY = "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"
HF_API_URL = "https://api-inference.huggingface.co/models/facebook/nllb-200-distilled-600M"
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

LANGUAGES = {
    "Hindi": {"token_id": 256047, "code": "hin_Deva", "iso": "hi"},
    "Tamil": {"token_id": 256157, "code": "tam_Taml", "iso": "ta"},
    "Telugu": {"token_id": 256082, "code": "tel_Telu", "iso": "te"}
}
MAX_LENGTH_DEFAULT = 512

DIGIT_MAP = {
    "Hindi": "०१२३४५६७८९",
    "Tamil": "௦௧௨௩௪௫௬௭௮௯",
    "Telugu": "౦౧౨౩౪౫౬౭౮౯"
}
LATIN_DIGITS = "0123456789"

# Utility functions
def parse_user_entities(user_input):
    return sorted({e.strip() for e in user_input.split(',') if e.strip()}, key=len, reverse=True)

def parse_user_languages(user_input):
    selected = [lang.strip().capitalize() for lang in user_input.split(',')]
    valid = [lang for lang in selected if lang in LANGUAGES]
    return valid or list(LANGUAGES.keys())

def replace_with_placeholders(text, entities):
    placeholder_map = {}
    modified_text = text
    patterns = [
        (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'), "emails"),
        (re.compile(r'https?://\S+|www\.\S+'), "URLs"),
        (re.compile(r'@\w+'), "usernames"),
        (re.compile(r'[\$£€%#@&*]\s*\d+(?:\.\d+)?|\d+(?:\.\d+)?[\$£€%#@&*]'), "symbols_with_numbers"),
        (re.compile(r'[\$£€%#@&*]'), "symbols")
    ]
    
    for pattern, _ in patterns:
        for match in pattern.findall(modified_text):
            placeholder = f"__PRESERVE{len(placeholder_map):03d}__"
            placeholder_map[placeholder] = match
            modified_text = modified_text.replace(match, placeholder)

    for entity in entities:
        pattern = re.compile(re.escape(entity), re.IGNORECASE)
        modified_text = pattern.sub(lambda m: f"__PRESERVE{len(placeholder_map):03d}__", modified_text)
        placeholder_map[f"__PRESERVE{len(placeholder_map):03d}__"] = entity

    return modified_text, placeholder_map

def convert_numbers_to_script(text, target_lang):
    digit_map = DIGIT_MAP[target_lang]
    return re.sub(r'\d+', lambda m: ''.join(digit_map[int(d)] for d in m.group()), text)

def translate_batch(texts, target_lang, fast_mode=False):
    if not texts:
        return []
    
    translated_texts = []
    batch_size = 8 if fast_mode else 4
    max_retries = 3
    lang_data = LANGUAGES[target_lang]

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        max_length = max(MAX_LENGTH_DEFAULT, max(len(t.split()) for t in batch) * 2)
        payload = {
            "inputs": batch,
            "parameters": {
                "max_length": max_length,
                "forced_bos_token_id": lang_data["token_id"],
                "src_lang": "eng_Latn",
                "tgt_lang": lang_data["code"]
            }
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(HF_API_URL, headers=HEADERS, json=payload)
                response.raise_for_status()
                result = response.json()

                if isinstance(result, list):
                    translated = [r.get("translation_text", "") for r in result]
                    translated_texts.extend([re.sub(r'^\s*…|\.+$', '', t.strip()) for t in translated])
                    break
                else:
                    if "estimated_time" in result:
                        wait_time = result["estimated_time"] + 5
                        print(f"Model loading, retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    raise ValueError(f"Unexpected response format: {result}")

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 503 and attempt < max_retries - 1:
                    wait_time = 10 * (attempt + 1)
                    print(f"Server overloaded, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                raise

            time.sleep(1)

    return translated_texts

# PDF processing functions (extract_pdf_components, split_block_into_subblocks, 
# translate_chunk, rebuild_pdf, redistribute_translated_text) remain the same as original
# ...

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate_pdf():
    if 'pdf_file' not in request.files:
        return jsonify({'error': 'No PDF file uploaded'}), 400

    pdf_file = request.files['pdf_file']
    if pdf_file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filename = secure_filename(pdf_file.filename)
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    pdf_file.save(pdf_path)

    entities = parse_user_entities(request.form.get('entities', ''))
    languages = parse_user_languages(request.form.get('languages', 'Hindi,Tamil,Telugu'))

    try:
        components = extract_pdf_components(pdf_path)
        output_files = []

        for lang in languages:
            start_time = time.time()
            translate_chunk(components, entities, lang, fast_mode=len(components) <= 5)
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], f"translated_{lang}_{filename}")
            rebuild_pdf(components, lang, output_path, pdf_path)
            output_files.append(output_path)
            print(f"{lang} translation completed in {time.time()-start_time:.2f}s")

        return jsonify({
            'message': f"Translation completed for {', '.join(languages)}",
            'files': [os.path.basename(f) for f in output_files]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    return send_file(file_path, as_attachment=True) if os.path.exists(file_path) else (jsonify({'error': 'File not found'}), 404)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
