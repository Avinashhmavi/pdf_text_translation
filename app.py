from flask import Flask, render_template, request, send_file, jsonify
import fitz  # PyMuPDF
import re
import os
import requests
from io import BytesIO

app = Flask(__name__)

# Grok API Configuration
GROQ_API_KEY = "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"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "gemma2-9b-it"  # Using Gemma2 9B Instruct model

# Language Configuration
LANGUAGES = {
    "Hindi": {"code": "hin_Deva", "iso": "hi"},
    "Tamil": {"code": "tam_Taml", "iso": "ta"},
    "Telugu": {"code": "tel_Telu", "iso": "te"}
}
DIGIT_MAP = {
    "Hindi": "०१२३४५६७८९",
    "Tamil": "௦௧௨௩௪௫௬௭௮௯",
    "Telugu": "౦౧౨౩౪౫౬౭౮౯"
}
LATIN_DIGITS = "0123456789"

# Utility functions
def parse_user_entities(user_input):
    entities = [e.strip() for e in user_input.split(',') if e.strip()]
    print(f"📌 Entities to preserve: {', '.join(entities) if entities else 'None'}")
    return sorted(set(entities), key=len, reverse=True)

def parse_user_languages(user_input):
    selected = [lang.strip().capitalize() for lang in user_input.split(',')]
    valid = [lang for lang in selected if lang in LANGUAGES]
    if not valid:
        print("⚠️ No valid languages selected. Using all available.")
        return list(LANGUAGES.keys())
    print(f"🌍 Selected languages: {', '.join(valid)}")
    return valid

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
        matches = pattern.findall(modified_text)
        for match in matches:
            placeholder = f"__PRESERVE{len(placeholder_map):03d}__"
            placeholder_map[placeholder] = match
            modified_text = modified_text.replace(match, placeholder)
    for entity in entities:
        pattern = re.compile(r'\b' + re.escape(entity) + r'\b', re.IGNORECASE)
        def replacer(match):
            original = match.group()
            placeholder = f"__PRESERVE{len(placeholder_map):03d}__"
            placeholder_map[placeholder] = original
            return placeholder
        modified_text, count = pattern.subn(replacer, modified_text)
        if count > 0:
            print(f"🔧 Replaced '{entity}' {count} time(s)")
    return modified_text, placeholder_map

def convert_numbers_to_script(text, target_lang):
    digit_map = DIGIT_MAP[target_lang]
    def replace_digit(match):
        number = match.group(0)
        converted = ''.join(digit_map[int(d)] if d in LATIN_DIGITS else d for d in number)
        return converted
    pattern = re.compile(r'(?<!__PRESERVE\d{3}__)\b\d+(?:\.\d+)?\b(?![^_]*__)')
    return pattern.sub(replace_digit, text)

def translate_batch(texts, target_lang):
    if not texts:
        return []
    translated_texts = []
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    for text in texts:
        try:
            payload = {
                "model": MODEL_NAME,
                "messages": [
                    {
                        "role": "system",
                        "content": f"Translate the following text to {target_lang} accurately. Return only the translated text without additional formatting or metadata."
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                "max_tokens": 500,
                "temperature": 0.1
            }
            
            response = requests.post(GROQ_API_URL, headers=headers, json=payload)
            response.raise_for_status()  # Raise exception for bad status codes
            result = response.json()
            
            # Debug logging
            print(f"Raw API response: {result}")
            
            # Extract translated text from Grok's response
            translated = result.get("choices", [{}])[0].get("message", {}).get("content", text)
            cleaned_text = re.sub(r'^\.+|\s*\.+$|^\s*…', '', translated.strip())
            translated_texts.append(cleaned_text)
            
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Translation error: {str(e)}")
            translated_texts.append(text)  # Fallback to original text
        except Exception as e:
            print(f"⚠️ Unexpected error: {str(e)}")
            translated_texts.append(text)  # Fallback to original text
            
    return translated_texts

# PDF processing functions
def extract_pdf_components(pdf_path):
    doc = fitz.open(pdf_path)
    components = []
    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        text_blocks = []
        for b in blocks:
            if b["type"] == 0:
                lines = []
                for line in b["lines"]:
                    if line["spans"]:
                        text = "".join(span["text"].strip() for span in line["spans"])
                        if text.strip():
                            lines.append({
                                "text": text,
                                "y_pos": line["spans"][0]["origin"][1],
                                "x_pos": line["spans"][0]["origin"][0],
                                "font_size": line["spans"][0]["size"],
                                "line_bbox": line["bbox"]
                            })
                if lines:
                    text_blocks.append({"bbox": b["bbox"], "lines": lines})
        components.append({"page_num": page_num, "text_blocks": text_blocks, "size": (page.rect.width, page.rect.height)})
    doc.close()
    return components

def split_block_into_subblocks(block):
    lines = block["lines"]
    if not lines:
        return []
    subblocks = []
    current_subblock = {"text": "", "lines": [], "is_short": False}
    for i, line in enumerate(lines):
        text = line["text"].strip()
        if not text:
            continue
        current_subblock["text"] += " " + text if current_subblock["text"] else text
        current_subblock["lines"].append(line)
        subblocks.append(current_subblock)
        current_subblock = {"text": "", "lines": [], "is_short": False}
    return subblocks

def translate_chunk(chunk, entities, target_lang):
    all_subblocks = []
    for page in chunk:
        for block in page["text_blocks"]:
            subblocks = split_block_into_subblocks(block)
            block["subblocks"] = subblocks
            all_subblocks.extend(subblocks)

    texts = []
    placeholder_maps = []
    for subblock in all_subblocks:
        original_text = subblock["text"]
        if not original_text.strip():
            subblock["translated_text"] = ""
            continue
        modified_text, placeholder_map = replace_with_placeholders(original_text, entities)
        texts.append(modified_text)
        placeholder_maps.append(placeholder_map)

    if texts:
        translated_texts = translate_batch(texts, target_lang)
        for subblock, translated_text, placeholder_map in zip(
            [sb for sb in all_subblocks if sb["text"].strip()], translated_texts, placeholder_maps
        ):
            for placeholder, original in placeholder_map.items():
                if placeholder in translated_text:
                    translated_text = translated_text.replace(placeholder, original)
                else:
                    translated_text += f" {original}"
            translated_text = convert_numbers_to_script(translated_text, target_lang)
            subblock["translated_text"] = translated_text

def rebuild_pdf(components, target_lang, output_path, original_pdf_path):
    doc = fitz.open(original_pdf_path)
    lang_iso = LANGUAGES[target_lang]["iso"]
    for page_data in components:
        page = doc[page_data["page_num"]]
        for block in page_data["text_blocks"]:
            original_bbox = fitz.Rect(block["bbox"])
            page.add_redact_annot(original_bbox)
            page.apply_redactions()
            for subblock in block["subblocks"]:
                if "translated_text" not in subblock or not subblock["translated_text"].strip():
                    continue
                for line, translated_line in zip(subblock["lines"], [subblock["translated_text"]]):
                    line_rect = fitz.Rect(line["line_bbox"])
                    font_size = line["font_size"]
                    if translated_line.strip():
                        html = f'<p lang="{lang_iso}" style="margin: 0; padding: 0;">{translated_line}</p>'
                        css = f"p {{ font-size: {font_size}pt; }}"
                        page.insert_htmlbox(line_rect, html, css=css, overlay=True)
    doc.save(output_path, garbage=4, deflate=True)
    doc.close()

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    if 'pdf_file' not in request.files:
        return jsonify({"error": "No PDF file uploaded"}), 400
    
    pdf_file = request.files['pdf_file']
    entities_input = request.form.get('entities', '')
    languages_input = request.form.get('languages', 'Hindi,Tamil,Telugu')
    
    pdf_data = pdf_file.read()
    pdf_path = "temp.pdf"
    with open(pdf_path, "wb") as f:
        f.write(pdf_data)

    entities = parse_user_entities(entities_input)
    languages = parse_user_languages(languages_input)
    components = extract_pdf_components(pdf_path)
    
    output_files = []
    for lang in languages:
        try:
            translate_chunk(components, entities, lang)
            output_path = f"translated_{lang}.pdf"
            rebuild_pdf(components, lang, output_path, pdf_path)
            if os.path.exists(output_path):
                output_files.append(output_path)
            else:
                print(f"⚠️ Failed to generate {output_path}")
        except Exception as e:
            print(f"⚠️ Error processing {lang}: {str(e)}")
            continue

    if not output_files:
        return jsonify({"error": "Translation failed"}), 500

    return send_file(
        output_files[0],
        as_attachment=True,
        download_name=f"translated_{languages[0]}.pdf",
        mimetype='application/pdf'
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
