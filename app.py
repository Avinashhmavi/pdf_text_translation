import os
import time
from flask import Flask, request, render_template, send_file, jsonify
import fitz  # PyMuPDF
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import re
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration (unchanged from your code)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

MODEL_NAME = "facebook/nllb-200-3.3B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LANGUAGES = {
    "Hindi": {"code": "hin_Deva", "iso": "hi"},
    "Tamil": {"code": "tam_Taml", "iso": "ta"},
    "Telugu": {"code": "tel_Telu", "iso": "te"}
}
MAX_LENGTH_DEFAULT = 256
MEMORY_THRESHOLD = 0.8

DIGIT_MAP = {
    "Hindi": "०१२३४५६७८९",
    "Tamil": "௦௧௨௩௪௫௬௭௮௯",
    "Telugu": "౦౧౨౩౪౫౬౭౮౯"
}
LATIN_DIGITS = "0123456789"

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Initialize model (unchanged)
def initialize_model():
    print("🔄 Initializing translation model...")
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    ).to(DEVICE).eval()
    print(f"✅ Model loaded in {time.time()-start:.2f}s")
    return tokenizer, model

tokenizer, model = initialize_model()

# Utility functions (unchanged)
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
    print(f"🔍 Modified text with placeholders: '{modified_text}'")
    return modified_text, placeholder_map

def convert_numbers_to_script(text, target_lang):
    digit_map = DIGIT_MAP[target_lang]
    def replace_digit(match):
        number = match.group(0)
        converted = ''.join(digit_map[int(d)] if d in LATIN_DIGITS else d for d in number)
        return converted
    pattern = re.compile(r'(?<!__PRESERVE\d{3}__)\b\d+(?:\.\d+)?\b(?![^_]*__)')
    converted_text = pattern.sub(replace_digit, text)
    return converted_text

def get_dynamic_batch_size(num_texts, fast_mode):
    if DEVICE != "cuda":
        return min(8, num_texts)
    total_memory = torch.cuda.get_device_properties(0).total_memory
    free_memory = total_memory - torch.cuda.memory_allocated()
    tokens_per_text = MAX_LENGTH_DEFAULT
    bytes_per_text = tokens_per_text * 4 * (3 if fast_mode else 1)
    max_batch = max(1, min(free_memory // bytes_per_text, num_texts))
    return min(16 if fast_mode else 4, max_batch)

def translate_batch(texts, target_lang_code, fast_mode=False):
    if not texts:
        return []
    batch_size = get_dynamic_batch_size(len(texts), fast_mode)
    translated_texts = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        max_length = max(MAX_LENGTH_DEFAULT, max(len(t.split()) for t in batch) * 2)
        try:
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(DEVICE)
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_lang_code),
                    max_length=max_length,
                    num_beams=3 if fast_mode else 1,
                    use_cache=True,
                    early_stopping=True
                )
            translated = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            translated_texts.extend([re.sub(r'^\.+|\s*\.+$|^\s*…', '', t.strip()) for t in translated])
            del inputs, outputs
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
        except RuntimeError as e:
            print(f"⚠️ Memory error: {e}. Reducing batch size and retrying...")
            if batch_size > 1:
                batch_size = max(1, batch_size // 2)
                translated_texts.extend(translate_batch(batch, target_lang_code, fast_mode))
            else:
                raise
    return translated_texts

def reset_gpu_memory():
    global model, tokenizer
    if DEVICE == "cuda":
        del model
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print("🔄 Refreshing GPU memory...")
        start = time.time()
        tokenizer, model = initialize_model()
        print(f"✅ GPU memory refreshed in {time.time()-start:.2f}s")

def check_memory_and_reset(total_pages):
    if DEVICE != "cuda" or total_pages <= 5:
        return False
    total_memory = torch.cuda.get_device_properties(0).total_memory
    allocated_memory = torch.cuda.memory_allocated()
    if allocated_memory / total_memory > MEMORY_THRESHOLD:
        reset_gpu_memory()
        return True
    return False

def join_spans(spans):
    if not spans:
        return ""
    spans = sorted(spans, key=lambda s: s["bbox"][0])
    text_parts = [spans[0]["text"].strip()]
    for i in range(1, len(spans)):
        span1, span2 = spans[i - 1], spans[i]
        d = span2["bbox"][0] - span1["bbox"][2]
        text2 = span2["text"].strip()
        if not text2:
            continue
        if len(span1["text"]) > 0 and len(text2) > 0:
            width1 = span1["bbox"][2] - span1["bbox"][0]
            width2 = span2["bbox"][2] - span2["bbox"][0]
            min_avg_char_width = min(width1 / len(span1["text"]), width2 / len(text2))
            if d < 0.5 * min_avg_char_width or d < 0:
                text_parts.append(text2)
            else:
                text_parts.append(" " + text2)
        else:
            text_parts.append(text2)
    return "".join(text_parts)

def extract_pdf_components(pdf_path):
    print(f"\n📄 Extracting components from {pdf_path}...")
    doc = fitz.open(pdf_path)
    components = []
    for page_num, page in enumerate(doc):
        print(f"\n📖 Processing page {page_num+1}")
        blocks = page.get_text("dict")["blocks"]
        text_blocks = []
        for b in blocks:
            if b["type"] == 0:  # Text block
                lines = []
                for line in b["lines"]:
                    if line["spans"]:
                        text = join_spans(line["spans"])
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
    max_words_per_subblock = 50
    for i, line in enumerate(lines):
        text = line["text"].strip()
        if not text:
            continue
        is_short = len(text.split()) <= 3 and len(text) < 20
        font_size = line["font_size"]
        gap = (lines[i + 1]["y_pos"] - line["y_pos"] - font_size) if i < len(lines) - 1 else font_size
        x_shift = abs(line["x_pos"] - lines[i-1]["x_pos"]) if i > 0 else 0
        current_words = current_subblock["text"].split()
        new_words = text.split()
        if len(current_words + new_words) > max_words_per_subblock or font_size > 20:
            subblocks.append(current_subblock)
            current_subblock = {"text": "", "lines": [], "is_short": False}
        if current_subblock["text"]:
            if (is_short or gap > font_size * 0.5 or x_shift > 10):
                subblocks.append(current_subblock)
                current_subblock = {"text": "", "lines": [], "is_short": False}
        current_subblock["text"] += " " + text if current_subblock["text"] else text
        current_subblock["lines"].append(line)
        current_subblock["is_short"] = is_short and len(current_subblock["lines"]) == 1
        if i == len(lines) - 1 or (is_short and gap > font_size * 0.5):
            subblocks.append(current_subblock)
            current_subblock = {"text": "", "lines": [], "is_short": False}
    return subblocks

def translate_chunk(chunk, entities, target_lang, fast_mode=False):
    target_lang_code = LANGUAGES[target_lang]["code"]
    all_subblocks = []
    for page in chunk:
        for block in page["text_blocks"]:
            subblocks = split_block_into_subblocks(block)
            block["subblocks"] = subblocks
            all_subblocks.extend(subblocks)
    if not all_subblocks:
        return
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
        translated_texts = translate_batch(texts, target_lang_code, fast_mode=fast_mode)
        for subblock, translated_text, placeholder_map in zip([sb for sb in all_subblocks if sb["text"].strip()], translated_texts, placeholder_maps):
            for placeholder, original in placeholder_map.items():
                if placeholder in translated_text:
                    translated_text = translated_text.replace(placeholder, original)
                    print(f"🔄 Restored '{original}' in '{translated_text}'")
                else:
                    translated_text += f" {original}"
                    print(f"🔄 Appended '{original}' to '{translated_text}'")
            translated_text = convert_numbers_to_script(translated_text, target_lang)
            print(f"🔢 Converted numbers in '{translated_text}'")
            subblock["translated_text"] = translated_text

def rebuild_pdf(components, target_lang, output_path, original_pdf_path, use_white_background=False):
    print(f"\n🏗️ Rebuilding {target_lang} PDF...")
    doc = fitz.open(original_pdf_path)
    lang_iso = LANGUAGES[target_lang]["iso"]
    for page_data in components:
        page = doc[page_data["page_num"]]
        links = list(page.get_links())
        for block in page_data["text_blocks"]:
            original_bbox = fitz.Rect(block["bbox"])
            if use_white_background:
                page.draw_rect(original_bbox, color=(1, 1, 1), fill=(1, 1, 1), fill_opacity=1.0)
            else:
                page.add_redact_annot(original_bbox)
                page.apply_redactions()
            for subblock in block["subblocks"]:
                if "translated_text" not in subblock or not subblock["translated_text"].strip():
                    continue
                translated_lines = redistribute_translated_text(subblock["translated_text"], subblock["lines"])
                for i, (original_line, translated_line) in enumerate(zip(subblock["lines"], translated_lines)):
                    line_rect = fitz.Rect(original_line["line_bbox"])
                    font_size = original_line["font_size"]
                    if translated_line.strip():
                        html = f'<div style="width: 100%; height: 100%; padding: 0; margin: 0;"><p lang="{lang_iso}" style="margin: 0; padding: 0;">{translated_line}</p></div>'
                        css = f"p {{ font-size: {font_size}pt; }}"
                        try:
                            page.insert_htmlbox(line_rect, html, css=css, scale_low=0, rotate=0, oc=0, opacity=1, overlay=True)
                            print(f"✓ Inserted line {i+1} at {line_rect.top_left}: '{translated_line[:30]}...'")
                        except Exception as e:
                            print(f"⚠️ Error inserting line {i+1} at {line_rect.top_left}: {e}")
        for link in links:
            page.insert_link(link)
            print(f"🔗 Restored link to: {link.get('uri', 'unknown destination')}")
    print(f"💾 Saving to {output_path}")
    doc.save(output_path, garbage=4, deflate=True)
    doc.close()

def redistribute_translated_text(translated_text, original_lines):
    if not original_lines or not translated_text.strip():
        return [""] * len(original_lines)
    translated_words = translated_text.split()
    translated_lines = []
    word_idx = 0
    default_font = fitz.Font("helv")
    for line in original_lines:
        max_width = line["line_bbox"][2] - line["line_bbox"][0]
        font_size = line["font_size"]
        current_line = []
        current_width = 0
        while word_idx < len(translated_words):
            word = translated_words[word_idx]
            word_width = default_font.text_length(word + " ", fontsize=font_size)
            if current_width + word_width <= max_width:
                current_line.append(word)
                current_width += word_width
                word_idx += 1
            else:
                break
        translated_lines.append(" ".join(current_line) if current_line else "")
    while len(translated_lines) < len(original_lines):
        translated_lines.append("")
    if word_idx < len(translated_words):
        remaining_text = " ".join(translated_words[word_idx:])
        translated_lines[-1] = translated_lines[-1] + " " + remaining_text if translated_lines[-1] else remaining_text
    return translated_lines

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate_pdf():
    if 'pdf_file' not in request.files:
        return jsonify({'error': 'No PDF file uploaded'}), 400
    
    pdf_file = request.files['pdf_file']
    entities_input = request.form.get('entities', '')
    languages_input = request.form.get('languages', 'Hindi,Tamil,Telugu')
    
    if pdf_file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filename = secure_filename(pdf_file.filename)
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    pdf_file.save(pdf_path)

    entities = parse_user_entities(entities_input)
    languages = parse_user_languages(languages_input)
    components = extract_pdf_components(pdf_path)
    total_pages = len(components)
    fast_mode = total_pages <= 5
    output_files = []

    for lang in languages:
        start_time = time.time()
        print(f"\n🚀 Starting {lang} translation")
        if fast_mode:
            translate_chunk(components, entities, lang, fast_mode=True)
            print(f"✅ Translated {total_pages} pages in one pass")
        else:
            chunk_size = 2
            num_chunks = (total_pages + chunk_size - 1) // chunk_size
            for i in range(0, total_pages, chunk_size):
                check_memory_and_reset(total_pages)
                chunk = components[i:i + chunk_size]
                translate_chunk(chunk, entities, lang, fast_mode=False)
                print(f"✅ Chunk {i // chunk_size + 1}/{num_chunks} translated ({len(chunk)} pages)")
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], f"translated_{lang}_{filename}")
        rebuild_pdf(components, lang, output_path, pdf_path, use_white_background=False)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            output_files.append(output_path)
        print(f"\n✅ {lang} translation completed in {time.time()-start_time:.2f}s")

    if not output_files:
        return jsonify({'error': 'No translated PDFs generated'}), 500
    
    return jsonify({
        'message': f"Translation completed for {', '.join(languages)}",
        'files': [os.path.basename(f) for f in output_files]
    })

@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)
