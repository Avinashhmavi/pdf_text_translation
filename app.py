from flask import Flask, render_template, request, send_file, jsonify, make_response
import fitz  # PyMuPDF
import re
import os
import requests
import json
import time
import tempfile
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import threading
import uuid
import shutil

app = Flask(__name__)

# Load OpenAI API Key from environment variables (set by Render from /etc/secrets/.env)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Ensure it's set in /etc/secrets/.env on Render.")

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
MODEL_NAME = "gpt-4o"

# Define translation batch size and concurrency
MAX_BATCH_SIZE = 10
MAX_CONCURRENT_REQUESTS = 5
REQUEST_TIMEOUT = 30  # seconds

# Language Configuration
LANGUAGES = {
    "Hindi": {"code": "hin_Deva", "iso": "hi", "font": "NotoSansDevanagari"},
    "Tamil": {"code": "tam_Taml", "iso": "ta", "font": "NotoSansTamil"},
    "Telugu": {"code": "tel_Telu", "iso": "te", "font": "NotoSansTelugu"}
}

DIGIT_MAP = {
    "Hindi": "०१२३४५६७८९",
    "Tamil": "௦௧௨௩௪௫௬௭௮௯",
    "Telugu": "౦౧౨౩౪౫౬౭౮౯"
}

LATIN_DIGITS = "0123456789"

translation_jobs = {}
local_thread_storage = threading.local()

class TranslationJob:
    def __init__(self, job_id, languages, total_pages):
        self.job_id = job_id
        self.languages = languages
        self.total_pages = total_pages
        self.progress = {lang: 0 for lang in languages}
        self.status = "processing"
        self.output_files = {}
        self.start_time = time.time()
        self.temp_dir = tempfile.mkdtemp()

    def update_progress(self, language, page_count):
        self.progress[language] = page_count
        return self.get_overall_progress()
    
    def get_overall_progress(self):
        if not self.languages:
            return 0
        total = sum(self.progress.values())
        max_possible = len(self.languages) * self.total_pages
        return int((total / max_possible) * 100) if max_possible > 0 else 0
    
    def complete(self, output_files):
        self.status = "completed"
        self.output_files = output_files
        self.end_time = time.time()
        
    def fail(self, error_message):
        self.status = "failed"
        self.error = error_message
        self.end_time = time.time()
    
    def cleanup(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

def log_info(message):
    print(f"[INFO] {message}")

def log_error(message):
    print(f"[ERROR] {message}")

def log_debug(message):
    if app.debug:
        print(f"[DEBUG] {message}")

def extract_preserve_patterns(text):
    patterns = {
        "emails": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "urls": r'https?://\S+|www\.\S+',
        "dates": r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}\b',
        "percentages": r'\b\d+(?:\.\d+)?%\b',
        "money": r'[\$£€¥₹]\s*\d+(?:,\d{3})*(?:\.\d+)?|\b\d+(?:,\d{3})*(?:\.\d+)?\s*[\$£€¥₹]',
        "times": r'\b\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?\b',
        "phone_numbers": r'\b(?:\+\d{1,3}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b',
        "model_numbers": r'\b[A-Z0-9]+-[A-Z0-9]+\b|\b[A-Z]+\d+[A-Z]*\b',
        "measurements": r'\b\d+(?:\.\d+)?\s*(?:mm|cm|m|km|in|ft|yd|mi|g|kg|lb|oz|ml|l|gal)\b',
        "product_codes": r'\b[A-Z]{2,}\d{2,}\b'
    }
    
    matches = {}
    for pattern_name, pattern in patterns.items():
        found = re.finditer(pattern, text, re.IGNORECASE)
        for match in found:
            matched_text = match.group(0)
            matches[matched_text] = {"type": pattern_name, "span": match.span()}
    
    return matches

def replace_with_placeholders(text, entities):
    auto_entities = extract_preserve_patterns(text)
    all_entities = {entity: {"type": "user_specified", "span": None} for entity in entities if entity.strip()}
    all_entities.update(auto_entities)
    
    sorted_entities = sorted(all_entities.keys(), key=len, reverse=True)
    modified_text = text
    placeholder_map = {}
    
    for entity in sorted_entities:
        if not entity or entity not in modified_text:
            continue
        placeholder = f"__PRESERVE{len(placeholder_map):03d}__"
        placeholder_map[placeholder] = entity
        modified_text = modified_text.replace(entity, placeholder)
        log_debug(f"Preserved: '{entity}' -> '{placeholder}'")
    
    return modified_text, placeholder_map

def restore_placeholders(text, placeholder_map):
    result = text
    for placeholder, original in placeholder_map.items():
        if placeholder in result:
            result = result.replace(placeholder, original)
        else:
            result += f" {original}"
    return result

def convert_numbers_to_script(text, target_lang):
    if target_lang not in DIGIT_MAP:
        return text
        
    digit_map = DIGIT_MAP[target_lang]
    
    def replace_digit(match):
        number = match.group(0)
        if "__PRESERVE" in number or number.endswith("__"):
            return number
        return ''.join(digit_map[int(d)] if d in LATIN_DIGITS else d for d in number)
        
    pattern = re.compile(r'(?<!__PRESERVE\d{3}__)\b\d+(?:\.\d+)?\b(?![^_]*__)')
    return pattern.sub(replace_digit, text)

def translate_text_batch(text_batch, target_lang, retry_count=2):
    if not text_batch:
        return []
        
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    system_prompt = f"""
    Translate the following text accurately to {target_lang}.
    Follow these strict requirements:
    1. Preserve ALL placeholders like __PRESERVExxx__ exactly as they appear
    2. Keep technical terms where appropriate
    3. Maintain paragraph structure and formatting
    4. Return ONLY the translation without explanations
    5. If unsure about any part, preserve the original content
    """
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n\n".join(text_batch)}
        ],
        "temperature": 0.2,
        "max_tokens": 4000
    }
    
    for attempt in range(retry_count + 1):
        try:
            response = requests.post(
                OPENAI_API_URL,
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            
            result = response.json()
            translated_content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            log_info(f"Successfully translated batch to {target_lang}")
            
            if len(text_batch) > 1:
                splits = translated_content.split("\n\n")
                if len(splits) < len(text_batch):
                    if re.search(r'^\d+\.\s+', translated_content):
                        splits = re.split(r'\n\d+\.\s+', '\n' + translated_content)[1:]
            else:
                splits = [translated_content]
                
            if len(splits) != len(text_batch):
                log_error(f"Mismatch in translation batch: expected {len(text_batch)}, got {len(splits)}")
                if len(splits) < len(text_batch):
                    splits.extend(["" for _ in range(len(text_batch) - len(splits))])
                else:
                    splits = splits[:len(text_batch)]
            
            return [s.strip() for s in splits]
            
        except requests.exceptions.RequestException as e:
            log_error(f"API error on attempt {attempt+1}: {str(e)} - Response: {response.text if 'response' in locals() else 'No response'}")
            if attempt < retry_count:
                time.sleep(2 * (attempt + 1))
            else:
                return text_batch
                
        except Exception as e:
            log_error(f"Unexpected error on attempt {attempt+1}: {str(e)}")
            if attempt < retry_count:
                time.sleep(2 * (attempt + 1))
            else:
                return text_batch
    
    return text_batch

def process_translations_with_threading(texts, target_lang, entities):
    if not texts:
        return []
        
    batch_size = min(MAX_BATCH_SIZE, len(texts))
    batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
    log_info(f"Processing {len(texts)} texts in {len(batches)} batches for {target_lang}")
    
    all_results = []
    placeholder_maps = []
    
    for i, text in enumerate(texts):
        modified_text, placeholder_map = replace_with_placeholders(text, entities)
        texts[i] = modified_text
        placeholder_maps.append(placeholder_map)
    
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
        batch_results = list(executor.map(
            lambda batch: translate_text_batch(batch, target_lang),
            batches
        ))
    
    for batch in batch_results:
        all_results.extend(batch)
    
    for i, (translated, placeholder_map) in enumerate(zip(all_results, placeholder_maps)):
        restored = restore_placeholders(translated, placeholder_map)
        all_results[i] = convert_numbers_to_script(restored, target_lang)
    
    return all_results

def extract_pdf_components(pdf_path):
    doc = fitz.open(pdf_path)
    components = []
    total_pages = len(doc)
    
    log_info(f"Extracting content from PDF: {total_pages} pages")
    
    for page_num, page in enumerate(doc):
        page_dict = page.get_text("dict")
        blocks = page_dict.get("blocks", [])
        
        text_blocks = []
        for b in blocks:
            if b["type"] == 0:
                lines = []
                for line in b.get("lines", []):
                    spans = line.get("spans", [])
                    if spans:
                        line_text = " ".join([span.get("text", "").strip() for span in spans])
                        if line_text.strip():
                            lines.append({
                                "text": line_text,
                                "font": spans[0].get("font", ""),
                                "font_size": spans[0].get("size", 10),
                                "bbox": line["bbox"],
                                "color": spans[0].get("color", 0)
                            })
                
                if lines:
                    text_blocks.append({
                        "bbox": b["bbox"],
                        "lines": lines
                    })
        
        components.append({
            "page_num": page_num,
            "blocks": text_blocks,
            "size": (page.rect.width, page.rect.height),
            "rotation": page.rotation
        })
    
    doc.close()
    return components, total_pages

def translate_pdf_components(components, target_lang, entities, job_id=None):
    all_texts = []
    text_map = []
    
    for page_idx, page in enumerate(components):
        for block_idx, block in enumerate(page["blocks"]):
            for line_idx, line in enumerate(block["lines"]):
                all_texts.append(line["text"])
                text_map.append((page_idx, block_idx, line_idx))
    
    log_info(f"Translating {len(all_texts)} text elements to {target_lang}")
    translated_texts = process_translations_with_threading(all_texts, target_lang, entities)
    
    for (page_idx, block_idx, line_idx), translated_text in zip(text_map, translated_texts):
        components[page_idx]["blocks"][block_idx]["lines"][line_idx]["translated"] = translated_text
        
        if job_id and job_id in translation_jobs:
            job = translation_jobs[job_id]
            current_page = page_idx + 1
            job.update_progress(target_lang, current_page)
    
    return components

def rebuild_translated_pdf(components, original_pdf_path, output_path, target_lang):
    input_doc = fitz.open(original_pdf_path)
    output_doc = fitz.open()
    
    log_info(f"Rebuilding PDF with {target_lang} translations: {output_path}")
    
    lang_config = LANGUAGES.get(target_lang, {})
    lang_font = lang_config.get("font", "helv")
    lang_iso = lang_config.get("iso", "")
    
    for page_data in components:
        page_num = page_data["page_num"]
        input_page = input_doc[page_num]
        
        new_page = output_doc.new_page(
            width=page_data["size"][0],
            height=page_data["size"][1]
        )
        
        new_page.show_pdf_page(
            new_page.rect,
            input_doc,
            page_num
        )
        
        for block in page_data["blocks"]:
            rect = fitz.Rect(block["bbox"])
            new_page.draw_rect(rect, color=(1, 1, 1), fill=(1, 1, 1))
        
        for block in page_data["blocks"]:
            for line in block["lines"]:
                if "translated" in line and line["translated"].strip():
                    rect = fitz.Rect(line["bbox"])
                    font_size = line["font_size"]
                    
                    try:
                        text_options = {
                            "fontname": lang_font,
                            "fontsize": font_size,
                            "color": (0, 0, 0)
                        }
                        
                        new_page.insert_text(
                            (rect.x0, rect.y0 + font_size * 0.8),
                            line["translated"],
                            **text_options
                        )
                    except Exception as e:
                        log_error(f"Error inserting text: {str(e)}")
                        html = f'<span lang="{lang_iso}" style="font-size:{font_size}pt">{line["translated"]}</span>'
                        new_page.insert_htmlbox(rect, html)
    
    output_doc.save(output_path, garbage=4, deflate=True)
    input_doc.close()
    output_doc.close()
    log_info(f"Completed PDF rebuild for {target_lang}")
    
    return output_path

def translate_pdf(job_id, pdf_path, entities, languages):
    try:
        components, total_pages = extract_pdf_components(pdf_path)
        log_info(f"Extracted {total_pages} pages from PDF")
        
        if job_id in translation_jobs:
            translation_jobs[job_id].total_pages = total_pages
        
        output_files = {}
        
        for lang in languages:
            try:
                log_info(f"Starting translation to {lang}")
                translated_components = translate_pdf_components(
                    components.copy(),
                    lang,
                    entities,
                    job_id
                )
                
                job = translation_jobs.get(job_id, None)
                output_dir = job.temp_dir if job else os.path.dirname(pdf_path)
                output_path = os.path.join(output_dir, f"translated_{lang}_{job_id}.pdf")
                
                rebuild_translated_pdf(
                    translated_components,
                    pdf_path,
                    output_path,
                    lang
                )
                
                output_files[lang] = output_path
                log_info(f"Translation to {lang} completed: {output_path}")
                
            except Exception as e:
                log_error(f"Error translating to {lang}: {str(e)}")
                continue
        
        if job_id in translation_jobs:
            if output_files:
                translation_jobs[job_id].complete(output_files)
            else:
                translation_jobs[job_id].fail("Failed to generate any translations")
        
        return output_files
        
    except Exception as e:
        log_error(f"Error in translation process: {str(e)}")
        if job_id in translation_jobs:
            translation_jobs[job_id].fail(str(e))
        raise

@app.route('/')
def index():
    return render_template('index.html', languages=LANGUAGES.keys())

@app.route('/translate', methods=['POST'])
def translate():
    if 'pdf' not in request.files:
        return jsonify({'error': 'No PDF file provided'}), 400

    pdf_file = request.files['pdf']
    if pdf_file.filename == '' or not pdf_file:
        return jsonify({'error': 'No file selected or invalid file'}), 400
    
    pdf_file.seek(0, os.SEEK_END)
    if pdf_file.tell() == 0:
        return jsonify({'error': 'Empty PDF file'}), 400
    pdf_file.seek(0)

    languages = request.form.getlist('languages')
    entities = request.form.get('entities', '').split(',')
    
    if not languages:
        return jsonify({'error': 'No languages selected'}), 400

    job_id = str(uuid.uuid4())
    temp_input = os.path.join(tempfile.gettempdir(), f"input_{job_id}.pdf")
    pdf_file.save(temp_input)
    
    job = TranslationJob(job_id, languages, 0)
    translation_jobs[job_id] = job
    
    executor = ThreadPoolExecutor(max_workers=1)
    executor.submit(translate_pdf, job_id, temp_input, entities, languages)
    
    return jsonify({'job_id': job_id}), 202

@app.route('/progress/<job_id>')
def get_progress(job_id):
    job = translation_jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
        
    response = {
        'status': job.status,
        'progress': job.get_overall_progress(),
        'languages': {lang: prog for lang, prog in job.progress.items()}
    }
    
    if job.status == 'completed':
        response['files'] = {lang: f'/download/{job_id}/{lang}' for lang in job.output_files}
    elif job.status == 'failed':
        response['error'] = getattr(job, 'error', 'Unknown error')
        
    return jsonify(response)

@app.route('/download/<job_id>/<language>')
def download_file(job_id, language):
    job = translation_jobs.get(job_id)
    if not job or job.status != 'completed' or language not in job.output_files:
        return jsonify({'error': 'File not found'}), 404
        
    file_path = job.output_files[language]
    response = send_file(
        file_path,
        as_attachment=True,
        download_name=f'translated_{language}_{os.path.basename(file_path)}'
    )
    
    if all(lang in job.output_files for lang in job.languages):
        job.cleanup()
        translation_jobs.pop(job_id, None)
        
    return response

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
