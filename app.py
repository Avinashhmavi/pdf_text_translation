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

# Groq API Configuration
GROQ_API_KEY = "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"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "llama3-70b-8192"  # Using a more capable model for translation

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

# Translation status tracking
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
    """Centralized logging function"""
    print(f"[INFO] {message}")

def log_error(message):
    """Centralized error logging function"""
    print(f"[ERROR] {message}")

def log_debug(message):
    """Debug logging function (only when debug is enabled)"""
    if app.debug:
        print(f"[DEBUG] {message}")

# Enhanced entity preservation
def extract_preserve_patterns(text):
    """Extract patterns that should be preserved across translation"""
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
    """Replace entities and patterns with placeholders for preservation"""
    # First extract automatic patterns
    auto_entities = extract_preserve_patterns(text)
    
    # Add user-specified entities
    all_entities = {entity: {"type": "user_specified", "span": None} for entity in entities if entity.strip()}
    all_entities.update(auto_entities)
    
    # Sort entities by length (descending) to avoid partial replacements
    sorted_entities = sorted(all_entities.keys(), key=len, reverse=True)
    
    # Replace with placeholders
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
    """Restore original entities from placeholders"""
    result = text
    for placeholder, original in placeholder_map.items():
        if placeholder in result:
            result = result.replace(placeholder, original)
        else:
            # If placeholder is lost, append the original at the end
            result += f" {original}"
    return result

def convert_numbers_to_script(text, target_lang):
    """Convert Latin digits to the corresponding script's digits"""
    if target_lang not in DIGIT_MAP:
        return text
        
    digit_map = DIGIT_MAP[target_lang]
    
    def replace_digit(match):
        number = match.group(0)
        # Skip if the number is part of a placeholder
        if "__PRESERVE" in number or number.endswith("__"):
            return number
            
        return ''.join(digit_map[int(d)] if d in LATIN_DIGITS else d for d in number)
        
    # Look for numbers not inside placeholders
    pattern = re.compile(r'(?<!__PRESERVE\d{3}__)\b\d+(?:\.\d+)?\b(?![^_]*__)')
    return pattern.sub(replace_digit, text)

def translate_text_batch(text_batch, target_lang, retry_count=2):
    """Translate a batch of texts to the target language"""
    if not text_batch:
        return []
        
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Build system prompt based on target language
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
                GROQ_API_URL, 
                headers=headers, 
                json=payload,
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            
            result = response.json()
            translated_content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # Split the response back into individual translations
            # Check if we need to split (if we sent multiple items)
            if len(text_batch) > 1:
                # Try to split by double newlines
                splits = translated_content.split("\n\n")
                # If we don't get enough splits, try other separators
                if len(splits) < len(text_batch):
                    # Try numbered list format (1. text 2. text)
                    if re.search(r'^\d+\.\s+', translated_content):
                        splits = re.split(r'\n\d+\.\s+', '\n' + translated_content)[1:]
            else:
                splits = [translated_content]
                
            # Ensure we have the right number of translations
            if len(splits) != len(text_batch):
                log_error(f"Mismatch in translation batch: expected {len(text_batch)}, got {len(splits)}")
                # Pad with empty strings or the original text to match
                if len(splits) < len(text_batch):
                    splits.extend(["" for _ in range(len(text_batch) - len(splits))])
                else:
                    splits = splits[:len(text_batch)]
            
            return [s.strip() for s in splits]
            
        except requests.exceptions.RequestException as e:
            log_error(f"API error on attempt {attempt+1}: {str(e)}")
            if attempt < retry_count:
                time.sleep(2 * (attempt + 1))  # Exponential backoff
            else:
                log_error("Translation failed after all retries")
                return text_batch  # Return original as fallback
                
        except Exception as e:
            log_error(f"Unexpected error on attempt {attempt+1}: {str(e)}")
            if attempt < retry_count:
                time.sleep(2 * (attempt + 1))
            else:
                return text_batch
    
    return text_batch  # Default fallback

def process_translations_with_threading(texts, target_lang, entities):
    """Process translations with threading for larger batches"""
    if not texts:
        return []
        
    # Process in batches for better performance
    batch_size = min(MAX_BATCH_SIZE, len(texts))
    batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
    log_info(f"Processing {len(texts)} texts in {len(batches)} batches for {target_lang}")
    
    all_results = []
    placeholder_maps = []
    
    # Prepare text and create placeholders
    for i, text in enumerate(texts):
        modified_text, placeholder_map = replace_with_placeholders(text, entities)
        texts[i] = modified_text
        placeholder_maps.append(placeholder_map)
    
    # Translate in parallel batches
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
        batch_results = list(executor.map(
            lambda batch: translate_text_batch(batch, target_lang),
            batches
        ))
    
    # Flatten results
    for batch in batch_results:
        all_results.extend(batch)
    
    # Restore placeholders and convert numbers
    for i, (translated, placeholder_map) in enumerate(zip(all_results, placeholder_maps)):
        restored = restore_placeholders(translated, placeholder_map)
        all_results[i] = convert_numbers_to_script(restored, target_lang)
    
    return all_results

# PDF processing functions
def extract_pdf_components(pdf_path):
    """Extract text components from PDF with position data"""
    doc = fitz.open(pdf_path)
    components = []
    total_pages = len(doc)
    
    log_info(f"Extracting content from PDF: {total_pages} pages")
    
    for page_num, page in enumerate(doc):
        page_dict = page.get_text("dict")
        blocks = page_dict.get("blocks", [])
        
        text_blocks = []
        for b in blocks:
            if b["type"] == 0:  # Text block
                lines = []
                for line in b.get("lines", []):
                    spans = line.get("spans", [])
                    if spans:
                        # Combine spans in the same line
                        line_text = " ".join([span.get("text", "").strip() for span in spans])
                        if line_text.strip():
                            # Store text with font and position data
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
    """Translate all text components in the PDF"""
    # Extract all texts for batch translation
    all_texts = []
    text_map = []  # To track where each text belongs
    
    for page_idx, page in enumerate(components):
        for block_idx, block in enumerate(page["blocks"]):
            for line_idx, line in enumerate(block["lines"]):
                all_texts.append(line["text"])
                text_map.append((page_idx, block_idx, line_idx))
    
    # Translate all texts in optimized batches
    log_info(f"Translating {len(all_texts)} text elements to {target_lang}")
    translated_texts = process_translations_with_threading(all_texts, target_lang, entities)
    
    # Map translated texts back to their original positions
    for (page_idx, block_idx, line_idx), translated_text in zip(text_map, translated_texts):
        components[page_idx]["blocks"][block_idx]["lines"][line_idx]["translated"] = translated_text
        
        # Update progress if job tracking is enabled
        if job_id and job_id in translation_jobs:
            job = translation_jobs[job_id]
            current_page = page_idx + 1
            job.update_progress(target_lang, current_page)
    
    return components

def rebuild_translated_pdf(components, original_pdf_path, output_path, target_lang):
    """Rebuild PDF with translated text"""
    input_doc = fitz.open(original_pdf_path)
    output_doc = fitz.open()
    
    log_info(f"Rebuilding PDF with {target_lang} translations: {output_path}")
    
    # Get language-specific font
    lang_config = LANGUAGES.get(target_lang, {})
    lang_font = lang_config.get("font", "")
    lang_iso = lang_config.get("iso", "")
    
    # Process each page
    for page_data in components:
        page_num = page_data["page_num"]
        input_page = input_doc[page_num]
        
        # Create new page with same dimensions & rotation
        new_page = output_doc.new_page(
            width=page_data["size"][0],
            height=page_data["size"][1]
        )
        
        # First copy original page content as background
        new_page.show_pdf_page(
            new_page.rect,
            input_doc,
            page_num
        )
        
        # Cover original text with white boxes
        for block in page_data["blocks"]:
            rect = fitz.Rect(block["bbox"])
            new_page.draw_rect(rect, color=(1, 1, 1), fill=(1, 1, 1))
        
        # Add translated text
        for block in page_data["blocks"]:
            for line in block["lines"]:
                if "translated" in line and line["translated"].strip():
                    # Position text properly
                    rect = fitz.Rect(line["bbox"])
                    font_size = line["font_size"]
                    
                    # Insert translated text with appropriate font
                    try:
                        text_options = {
                            "fontname": lang_font or "helv",
                            "fontsize": font_size,
                            "color": (0, 0, 0)
                        }
                        
                        # Add translated text
                        new_page.insert_text(
                            (rect.x0, rect.y0 + font_size * 0.8),  # Position text at top of rect
                            line["translated"],
                            **text_options
                        )
                    except Exception as e:
                        log_error(f"Error inserting text: {str(e)}")
                        # Fallback to HTML insertion if font issues
                        html = f'<span lang="{lang_iso}" style="font-size:{font_size}pt">{line["translated"]}</span>'
                        new_page.insert_htmlbox(rect, html)
    
    # Save the new document
    output_doc.save(output_path, garbage=4, deflate=True)
    
    input_doc.close()
    output_doc.close()
    log_info(f"Completed PDF rebuild for {target_lang}")
    
    return output_path

def translate_pdf(job_id, pdf_path, entities, languages):
    """Main function to coordinate the PDF translation process"""
    try:
        # Extract content from PDF
        components, total_pages = extract_pdf_components(pdf_path)
        
        # Initialize job tracking if needed
        if job_id in translation_jobs:
            translation_jobs[job_id].total_pages = total_pages
        
        output_files = {}
        
        # Process each target language
        for lang in languages:
            try:
                # Translate components
                translated_components = translate_pdf_components(
                    components.copy(), 
                    lang, 
                    entities,
                    job_id
                )
                
                # Rebuild PDF with translations
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
        
        # Update job status if tracking
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

# Flask Routes
@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/templates/index.html')
def serve_template():
    """Serve the template file"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Multilingual PDF Translator</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                line-height: 1.6;
            }
            h1 {
                color: #2c3e50;
                text-align: center;
            }
            .container {
                background-color: #f9f9f9;
                border-radius: 5px;
                padding: 20px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            .form-group {
                margin-bottom: 15px;
            }
            label {
                display: block;
                margin-bottom: 5px;
                font-weight: bold;
            }
            input[type="text"], input[type="file"] {
                width: 100%;
                padding: 8px;
                box-sizing: border-box;
                border: 1px solid #ddd;
                border-radius: 4px;
            }
            button {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 10px 15px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
            }
            button:hover {
                background-color: #2980b9;
            }
            .checkbox-group {
                display: flex;
                gap: 15px;
            }
            .checkbox-item {
                display: flex;
                align-items: center;
                gap: 5px;
            }
            #progress-container {
                display: none;
                margin-top: 20px;
                text-align: center;
            }
            #progress-bar {
                width: 100%;
                height: 20px;
                background-color: #e0e0e0;
                border-radius: 10px;
                overflow: hidden;
                margin-bottom: 10px;
            }
            #progress-fill {
                height: 100%;
                background-color: #27ae60;
                width: 0%;
                transition: width 0.5s;
            }
        </style>
    </head>
    <body>
        <h1>Multilingual PDF Translator</h1>
        <div class="container">
            <form id="translation-form" enctype="multipart/form-data">
                <div class="form-group">
                    <label for="pdf_file">Upload PDF:</label>
                    <input type="file" id="pdf_file" name="pdf_file" accept=".pdf" required>
                </div>
                
                <div class="form-group">
                    <label>Select Target Languages:</label>
                    <div class="checkbox-group">
                        <div class="checkbox-item">
                            <input type="checkbox" id="lang_hindi" name="languages" value="Hindi" checked>
                            <label for="lang_hindi">Hindi</label>
                        </div>
                        <div class="checkbox-item">
                            <input type="checkbox" id="lang_tamil" name="languages" value="Tamil">
                            <label for="lang_tamil">Tamil</label>
                        </div>
                        <div class="checkbox-item">
                            <input type="checkbox" id="lang_telugu" name="languages" value="Telugu">
                            <label for="lang_telugu">Telugu</label>
                        </div>
                    </div>
                </div>
                
                <div class="form-group">
                    <label for="entities">Words to Keep in Original Language (comma-separated):</label>
                    <input type="text" id="entities" name="entities" placeholder="e.g., Chart, Histogram, Dashboard">
                </div>
                
                <button type="submit" id="submit-btn">Translate PDF</button>
            </form>
            
            <div id="progress-container">
                <div id="progress-bar">
                    <div id="progress-fill"></div>
                </div>
                <div id="progress-text">Processing... 0%</div>
            </div>
        </div>
        
        <script>
            document.getElementById('translation-form').addEventListener('submit', function(e) {
                e.preventDefault();
                
                // Get form data
                const formData = new FormData(this);
                
                // Get selected languages
                const selectedLanguages = [];
                document.querySelectorAll('input[name="languages"]:checked').forEach(checkbox => {
                    selectedLanguages.push(checkbox.value);
                });
                
                if (selectedLanguages.length === 0) {
                    alert('Please select at least one target language.');
                    return;
                }
                
                // Add languages to form data
                formData.set('languages', selectedLanguages.join(','));
                
                // Show progress container
                document.getElementById('progress-container').style.display = 'block';
                document.getElementById('submit-btn').disabled = true;
                
                // Start translation job
                fetch('/start-translation', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    if (data.job_id) {
                        // Poll for progress
                        pollProgress(data.job_id);
                    } else {
                        alert('Error: ' + (data.error || 'Unknown error'));
                        document.getElementById('progress-container').style.display = 'none';
                        document.getElementById('submit-btn').disabled = false;
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    alert('Error starting translation job.');
                    document.getElementById('progress-container').style.display = 'none';
                    document.getElementById('submit-btn').disabled = false;
                });
            });
            
            function pollProgress(jobId) {
                const interval = setInterval(() => {
                    fetch(`/job-status/${jobId}`)
                    .then(response => response.json())
                    .then(data => {
                        // Update progress
                        const progressFill = document.getElementById('progress-fill');
                        const progressText = document.getElementById('progress-text');
                        
                        progressFill.style.width = data.progress + '%';
                        progressText.textContent = `Processing... ${data.progress}%`;
                        
                        // Check if job is complete
                        if (data.status === 'completed') {
                            clearInterval(interval);
                            progressText.textContent = 'Translation completed!';
                            
                            // Download files
                            setTimeout(() => {
                                for (const lang in data.output_files) {
                                    window.location.href = `/download-pdf/${jobId}/${lang}`;
                                }
                                document.getElementById('submit-btn').disabled = false;
                            }, 1000);
                        } 
                        else if (data.status === 'failed') {
                            clearInterval(interval);
                            progressText.textContent = 'Translation failed: ' + (data.error || 'Unknown error');
                            document.getElementById('submit-btn').disabled = false;
                        }
                    })
                    .catch(error => {
                        console.error('Error polling job status:', error);
                    });
                }, 2000);
            }
        </script>
    </body>
    </html>
    """

@app.route('/start-translation', methods=['POST'])
def start_translation():
    """Start a new translation job"""
    if 'pdf_file' not in request.files:
        return jsonify({"error": "No PDF file uploaded"}), 400
    
    pdf_file = request.files['pdf_file']
    if not pdf_file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "Uploaded file is not a PDF"}), 400
    
    # Create unique job ID
    job_id = str(uuid.uuid4())
    
    # Parse form data
    entities_input = request.form.get('entities', '')
    languages_input = request.form.get('languages', 'Hindi')
    
    entities = [e.strip() for e in entities_input.split(',') if e.strip()]
    languages = parse_user_languages(languages_input)
    
    if not languages:
        return jsonify({"error": "No valid languages selected"}), 400
    
    # Create temporary directory for this job
    temp_dir = tempfile.mkdtemp()
    
    # Save the uploaded PDF
    pdf_path = os.path.join(temp_dir, "input.pdf")
    pdf_file.save(pdf_path)
    
    # Initialize job tracking
    job = TranslationJob(job_id, languages, 0)  # Pages will be updated during extraction
    translation_jobs[job_id] = job
    
    # Start processing in a separate thread
    threading.Thread(
        target=translate_pdf,
        args=(job_id, pdf_path, entities, languages)
    ).start()
    
    return jsonify({"job_id": job_id, "message": "Translation started"}), 202

def parse_user_languages(user_input):
    """Parse and validate language selection from user input"""
    if isinstance(user_input, list):
        selected = [lang.strip().capitalize() for lang in user_input]
    else:
        selected = [lang.strip().capitalize() for lang in user_input.split(',')]
    
    valid = [lang for lang in selected if lang in LANGUAGES]
    
    if not valid:
        log_info("No valid languages selected. Using Hindi as default.")
        return ["Hindi"]
    
    return valid

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