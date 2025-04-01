from flask import Flask, render_template, request, send_file, jsonify
import fitz  # PyMuPDF
import re
import os
import requests
from io import BytesIO

app = Flask(__name__)

# Groq API Configuration
GROQ_API_KEY = "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"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "llama3-70b-8192"  # Verified supported model

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

# Enhanced OCR cleaning
def clean_ocr_text(text):
    replacements = {
        # English OCR fixes
        "vijungtai": "visualization",
        "sradied": "stacked",
        "cudume": "column",
        "aelationshlp": "relationship",
        
        # Hindi preservation/corrections
        "चाट": "chart",
        "इटा": "data",
        "दृथ्य": "visual",
        "परिद्य": "परिदृश्य",
        "अंतदृष्टि": "अंतर्दृष्टि",
        "मुख्य अंतर्दूहि": "मुख्य अंतर्दृष्टि",
        "बार चार्ट": "बार चार्ट",  # Preserve correct Hindi
        "विज़ुअलाइज़ेशन": "विज़ुअलाइज़ेशन",
        
        # Remove garbage text
        "महु": "",
        "草要 3दिदृष्टि": "मुख्य अंतर्दृष्टि"
    }
    for wrong, correct in replacements.items():
        text = text.replace(wrong, correct)
    text = re.sub(r'(\b\w+\b)(\s+\1)+', r'\1', text)  # Remove repetitions
    return re.sub(r'\s+', ' ', text).strip()

# Enhanced entity preservation
def replace_with_placeholders(text, entities):
    placeholder_map = {}
    modified_text = text
    
    # Preserve technical patterns
    patterns = [
        (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'), "email"),
        (re.compile(r'https?://\S+|www\.\S+'), "url"),
        (re.compile(r'@\w+'), "username"),
        (re.compile(r'[\$£€%#@&*]\s*\d+(?:\.\d+)?|\d+(?:\.\d+)?[\$£€%#@&*]'), "symbol_num"),
    ]
    
    for pattern, _ in patterns:
        for match in pattern.findall(modified_text):
            placeholder = f"__PRESERVE{len(placeholder_map):03d}__"
            placeholder_map[placeholder] = match
            modified_text = modified_text.replace(match, placeholder)

    # Preserve user entities
    for entity in entities:
        modified_text = re.sub(
            re.escape(entity), 
            lambda m: f"__PRESERVE{len(placeholder_map):03d}__",
            modified_text,
            flags=re.IGNORECASE
        )
        placeholder_map[f"__PRESERVE{len(placeholder_map):03d}__"] = entity

    print(f"🔍 Placeholder mapping: {placeholder_map}")
    return modified_text, placeholder_map

# Strict translation function
def translate_batch(texts, target_lang):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    translated = []
    for text in texts:
        try:
            response = requests.post(
                GROQ_API_URL,
                headers=headers,
                json={
                    "model": MODEL_NAME,
                    "messages": [
                        {
                            "role": "system",
                            "content": f"""
                            YOU MUST TRANSLATE TO {target_lang} EXACTLY:
                            - Preserve __PRESERVExxx__ placeholders
                            - Keep technical terms (Chart, Histogram, etc)
                            - NEVER return English original text
                            - Output ONLY the translation
                            """
                        },
                        {"role": "user", "content": text}
                    ],
                    "temperature": 0.2,
                    "max_tokens": 1000
                }
            )
            result = response.json()
            translated_text = result['choices'][0]['message']['content'].strip()
            translated.append(translated_text)
            print(f"✅ Translated: {text[:50]}... -> {translated_text[:50]}...")
        except Exception as e:
            print(f"🚨 Translation failed: {str(e)}")
            translated.append(text)  # Fallback
    
    return translated

# Enhanced PDF processing
def extract_pdf_components(pdf_path):
    doc = fitz.open(pdf_path)
    components = []
    
    for page_num, page in enumerate(doc):
        print(f"\n📄 PAGE {page_num+1} CONTENT:")
        blocks = page.get_text("dict")["blocks"]
        page_blocks = []
        
        for block in blocks:
            if block["type"] == 0:  # Text block
                lines = []
                for line in block["lines"]:
                    line_text = " ".join(span["text"].strip() for span in line["spans"])
                    if line_text:
                        print(f"| {line_text}")  # Debug output
                        lines.append({
                            "text": line_text,
                            "bbox": line["bbox"],
                            "font_size": line["spans"][0]["size"]
                        })
                
                if lines:
                    page_blocks.append({
                        "bbox": block["bbox"],
                        "lines": lines
                    })
        
        components.append({
            "page_num": page_num,
            "blocks": page_blocks,
            "size": (page.rect.width, page.rect.height)
        })
    
    doc.close()
    return components

# Font-embedded PDF reconstruction
def rebuild_pdf(components, lang, output_path):
    doc = fitz.open()
    
    for page_data in components:
        page = doc.new_page(width=page_data["size"][0], height=page_data["size"][1])
        
        for block in page_data["blocks"]:
            for line in block["lines"]:
                if "translated" in line:
                    text = line["translated"]
                    font_size = line["font_size"]
                    
                    # Hindi/Tamil/Telugu compatible font
                    page.insert_text(
                        point=(line["bbox"][0], line["bbox"][1] + font_size),
                        text=text,
                        fontname="notos",
                        fontsize=font_size,
                        encoding=fitz.TEXT_ENCODING_UNICODE
                    )
    
    doc.save(output_path, garbage=4, deflate=True)
    doc.close()

# Main translation flow
@app.route('/')
def home():
    """Render the main upload page"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>PDF Translator</title>
    </head>
    <body>
        <h1>Upload PDF for Translation</h1>
        <form action="/translate" method="post" enctype="multipart/form-data">
            <input type="file" name="pdf_file" accept=".pdf" required>
            <br><br>
            <input type="submit" value="Translate to Hindi">
        </form>
    </body>
    </html>
    """

@app.route('/translate', methods=['POST'])
def handle_translation():
    if 'pdf_file' not in request.files:
        return jsonify({"error": "No PDF uploaded"}), 400
    
    pdf_file = request.files['pdf_file']
    pdf_path = "temp.pdf"
    pdf_file.save(pdf_path)
    
    # Process PDF
    components = extract_pdf_components(pdf_path)
    
    # Extract all text chunks
    text_chunks = []
    for page in components:
        for block in page["blocks"]:
            for line in block["lines"]:
                text_chunks.append(line["text"])
    
    # Translate with entity preservation
    entities = request.form.get('entities', '').split(',')
    translated_chunks = []
    for chunk in text_chunks:
        clean_text = clean_ocr_text(chunk)
        modified_text, placeholders = replace_with_placeholders(clean_text, entities)
        translated = translate_batch([modified_text], "Hindi")[0]
        
        # Restore placeholders
        for ph, original in placeholders.items():
            translated = translated.replace(ph, original)
        
        translated_chunks.append(translated)
    
    # Rebuild PDF with translations
    output_path = "translated.pdf"
    idx = 0
    for page in components:
        for block in page["blocks"]:
            for line in block["lines"]:
                if idx < len(translated_chunks):
                    line["translated"] = translated_chunks[idx]
                    idx += 1
    
    rebuild_pdf(components, "Hindi", output_path)
    
    return send_file(output_path, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
