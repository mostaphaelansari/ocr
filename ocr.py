import streamlit as st
import pytesseract
from PIL import Image
import numpy as np
import json
from transformers import T5Tokenizer, T5ForConditionalGeneration

def load_image(uploaded_file):
    # Load image using PIL
    image = Image.open(uploaded_file)
    # Convert image to RGB if it has an alpha channel
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    return image

def ocr_image(image):
    # Perform OCR using pytesseract
    ocr_result = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    
    # Extract text, coordinates, and confidence
    extracted_text = []
    for i in range(len(ocr_result['text'])):
        if int(ocr_result['conf'][i]) > 0:  # Filter out empty and low-confidence results
            text_data = {
                "text": ocr_result['text'][i],
                "coordinates": {
                    "x1": ocr_result['left'][i],
                    "y1": ocr_result['top'][i],
                    "x2": ocr_result['left'][i] + ocr_result['width'][i],
                    "y2": ocr_result['top'][i] + ocr_result['height'][i]
                },
                "confidence": ocr_result['conf'][i]
            }
            extracted_text.append(text_data)
    return extracted_text

def format_table_rag(extracted_text):
    # Load the T5 model and tokenizer
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Sort extracted text by y1 coordinate to group rows
    extracted_text.sort(key=lambda x: x['coordinates']['y1'])

    text_blocks = [item['text'] for item in extracted_text]
    concatenated_text = " ".join(text_blocks)

    # Prepare input for T5
    input_text = f"Format this OCR result into a structured table: {concatenated_text}"
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

    # Generate structured text
    outputs = model.generate(inputs, max_length=512, num_beams=4, early_stopping=True)
    structured_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return structured_text

def main():
    st.title("OCR Extraction with Tesseract and RAG")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        image = load_image(uploaded_file)
        
        # Display the uploaded image
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Extracting text...")
        
        # Perform OCR
        extracted_text = ocr_image(image)
        
        # Format the extracted text into a table using RAG
        table = format_table_rag(extracted_text)
        
        # Display results
        st.write("Extracted Table:")
        st.text(table)
        
        # Save results to JSON file
        output_path = 'ocr_table_results.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({"structured_text": table}, f, ensure_ascii=False, indent=4)
        
        st.write(f"OCR table results saved to {output_path}")

if __name__ == "__main__":
    main()
