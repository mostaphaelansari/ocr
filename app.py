import streamlit as st
import pytesseract
from PIL import Image
import json
import cv2
import numpy as np
import os

# Define the path to the Tesseract executable based on the operating system
if os.name == 'nt':
    # Windows
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
elif os.name == 'posix':
    # macOS or Linux
    pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'  # macOS
    # pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Linux

def preprocess_image(image):
    # Convert PIL image to numpy array
    img = np.array(image)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Convert back to PIL image
    processed_image = Image.fromarray(thresh)
    
    return processed_image

def ocr_image(image):
    # Perform OCR on the image
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    
    # Extract text, coordinates, and confidence
    extracted_text = []
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 0:  # Filter out empty results
            text_data = {
                "text": data['text'][i],
                "coordinates": {
                    "x1": data['left'][i],
                    "y1": data['top'][i],
                    "x2": data['left'][i] + data['width'][i],
                    "y2": data['top'][i] + data['height'][i]
                },
                "confidence": data['conf'][i]
            }
            extracted_text.append(text_data)
    
    return extracted_text

def main():
    st.title("OCR Extraction with Tesseract")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Display the uploaded image
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Pre-processing the image...")
        
        # Pre-process the image
        processed_image = preprocess_image(image)
        
        # Display the processed image
        st.image(processed_image, caption='Processed Image.', use_column_width=True)
        st.write("Extracting text...")
        
        # Perform OCR
        results = ocr_image(processed_image)
        
        # Display results
        st.write("OCR Results:")
        st.json(results)
        
        # Save results to JSON file
        output_path = 'ocr_results.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        
        st.write(f"OCR results saved to {output_path}")

if __name__ == "__main__":
    main()