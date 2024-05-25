import streamlit as st
import keras_ocr
from PIL import Image
import numpy as np
import json

def load_image(uploaded_file):
    # Load image using PIL
    image = Image.open(uploaded_file)
    # Convert image to RGB if it has an alpha channel
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    return image

def ocr_image(image):
    # Create a Keras OCR pipeline
    pipeline = keras_ocr.pipeline.Pipeline()

    # Convert PIL image to numpy array
    img_array = np.array(image)

    # Perform OCR using Keras OCR
    predictions = pipeline.recognize([img_array])[0]

    # Extract text, coordinates, and confidence
    extracted_text = []
    for word, box in predictions:
        text_data = {
            "text": word,
            "coordinates": {
                "x1": int(box[0][0]),
                "y1": int(box[0][1]),
                "x2": int(box[2][0]),
                "y2": int(box[2][1])
            },
            "confidence": None  # Keras OCR does not provide confidence scores directly
        }
        extracted_text.append(text_data)

    return extracted_text

def format_table(extracted_text):
    # Sort extracted text by y1 coordinate to group rows
    extracted_text.sort(key=lambda x: x['coordinates']['y1'])

    # Initialize table structure
    table = {
        "header": [],
        "rows": []
    }
    current_row_y = -1
    current_row = []

    for item in extracted_text:
        y1 = item['coordinates']['y1']
        if current_row_y == -1 or abs(current_row_y - y1) < 10:  # If within the same row
            current_row.append(item['text'])
            current_row_y = y1
        else:
            if table['header'] == []:
                table['header'] = current_row
            else:
                table['rows'].append(current_row)
            current_row = [item['text']]
            current_row_y = y1

    if current_row:
        if table['header'] == []:
            table['header'] = current_row
        else:
            table['rows'].append(current_row)

    return table

def main():
    st.title("OCR Extraction with Keras OCR")
    
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
        
        # Format the extracted text into a table
        table = format_table(extracted_text)
        
        # Display results
        st.write("Extracted Table:")
        st.json(table)
        
        # Save results to JSON file
        output_path = 'ocr_table_results.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(table, f, ensure_ascii=False, indent=4)
        
        st.write(f"OCR table results saved to {output_path}")

if __name__ == "__main__":
    main()
