from models import resnet34
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
import numpy as np
from PIL import Image
import io
from torchvision import transforms
from PIL import Image, ImageOps, ImageEnhance
import difflib
import gensim
import tempfile
import pandas as pd
import os
import ast
import numpy as np
import csv
from collections import defaultdict
import base64
GOOGLE_API_KEY = "AIzaSyARnrGZ24tragAGA_vBdrWAuzDmS_uiSP4"
app = FastAPI()


spelling_model = gensim.models.KeyedVectors.load_word2vec_format('spellingCorrection_model/bnword2vec.txt')

def correct_word(word):
    if word in spelling_model:
        similar_words = spelling_model.similar_by_word(word)
        print(f"Words similar to '{word}':{word}")
        return word
    else:
        vocab = spelling_model.index_to_key
        closest_word = difflib.get_close_matches(word, vocab, n=1)
        if closest_word:
            closest_word = closest_word[0]
            print(f"The closest word is '{closest_word}'")
            return closest_word
        else:
            print(f"No similar word found in the model's vocabulary or through difflib.")
            return word

def detect(patches):
    predicted = []
    for patch in patches:
        img = Image.fromarray(patch)
        enhancer = ImageEnhance.Contrast(img)
        img_enhanced = enhancer.enhance(2.0)
        img_inv = ImageOps.invert(img_enhanced)
        img_inv = img_inv.resize((224,128)).convert('RGB')

        transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)) 
                ])

        img_ten = transform(img_inv)

        input_image = img_ten.unsqueeze(0) 

        input_image = input_image.to(device)

        model.eval()

        # Forward pass to get model predictions
        with torch.no_grad():
            outputs = model(input_image)

        # Find the index of the maximum probability for each class
        predicted_grapheme_idx = torch.argmax(outputs[0]).item()
        predicted_vowel_idx = torch.argmax(outputs[1]).item()
        predicted_consonant_idx = torch.argmax(outputs[2]).item()

        predicted_grapheme_root = torch.tensor(predicted_grapheme_idx)
        predicted_vowel_diacritic = torch.tensor(predicted_vowel_idx)
        predicted_consonant_diacritic = torch.tensor(predicted_consonant_idx)

        pred_char = []

        if(predicted_vowel_idx != 0 and predicted_consonant_idx != 0):
                if(predicted_consonant_idx == 2):
                    pred_char.append( consonant_diacritic_components[predicted_consonant_idx] +  grapheme_root_components[predicted_grapheme_idx] + vowel_diacritic_components[predicted_vowel_idx] )
                elif(predicted_consonant_idx == 1):
                    pred_char.append(grapheme_root_components[predicted_grapheme_idx] + vowel_diacritic_components[predicted_vowel_idx] + consonant_diacritic_components[predicted_consonant_idx])
                else:
                    pred_char.append(grapheme_root_components[predicted_grapheme_idx] + consonant_diacritic_components[predicted_consonant_idx] + vowel_diacritic_components[predicted_vowel_idx])
        
        elif(predicted_vowel_idx == 0 and predicted_consonant_idx == 0):
            pred_char.append(grapheme_root_components[predicted_grapheme_idx])

        else:
            if(predicted_vowel_idx == 0):
                if(predicted_consonant_idx == 2):
                    pred_char.append( consonant_diacritic_components[predicted_consonant_idx] +  grapheme_root_components[predicted_grapheme_idx] )
                else:
                    pred_char.append(grapheme_root_components[predicted_grapheme_idx] + consonant_diacritic_components[predicted_consonant_idx])
            else:
                pred_char.append(grapheme_root_components[predicted_grapheme_idx] + vowel_diacritic_components[predicted_vowel_idx])
        
        predicted.append(pred_char)  
              
    predicted = ''.join([char for sublist in predicted for char in sublist])    
    print(predicted)
    return correct_word(predicted)
   


# Load the model at startup
@app.on_event("startup")
async def startup_event():
    global model, device, grapheme_root_components, vowel_diacritic_components, consonant_diacritic_components
    


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = resnet34.resnet34().to(device)

    # Load the trained weights
    model.load_state_dict(torch.load('recongnition_model/char_level_trained_model_128x224_shoroborno_again.pth'))
    grapheme_root_components = ['০','১','২','৩','৪','৫','৬','৭','৮','৯','ং', 'ঃ', 'অ', 'আ', 'ই', 'ঈ', 'উ', 'ঊ', 'ঋ', 'এ', 'ঐ', 'ও', 'ঔ', 'ক', 'ক্ক', 'ক্ট', 'ক্ত', 'ক্ল', 'ক্ষ', 'ক্ষ্ণ', 'ক্ষ্ম', 'ক্স', 'খ', 'গ', 'গ্ধ', 'গ্ন', 'গ্ব', 'গ্ম', 'গ্ল', 'ঘ', 'ঘ্ন', 'ঙ', 'ঙ্ক', 'ঙ্ক্ত', 'ঙ্ক্ষ', 'ঙ্খ', 'ঙ্গ', 'ঙ্ঘ', 'চ', 'চ্চ', 'চ্ছ', 'চ্ছ্ব', 'ছ', 'জ', 'জ্জ', 'জ্জ্ব', 'জ্ঞ', 'জ্ব', 'ঝ', 'ঞ', 'ঞ্চ', 'ঞ্ছ', 'ঞ্জ', 'ট', 'ট্ট', 'ঠ', 'ড', 'ড্ড', 'ঢ', 'ণ', 'ণ্ট', 'ণ্ঠ', 'ণ্ড', 'ণ্ণ', 'ত', 'ত্ত', 'ত্ত্ব', 'ত্থ', 'ত্ন', 'ত্ব', 'ত্ম', 'থ', 'দ', 'দ্ঘ', 'দ্দ', 'দ্ধ', 'দ্ব', 'দ্ভ', 'দ্ম', 'ধ', 'ধ্ব', 'ন', 'ন্জ', 'ন্ট', 'ন্ঠ', 'ন্ড', 'ন্ত', 'ন্ত্ব', 'ন্থ', 'ন্দ', 'ন্দ্ব', 'ন্ধ', 'ন্ন', 'ন্ব', 'ন্ম', 'ন্স', 'প', 'প্ট', 'প্ত', 'প্ন', 'প্প', 'প্ল', 'প্স', 'ফ', 'ফ্ট', 'ফ্ফ', 'ফ্ল', 'ব', 'ব্জ', 'ব্দ', 'ব্ধ', 'ব্ব', 'ব্ল', 'ভ', 'ভ্ল', 'ম', 'ম্ন', 'ম্প', 'ম্ব', 'ম্ভ', 'ম্ম', 'ম্ল', 'য', 'র', 'ল', 'ল্ক', 'ল্গ', 'ল্ট', 'ল্ড', 'ল্প', 'ল্ব', 'ল্ম', 'ল্ল', 'শ', 'শ্চ', 'শ্ন', 'শ্ব', 'শ্ম', 'শ্ল', 'ষ', 'ষ্ক', 'ষ্ট', 'ষ্ঠ', 'ষ্ণ', 'ষ্প', 'ষ্ফ', 'ষ্ম', 'স', 'স্ক', 'স্ট', 'স্ত', 'স্থ', 'স্ন', 'স্প', 'স্ফ', 'স্ব', 'স্ম', 'স্ল', 'স্স', 'হ', 'হ্ন', 'হ্ব', 'হ্ম', 'হ্ল', 'ৎ', 'ড়', 'ঢ়', 'য়']
    vowel_diacritic_components = ['0', 'া', 'ি', 'ী', 'ু', 'ূ', 'ৃ', 'ে', 'ৈ', 'ো', 'ৌ']
    consonant_diacritic_components = ['0', 'ঁ', 'র্', 'র্য', '্য', '্র', '্র্য', 'র্্র']

@app.post("/predict/")
async def predict_text(file: UploadFile = File(...)):
    try:
        # Read the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert to numpy array
        image_np = np.array(image)
        
        width, height = image.size
        original_filename = file.filename

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp:
            temp_path = temp.name
            image.save(f"handw_s_demo_test/{original_filename}")
            # image.save(original_filename)
            run_yolo_detection()
            process_yolo_results()
        patches = image_patch(f"handw_s_demo_test/{original_filename}", width, height, 'handw_s_demo_sorted_annotations.csv')
        # patches = image_patch(original_filename, width, height, 'handw_s_demo_sorted_annotations.csv')
        # os.remove(f"handw_s_demo_test/{original_filename}")  # Clean up
        # os.remove(original_filename)  # Clean up
        
        if not patches:
            return JSONResponse(content={"error": "No text regions detected in the image"}, status_code=400)
        

        predicted_text = detect(patches)
        predicted_text = correct_word(predicted_text)
        
        return {"predicted_text": predicted_text}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


def calculate_coordinates(x_center, y_center, width, height):
    half_width = width / 2
    half_height = height / 2

    x1 = x_center - half_width
    y1 = y_center - half_height
    x3 = x_center + half_width
    y3 = y_center + half_height

    return x1, y1, x3, y3

def image_patch(path, wid=640, hei=640, csv_path='handw_s_demo_sorted_annotations.csv', image=None):
    filename = os.path.split(path)[-1]

    if(filename.split('.')[0].isdigit()):
        image_id_to_find = ''.join(c for c in filename if c.isdigit())
    else:
        image_id_to_find = filename.split('_')[0]

    df = pd.read_csv(csv_path)
    # print("image_id_to_find =====", image_id_to_find)
    matching_rows = df[df['image_id'] == int(image_id_to_find)]

    patches = []

    # Check if there are any matching rows
    if not matching_rows.empty:
        # Extract the first matching row
        row = matching_rows.iloc[0]

        x_c = ast.literal_eval(row['x_center'])
        y_c = ast.literal_eval(row['y_center'])
        w = ast.literal_eval(row['width'])
        h = ast.literal_eval(row['height'])

        x_c = [x * wid for x in x_c]
        y_c = [y * hei for y in y_c]
        w = [width * wid for width in w]
        h = [height * hei for height in h]

        if image is None:
            from PIL import Image
            image = Image.open(path)
        
        img_np = np.array(image)

        for i in range(len(x_c)):
            xc, yc, width, height = x_c[i], y_c[i], w[i], h[i]

            x1, y1, x3, y3 = calculate_coordinates(xc, yc, width, height)

            x1, y1, x3, y3 = int(x1), int(y1), int(x3), int(y3)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x3 = min(img_np.shape[1], x3)
            y3 = min(img_np.shape[0], y3)

            patch = img_np[y1:y3, x1:x3]
            patches.append(patch)
    else:
        print(f"No matching rows found for image_id {image_id_to_find}")
        print(df)
        
    return patches

@app.get("/runyolo")
def run_yolo_detection(source_folder='handw_s_demo_test', model_path='detection_model/weights2/last_medium.pt', conf=0.5, output_dir='runs/detect', name='predict14'):
    """
    Run YOLO detection on images in the source folder
    
    Args:
        source_folder: Path to folder containing images to process
        model_path: Path to YOLO model weights
        conf: Confidence threshold for detections
        output_dir: Custom directory to save detection results (optional)
    """
    command = f"yolo task=detect mode=predict model={model_path} conf={conf} source='{source_folder}' save_txt=True name='{name}' exist_ok=True"
    
    # Add output directory if specified
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        command += f" project='{output_dir}'"
    
    # Run YOLO command
    print(f"Running YOLO detection: {command}")
    os.system(command)
    
    # Return the expected output directory
    if output_dir:
        return os.path.join(output_dir, "predict")
    else:
        # Default YOLO output location
        return "runs/detect/predict"
    

@app.get('/makeAnnotation')
def process_yolo_results(labels_directory='runs/detect/predict14/labels', 
                         annotations_csv='handw_s_demo_annotations.csv', 
                         sorted_csv='handw_s_demo_sorted_annotations.csv'):
    """
    Process YOLO detection results from a labels directory, create annotations CSV and sorted annotations CSV
    
    Args:
        labels_directory: Path to the directory containing YOLO detection label files (.txt)
        annotations_csv: Path for the output annotations CSV file
        sorted_csv: Path for the output sorted annotations CSV file
        
    Returns:
        Tuple of (annotations_csv_path, sorted_csv_path)
    """
    # Step 1: Create annotations CSV from YOLO detection results
    data_dict = defaultdict(lambda: {'x_center': [], 'y_center': [], 'width': [], 'height': []})
    
    # Iterate over txt files in the directory
    for filename in os.listdir(labels_directory):
        if filename.endswith(".txt"):
            # Extract image_id from the filename
            image_id = filename.split('.txt')[0]
            
            # Read the content of the txt file
            with open(os.path.join(labels_directory, filename), 'r') as file:
                lines = file.readlines()
            
            # Parse each line and aggregate information
            for line in lines:
                parts = line.strip().split(' ')
                _, x_center, y_center, width, height = map(float, parts)
                
                # Append information to lists in the defaultdict
                data_dict[image_id]['x_center'].append(x_center)
                data_dict[image_id]['y_center'].append(y_center)
                data_dict[image_id]['width'].append(width)
                data_dict[image_id]['height'].append(height)
    
    # Write data to CSV file
    with open(annotations_csv, mode='w', newline='') as csv_file:
        fieldnames = ['image_id', 'x_center', 'y_center', 'width', 'height']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        # Write header
        writer.writeheader()
        
        # Write data rows
        for image_id, data in data_dict.items():
            writer.writerow({
                'image_id': image_id,
                'x_center': data['x_center'],
                'y_center': data['y_center'],
                'width': data['width'],
                'height': data['height']
            })
    
    print(f"CSV file '{annotations_csv}' created successfully.")
    
    # Step 2: Create sorted annotations CSV from the annotations CSV
    rows = []
    
    # Open the input CSV file
    with open(annotations_csv, 'r') as input_csv_file:
        # Create a CSV reader object with header
        csv_reader = csv.DictReader(input_csv_file)
        
        # Extract and sort data
        for row in csv_reader:
            image_id = row['image_id']
            x_c = ast.literal_eval(row['x_center'])
            y_c = ast.literal_eval(row['y_center'])
            w = ast.literal_eval(row['width'])
            h = ast.literal_eval(row['height'])
            
            # Sort the data based on x_center
            sorted_bundle = sorted(zip(x_c, y_c, w, h), key=lambda bundle: bundle[0])
            
            # Only proceed if there are elements to unpack
            if sorted_bundle:  # <-- This check is important
                sorted_x, sorted_y, sorted_w, sorted_h = zip(*sorted_bundle)
                
                # Create a new row with sorted values
                new_row = {
                    'image_id': image_id,
                    'x_center': str(sorted_x),
                    'y_center': str(sorted_y),
                    'width': str(sorted_w),
                    'height': str(sorted_h)
                }
                
                # Append the new row to the list
                rows.append(new_row)
    
    # Write the sorted data to a new CSV file
    fieldnames = ['image_id', 'x_center', 'y_center', 'width', 'height']
    import shutil
    with open(sorted_csv, 'w', newline='') as output_csv_file:
        csv_writer = csv.DictWriter(output_csv_file, fieldnames=fieldnames)
        
        # Write the header
        csv_writer.writeheader()
        
        # Write the sorted rows
        csv_writer.writerows(rows)
    
    print(f"Sorted data saved to {sorted_csv}")
    shutil.rmtree(f"runs/detect/predict14/labels")
    
    return annotations_csv, sorted_csv

import aiohttp

from PIL import Image, ImageEnhance

def process_image(input_image):
    """
    Converts the input image to grayscale and increases its contrast.

    Args:
        input_image (PIL.Image.Image): The input image.

    Returns:
        PIL.Image.Image: The processed image (grayscale and contrast-enhanced).
    """
    # Convert image to grayscale
    grayscale_image = input_image.convert("L")

    # Increase contrast
    enhancer = ImageEnhance.Contrast(grayscale_image)
    enhanced_image = enhancer.enhance(2.0)  # Increase contrast by a factor (e.g., 2.0)

    return enhanced_image

@app.post("/trascript")
async def getTranscript(file: UploadFile = File(...)):
    try:
        # Read the uploaded file
        contents = await file.read()
        
        # Convert to base64
        img_base64 = base64.b64encode(contents).decode('utf-8')
        
        

        # Gemini API endpoint
        api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
        
        # Prepare payload
        payload = {
            "contents": [{
                "parts": [
                    {"text": "Perform OCR on this image and return just the extracted Bangla text. NO line breaks or \\ns "},
                    {"inline_data": {"mime_type": file.content_type or "image/jpeg", "data": img_base64}}
                ]
            }],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 2048
            }
        }
        
        # Call Gemini API with proper error handling
        async with aiohttp.ClientSession() as session:
            async with session.post(
                api_url,
                headers={"Content-Type": "application/json", "x-goog-api-key": GOOGLE_API_KEY},
                json=payload
            ) as response:
                response_text = await response.text()
                
                if response.status != 200:
                    print(f"API Error: {response.status} - {response_text}")
                    return {"error": f"Gemini API error: {response.status}", "details": response_text}
                
                result = await response.json()
                
                try:
                    text = result['candidates'][0]['content']['parts'][0]['text']
                    return {"text": text.strip()}
                except (KeyError, IndexError) as e:
                    print(f"Response parsing error: {e}")
                    return {"error": "Could not parse API response", "response": result}
                
    except Exception as e:
        print(f"Exception: {str(e)}")
        return {"error": str(e)}
    


# @app.post("/trascript")
# async def getTranscript(file: UploadFile = File(...)):
#     try:
#         # Read the uploaded file
#         contents = await file.read()
        
#         # Convert to PIL Image for processing
#         input_image = Image.open(io.BytesIO(contents))
        
#         # Process image (enhance contrast and convert to grayscale)
#         processed_image = process_image(input_image)
        
#         # Convert processed image to bytes
#         buffered = io.BytesIO()
#         processed_image.save(buffered, format="PNG")
        
#         # Convert to base64
#         img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
#         # Gemini API endpoint
#         api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
        
#         # Prepare payload
#         payload = {
#             "contents": [{
#                 "parts": [
#                     {"text": "Perform OCR on this image and return just the extracted Bangla text. NO line breaks or \\ns "},
#                     {"inline_data": {"mime_type": "image/png", "data": img_base64}}
#                 ]
#             }],
#             "generationConfig": {
#                 "temperature": 0.1,
#                 "maxOutputTokens": 2048
#             }
#         }
        
#         # Call Gemini API with proper error handling
#         async with aiohttp.ClientSession() as session:
#             async with session.post(
#                 api_url,
#                 headers={"Content-Type": "application/json", "x-goog-api-key": GOOGLE_API_KEY},
#                 json=payload
#             ) as response:
#                 response_text = await response.text()
                
#                 if response.status != 200:
#                     print(f"API Error: {response.status} - {response_text}")
#                     return {"error": f"Gemini API error: {response.status}", "details": response_text}
                
#                 result = await response.json()
                
#                 try:
#                     text = result['candidates'][0]['content']['parts'][0]['text']
#                     return {"text": text.strip()}
#                 except (KeyError, IndexError) as e:
#                     print(f"Response parsing error: {e}")
#                     return {"error": "Could not parse API response", "response": result}
                
#     except Exception as e:
#         print(f"Exception: {str(e)}")
#         return {"error": str(e)}

# def process_image(input_image):
#     """
#     Converts the input image to grayscale and increases its contrast.

#     Args:
#         input_image (PIL.Image.Image): The input image.

#     Returns:
#         PIL.Image.Image: The processed image (grayscale and contrast-enhanced).
#     """
#     # Convert image to grayscale
#     grayscale_image = input_image.convert("L")

#     # Increase contrast
#     enhancer = ImageEnhance.Contrast(grayscale_image)
#     enhanced_image = enhancer.enhance(2.0)  # Increase contrast by a factor (e.g., 2.0)

#     return enhanced_image