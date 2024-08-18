import chromadb
import fitz
import pdfplumber
import os
import torch
from langchain.schema import Document
from langchain_text_splitters import CharacterTextSplitter
from chromadb import EmbeddingFunction, Embeddings
from chromadb import Documents
import open_clip
from PIL import Image
from openai import OpenAI
import pandas as pd
import json
import glob
from typing import List
import base64
import requests


def extract_text(pdf_paths: List[str], text_insert_collection):
    for pdf_path in pdf_paths:
        try:
            processing_pdf(pdf_path, text_insert_collection)
        except Exception as e:
            print(f"Error processing PDF '{pdf_path}': {e}")

class Embedding_Text(EmbeddingFunction):
    def __init__(self):
        try:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
            self.model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
            self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        except Exception as e:
            raise RuntimeError(f"Error in loading the model or tokenizer: {e}")

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for doc in input:
            try:
                if isinstance(doc, str):
                    with torch.no_grad():
                        text = self.tokenizer([doc])
                        text_embeddings = self.model.encode_text(text).detach().cpu().numpy()[0].tolist()
                        embeddings.append(text_embeddings)
            except Exception as e:
                print(f"Error in creating embeddings'{doc}': {e}")
        
        return embeddings

def processing_pdf(pdf_path: str, text_insert_collection):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                try:
                    # Extract text from the page
                    text = page.extract_text()
                    print(f"Page {page_num + 1} Text:\n{text}\n")
                    
                    # Create Document object
                    doc = Document(page_content=text, metadata={"source": pdf_path, "pageno": page_num + 1})
                    
                    # Split document into smaller chunks
                    texts = document_into_chunks(doc)
                    
                    # Add each chunk to the collection
                    adding_data_to_collection(pdf_path, texts, text_insert_collection)
                    
                except Exception as e:
                    print(f"Error processing page {page_num + 1} of PDF '{pdf_path}': {e}")
    except Exception as e:
        print(f"Error opening PDF '{pdf_path}': {e}")


def document_into_chunks(doc: Document) -> List[Document]:
    try:
        # Define parameters for splitting text
        text_splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=100,
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False,
        )
        
        # Split the document into chunks using the defined parameters
        return text_splitter.split_documents([doc])
    
    except Exception as e:
        print(f"Error splitting document into chunks: {e}")
        return []



def adding_data_to_collection(pdf_path: str, chunks: List[Document], text_insert_collection):
    try:
        for i, chunk in enumerate(chunks):
            text_insert_collection.add(
                ids=[f"{pdf_path}__{chunk.metadata['pageno']}__{i}"],
                documents=[chunk.page_content],
                metadatas=[chunk.metadata]
            )
    
    except Exception as e:
        print(f"Error adding chunks to collection: {e}")

        
def extract_images(pdf_paths: List[str], output_folders: List[str]):
   #extract images from multiple pdf
    try:
        # Validate input parameters
        if len(pdf_paths) != len(output_folders):
            raise ValueError("Length of pdf paths is not equal to length of output folders")
        
        # Process each PDF file to extract images
        for pdf_path, output_folder in zip(pdf_paths, output_folders):
            process_images(pdf_path, output_folder)
    
    except ValueError as ve:
        print(f"ValueError: {ve}")
    except Exception as e:
        print(f"Error: {e}")



def process_images(pdf_path, output_folder):
    try:
        document = fitz.open(pdf_path)
        extracted_images = {}

        for page_num in range(document.page_count):
            saved_images = img_page_processing(document, page_num, output_folder, pdf_path)
            extracted_images[page_num + 1] = saved_images

        document.close()
        return extracted_images
    
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")


def img_page_processing(document, page_num, output_folder, pdf_path):
    #process images from a specific page

    # Retrieve the specific page from the document
    page = document[page_num]
    
    # Extract all images from the page
    images = page.get_images(full=True)
    
    # Process each image on the page
    for img_index, img in enumerate(images):
        # Save the image to the output folder
        save_image(img, page_num, img_index, output_folder,document)


def save_image(img, page_num, img_index, output_folder,document):
    
    try:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        xref = img[0]  # Extract the xref (unique identifier) of the image
        base_image = document.extract_image(xref)  # Extract the image data
        image_bytes = base_image["image"]  # Image bytes (binary data)
        image_ext = base_image["ext"]  # Image file extension (e.g., 'png', 'jpg', etc.)
        
        # Construct the image filename based on page number, image index, and extension
        image_filename = f"{output_folder}/{output_folder}_page_{page_num + 1}_image_{img_index + 1}.{image_ext}"
        
        # Save the image to the specified output folder
        with open(image_filename, "wb") as image_file:
            image_file.write(image_bytes)
        
        # Print confirmation message
        print(f"Saved image: {image_filename}")
    
    except Exception as e:
        print(f"Error saving image from page {page_num + 1}, image {img_index + 1}: {e}")




def encoding_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except FileNotFoundError:
        print(f"Error: File '{image_path}' not found.")
    except Exception as e:
        print(f"Error encoding image '{image_path}': {str(e)}")
        return None



def processing_images(image_paths, api_key):
    #processing the encoded images and fowarding to api end point
    results = []
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    model = "gpt-4o"  # Specify the desired model
    
    for image_path in image_paths:
        base64_image = encoding_image(image_path)
        if base64_image:
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Study the image in detail and give me a brief summary"},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                        ],
                    }
                ],
                "max_tokens": 200,
            }
            try:
                response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
                response.raise_for_status()  
                message_response = response.json()['choices'][0]['message']['content']
                results.append(message_response)
            except requests.exceptions.RequestException as e:
                print(f"Error sending request for image '{image_path}': {str(e)}")
            except (KeyError, IndexError) as e:
                print(f"Error parsing response for image '{image_path}': {str(e)}")
    
    return results
 


def add_generated_summ_to_collection(summary, source_pdf, page_num, text_insert_collection):
    
    try:
        # Ensure summary is not empty
        if not summary:
            print(f"Empty summary for: {source_pdf}, Page number: {page_num}")
            return

        # Metadata for the summary
        metadata = {
            'source': source_pdf,
            'pageno': page_num,
        }

        # Generate unique ID based on summary length, source_pdf, and page_num
        unique_id = f"{source_pdf}__{metadata['pageno']}__{len(summary)}"

        # Add to collection with appropriate ID and metadata
        text_insert_collection.add(
            ids=[unique_id],
            documents=summary,   # Assuming summary is the document content
            metadatas=[metadata]   # Metadata associated with the summary
        )

        print("Successfully added summary to:", source_pdf, "pageno:", page_num)
    except Exception as e:
        print(f"Error adding summary to collection: {e}")


def process_images_in_folders(output_folders, api_key, text_insert_collection, pdf_paths):
    for folder, pdf_path in zip(output_folders, pdf_paths):
        image_paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('png', 'jpg', 'jpeg'))]
        for image_path in image_paths:
            base_name = os.path.basename(image_path)
            parts = base_name.split('_')
            try:
                # Find the index of 'page' and extract the subsequent part as page number
                page_index = parts.index('page')
                page_num_str = parts[page_index + 1]  # Get the next part after 'page'
                page_num = int(page_num_str)
            except ValueError:
                print(f"Warning: 'page' keyword not found in file name {base_name}, skipping...")
                continue
            except IndexError:
                print(f"Warning: Unable to extract valid page number from file name {base_name}, skipping...")
                continue

            # Example function calls (you should define these functions as per your needs)
            summary = processing_images([image_path], api_key)
            add_generated_summ_to_collection(summary, pdf_path, page_num, text_insert_collection)


def extract_data(pdf_path, output_folder, text_insert_collection):
    try:
        extract_text(pdf_path,text_insert_collection)
        extract_images(pdf_path,output_folder)
        
    
    except Exception as e:
        print(f"Error extracting data from PDF {pdf_path}: {e}")
