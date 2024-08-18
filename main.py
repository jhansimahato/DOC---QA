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
from dotenv import load_dotenv
from Extraction_Summarisation import extract_data, process_images_in_folders,Embedding_Text
from Output_Generator import qa_interactive_mode, create_qa_dataframe, save_df_to_excel




def create_chromadb_client():
    try:
        return chromadb.Client()
    except Exception as e:
        print(f"Error creating ChromaDB client: {e}")
        return None


def read_configuration(config_filepath):
    try:
        with open(config_filepath, 'r') as file:
            configuration = json.load(file)
        return configuration
    except FileNotFoundError:
        print(f"Error: The configuration file '{config_filepath}' was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not parse JSON from the configuration file '{config_filepath}'.")
        return None


def locate_pdfs(config_filepath='config.json'):
    config_data = read_configuration(config_filepath)
    if config_data is None:
        return [], []

    folder_path = config_data.get('input_folder', '')

    if not folder_path:
        print("Error: 'document_folder' not defined in the config file.")
        return [], []

    try:
        # Locate all PDF files within the specified folder
        pdf_files = glob.glob(os.path.join(folder_path, '*.pdf'))

        pdf_full_paths = []
        output_dirs = []

        if not pdf_files:
            print(f"No PDF files discovered in '{folder_path}'.")
        else:
            for pdf in pdf_files:
                # Append the absolute path of the PDF file to the list
                pdf_full_paths.append(os.path.abspath(pdf))

            # Generate output directories for each PDF found
            total_pdfs = len(pdf_files)
            for i in range(1, total_pdfs + 1):
                output_directory = os.path.join(os.path.dirname(folder_path), f"output_images_doc{i}")
                output_dirs.append(output_directory)
                os.makedirs(output_directory, exist_ok=True)

        return pdf_full_paths, output_dirs

    except Exception as error:
        print(f"An error occurred: {error}")
        return [], []




def main():
    # Initialize ChromaDB client and collections
    db = create_chromadb_client()
    load_dotenv()
    text_insert_collection = db.get_or_create_collection(name="text_and_image_embeddings", embedding_function=Embedding_Text())
    
    pdf_paths, output_folders = locate_pdfs()

    # Extract data from PDFs
    extract_data(pdf_paths, output_folders, text_insert_collection)
    # API key for OpenAI
    api_key = os.getenv('API_KEY')
    process_images_in_folders(output_folders,api_key,text_insert_collection,pdf_paths)
    # Start interactive session
    qa_pairs = qa_interactive_mode(api_key, text_insert_collection)
    
    # Create DataFrame from qa_pairs
    df = create_qa_dataframe(qa_pairs)

    output_file = 'output.xlsx'

    save_df_to_excel(df, output_file)
    
    # Display or further process the DataFrame
    print("Question-Answer DataFrame:")
    print(df)

# Entry point of the script
if __name__ == "__main__":
    main()