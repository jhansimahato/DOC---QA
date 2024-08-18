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


def qa_interactive_mode(api_token, text_collection):
    question_answer_pairs = []
    
    while True:
        user_query = input("Please enter your query (type @quit to end): ")
        
        if user_query.strip() == "@quit":
            print("Ending session.")
            break
        else:
            try:
                answer = handle_query(api_token, text_collection, user_query)
                if answer is not None:
                    question_answer_pairs.extend(answer)
                else:
                    print("Unable to process the question. Please try again.")
            except Exception as error:
                print(f"An error occurred while processing the question: {error}")
    
    return question_answer_pairs



def handle_query(api_token, text_collection, user_query):
    try:
        client = OpenAI(api_key=api_token)
        query_results = []

        # Query the text collection to find similar documents
        similar_documents = text_collection.query(query_texts=user_query, n_results=10)
        print(similar_documents)
        context_text = similar_documents["documents"][0]
        metadata = similar_documents['metadatas'][0]

        # Construct the conversation history for the AI model
        dialogue = [
            {"role": "assistant", "content": "Use the given context to respond to the question below. If you do not have enough information to answer, state that clearly instead of guessing."},
            {"role": "user", "content": f"context:\n{context_text}\n\nQuestion: {user_query}"}
        ]

        # Generate AI response using the GPT-4 model
        ai_response = client.chat.completions.create(
            model="gpt-4",
            messages=dialogue,
            temperature=0,
            max_tokens=1000
        )

        # Extract and save the answer along with metadata
        response_message = ai_response.choices[0].message
        query_results.append((user_query, response_message.content, metadata[0]['pageno'], metadata[0]['source'][10:]))

        return query_results

    except Exception as error:
        print(f"An error occurred while processing the question: {error}")
        return None




def create_qa_dataframe(qa_pairs):
    columns = ['Question', 'Answer', 'PageNo', 'Source_document']
    
    try:
        df = pd.DataFrame(qa_pairs, columns=columns)
    except Exception as e:
        print(f"Error occurred while creating DataFrame: {e}")
        return None
    
    return df


def save_df_to_excel(df, excel_file):
    
    try:
        df.to_excel(excel_file, index=False)
        print(f"DataFrame successfully saved to {excel_file}")
        return True
    except Exception as e:
        print(f"Error occurred while saving DataFrame to {excel_file}: {e}")
        return False