import fitz
import faiss
import numpy as np
import openai
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# CBA GenAI Studio API configuration
GENAI_API_URL = os.getenv('GENAI_API_URL')
GENAI_API_KEY = os.getenv('GENAI_API_KEY')
client=openai.OpenAI(api_key=GENAI_API_KEY, base_url=GENAI_API_URL, timeout=300)

ALL_MODELS=['bedrock-mistral-large-2402-v1',
    'bedrock-claude-3-5-sonnet-v2',
    'gpt-4o_v2024-05-13_NOFILTER_PTU',
    'bedrock-amazon-titan-text-lite-v1',
    # 'bedrock-amazon-titan-embed-image-v1',
    # 'bedrock-amazon-titan-embed-text-v2',
    # 'text-embedding-ada-002_v2',
    'bedrock-claude-3-7-sonnet',
    'bedrock-claude-3-haiku',
    'gpt-4o_v2024-05-13_NOFILTER_GaaS',
    'bedrock-mistral-7b-instruct-v0',
    # 'text-embedding-3-small_v1',
    'bedrock-amazon-titan-text-express-v1',
    # 'GenAI Assistant',
    'gpt-4o_v2024-05-13',
    'bedrock-claude-3-sonnet',
    'gpt-4o_v2024-05-13_USEAST',
    'gpt-4_vvision-preview',
    'gpt-4o-mini_v2024-07-18',
    # 'bedrock-cohere-embed-eng-v3',
    'gpt-4_v1106-Preview',
    # 'bedrock-cohere-embed-mul-v3',
    # 'bedrock-mistral-small-2402-v1',
    'aipe-claude-3-5-sonnet-v2',
    # 'text-embedding-3-large_v1',
    'o3-mini_v2025-01-31_EASTUS2',
    'gpt-4_v0613'
 ]
# EMB_MODEL="bedrock-cohere-embed-eng-v3"  
# EMB_MODEL="text-embedding-3-large_v1"  
# EMB_MODEL='bedrock-amazon-titan-embed-text-v2'  
EMB_MODEL='text-embedding-ada-002_v2' 

CHAT_MODEL="bedrock-claude-3-5-sonnet-v2"


def ask_ai_about_pdf(relevant_chunks, question, chat_history=None, model=CHAT_MODEL):
    system_prompt = """
    You are a knowledgeable assistant. Use the provided context as the primary source of information to answer the query. 
    If the context is insufficient or lacks details, supplement it with your general knowledge to provide a complete and accurate response. 
    Clearly prioritize the provided context when it applies. Quote the provided context to support your answers.
    Maintain conversational context from the chat history when relevant.
    """
    content = "\n".join(relevant_chunks)
    
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add chat history if available
    if chat_history:
        messages.extend(chat_history)
    
    # Add current context and question
    messages.extend([
        {"role": "user", "content": f"Document Excerpt: {content}"},
        {"role": "user", "content": f"Question: {question}"}
    ])
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=1000
    )
    return response.choices[0].message.content

def extract_and_chunk_pdf(pdf_path, chunk_size=500):
    doc = fitz.open(pdf_path)
    chunks = []
    for page in doc:
        text = page.get_text("text")
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i:i+chunk_size])
    return chunks

# Function to generate embeddings for text chunks
def embed_chunks(chunks):
    response = client.embeddings.create(input=chunks, model=EMB_MODEL)
    return [item.embedding for item in response.data]

# Function to create FAISS index for similarity search
def create_faiss_index(embeddings):
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)  # L2 distance for similarity
    index.add(np.array(embeddings))  # Add embeddings to index
    return index

# Function to find relevant chunk based on query
def search_chunks(queries, chunks, index, embeddings):
    """queries - list of questions/strings
    chunks - list of chunks
    """
    # query_embedding = np.array(get_embedding(query)).reshape(1, -1)
    query_embedding = np.array(embed_chunks(queries))
    distances, indices = index.search(query_embedding, k=5)  # Retrieve top 3 relevant chunks
    return [chunks[i] for i in indices[0]]

def main():
    chunks = extract_and_chunk_pdf('Understanding.pdf', chunk_size=2048)
    embeddings = np.load('embeddings.npy')
    index = create_faiss_index(embeddings)
    chat_history = []
    
    while True:
        question = input("""
=======================================================
Ask about Merchant Garnishee Tool (type 'quit' to stop):
""")
        if question=='quit':
            break
            
        queries = [question]
        relevant_chunks = search_chunks(queries, chunks, index, embeddings)

        answer = ask_ai_about_pdf(relevant_chunks, question, chat_history)
        print("Answer:", answer)
        
        # Update chat history
        chat_history.extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ])
        
        # Keep only last 10 messages to prevent context from growing too large
        if len(chat_history) > 20:
            chat_history = chat_history[-20:]

if __name__ == "__main__":
    main()
