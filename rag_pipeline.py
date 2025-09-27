# import json
# import numpy as np
# import faiss
# import torch
# from sentence_transformers import SentenceTransformer
# import ollama  # Import ollama library
# # from some_module import save_data
# from utils import save_data
# from rag.utils import save_data

# # Load and prepare data
# def load_local_data(data):
#     """Loads questions and contexts from a local JSON dataset in SQuAD-like format."""
#     questions = []
#     contexts = []
#     for article in data['data']:
#         for paragraph in article['paragraphs']:
#             context = paragraph['context']
#             for qa in paragraph['qas']:
#                 question = qa['question']
#                 questions.append(question)
#                 contexts.append(context)
#     return questions, contexts

# # Save data for later use
# def save_data(data, filename):
#     """Saves a list of strings to a file."""
#     with open(filename, 'w', encoding='utf-8') as f:
#         for item in data:
#             f.write(item + '\n')

# # Generate embeddings with the model
# def generate_embeddings(texts, model, batch_size=32):
#     """Generates embeddings for a list of texts using a sentence-transformer model."""
#     embeddings = model.encode(texts, convert_to_tensor=True, batch_size=batch_size)
#     return embeddings.cpu().numpy().astype(np.float32)

# # Embedding model initialization
# def initialize_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
#     """Initializes the sentence transformer model on the appropriate device."""
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = SentenceTransformer(model_name, device=device)
#     return model

# # Load and process local dataset
# def load_data_and_create_index(dataset_path='local_dataset.json'):
#     """Loads data from JSON, processes questions and contexts, and saves them to text files."""
#     try:
#         with open(dataset_path, 'r') as f:
#             data = json.load(f)
#     except FileNotFoundError:
#         print(f"Dataset file '{dataset_path}' not found.")
#         return None, None

#     questions, contexts = load_local_data(data)
#     save_data(questions, 'questions.txt')
#     save_data(contexts, 'contexts.txt')
#     return questions, contexts

# # Create and save FAISS index
# def create_faiss_index(embeddings, index_path='local_index.bin'):
#     """Creates a FAISS index for the embeddings and saves it to a file."""
#     dimension = embeddings.shape[1]
#     index = faiss.IndexFlatL2(dimension)
#     index.add(embeddings)
#     faiss.write_index(index, index_path)
#     return index

# # Retrieve contexts using FAISS
# def retrieve_contexts(query, index, model, contexts, k=5):
#     """Retrieves the top-k contexts most similar to the query."""
#     query_embedding = generate_embeddings([query], model)
#     distances, indices = index.search(np.array(query_embedding, dtype=np.float32), k)
#     return [contexts[idx] for idx in indices[0] if idx != -1]

# # Generate response using Ollama's llama model
# def generate_response_with_ollama(query, contexts, model_name='tinyllama'):
#     """Generates a response to a query using the retrieved contexts and an Ollama model."""
#     combined_contexts = " ".join(contexts)
#     input_text = f"Answer the question: '{query}' based on the following information: {combined_contexts}"

#     # Generate response using Ollama
#     response = ollama.chat(
#         model=model_name,
#         messages=[{'role': 'user', 'content': input_text}]
#     )
#     return response['message']['content']

# # Execute the entire pipeline
# if __name__ == "__main__":
#     # Initialize the embedding model
#     model = initialize_model("sentence-transformers/all-MiniLM-L6-v2")

#     # Load data and create questions and contexts files
#     questions, contexts = load_data_and_create_index('local_dataset.json')
#     if questions is None or contexts is None:
#         print("Failed to load data. Exiting...")
#         exit()

#     # Generate embeddings for contexts
#     context_embeddings = generate_embeddings(contexts, model)
#     np.save('context_embeddings.npy', context_embeddings)

#     # Create and save the FAISS index
#     index = create_faiss_index(context_embeddings, 'local_index.bin')
#     print("FAISS index created and saved.")

#     # Example query and retrieve similar contexts
#     example_query = "What is the distribution of news articles across different categories?"
#     retrieved_contexts = retrieve_contexts(example_query, index, model, contexts)

#     # Generate and display response using Ollama
#     response = generate_response_with_ollama(example_query, retrieved_contexts)
#     print("\nGenerated response:")
#     print(response)

import sys
import os

# Add the current script directory to Python's module search path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
import ollama  # Import ollama library
from utils import save_data  # Now Python can find utils.py

# Load and prepare data
def load_local_data(data):
    """Loads questions and contexts from a local JSON dataset in SQuAD-like format."""
    questions = []
    contexts = []
    for article in data['data']:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                question = qa['question']
                questions.append(question)
                contexts.append(context)
    return questions, contexts

# Generate embeddings with the model
def generate_embeddings(texts, model, batch_size=32):
    """Generates embeddings for a list of texts using a sentence-transformer model."""
    embeddings = model.encode(texts, convert_to_tensor=True, batch_size=batch_size)
    return embeddings.cpu().numpy().astype(np.float32)

# Embedding model initialization
def initialize_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Initializes the sentence transformer model on the appropriate device."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    return model

# Load and process local dataset
def load_data_and_create_index(dataset_path='local_dataset.json'):
    """Loads data from JSON, processes questions and contexts, and saves them to text files."""
    try:
        with open(dataset_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Dataset file '{dataset_path}' not found.")
        return None, None

    questions, contexts = load_local_data(data)
    save_data(questions, 'questions.txt')
    save_data(contexts, 'contexts.txt')
    return questions, contexts

# Create and save FAISS index
def create_faiss_index(embeddings, index_path='local_index.bin'):
    """Creates a FAISS index for the embeddings and saves it to a file."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, index_path)
    return index

# Retrieve contexts using FAISS
def retrieve_contexts(query, index, model, contexts, k=12):
    """Retrieves and ranks the most relevant contexts for the query."""
    query_embedding = generate_embeddings([query], model)
    distances, indices = index.search(query_embedding.astype(np.float32), k)
    
    # Filter and rank contexts based on semantic similarity
    relevant_contexts = []
    for idx, distance in zip(indices[0], distances[0]):
        if 0 <= idx < len(contexts) and distance < 1.5:  # Stricter relevance threshold
            relevant_contexts.append((contexts[idx], distance))
    
    # Sort by relevance and return top contexts
    relevant_contexts.sort(key=lambda x: x[1])
    return [ctx for ctx, _ in relevant_contexts[:5]] if relevant_contexts else None

# Generate response using Ollama's llama model
def generate_response_with_ollama(query, contexts, model_name='deepseek-r1'):
    """Generates a structured and accurate response using retrieved contexts."""
    system_prompt = """You are a helpful AI assistant that provides accurate, concise answers based on the given context. Follow these rules:
1. Use ONLY the information provided in the context
2. If the context doesn't fully answer the question, clearly state what information is missing
3. Structure your response in clear paragraphs
4. Support your answer with specific examples from the context
5. If the context is insufficient, acknowledge this and provide any relevant partial information
6. Be direct and clear in your response"""
    
    formatted_contexts = "\n\n".join([f"Context {i+1}:\n{ctx.strip()}" for i, ctx in enumerate(contexts)])
    input_text = f"Question: {query}\n\nRelevant Information:\n{formatted_contexts}\n\nBased solely on the above information, provide a clear and structured answer. If the information is not sufficient to answer the question completely, acknowledge this limitation and share what can be determined from the available context."
    
    # Generate response using Ollama
    response = ollama.chat(
        model=model_name,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': input_text}
        ])
    return response['message']['content']

# Execute the entire pipeline
if __name__ == "__main__":
    # Initialize the embedding model
    model = initialize_model("sentence-transformers/all-MiniLM-L6-v2")

    # Load data and create questions and contexts files
    questions, contexts = load_data_and_create_index('local_dataset.json')
    if questions is None or contexts is None:
        print("Failed to load data. Exiting...")
        exit()

    # Generate embeddings for contexts
    context_embeddings = generate_embeddings(contexts, model)
    np.save('context_embeddings.npy', context_embeddings)

    # Create and save the FAISS index
    index = create_faiss_index(context_embeddings, 'local_index.bin')
    print("FAISS index created and saved.")

    # Example query and retrieve similar contexts
    example_query = "What is the distribution of news articles across different categories?"
    retrieved_contexts = retrieve_contexts(example_query, index, model, contexts)

    # Generate and display response using Ollama
    response = generate_response_with_ollama(example_query, retrieved_contexts)
    print("\nGenerated response:")
    print(response)
