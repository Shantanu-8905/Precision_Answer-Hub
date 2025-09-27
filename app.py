import json
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
import ollama
import streamlit as st
import logging
import sys

# Configure logging and version compatibility
logging.getLogger("torch").setLevel(logging.ERROR)

def check_versions():
    if torch.__version__.startswith('2'):
        faiss.get_num_gpus = lambda: 0  # Workaround for torch 2.x
    st.sidebar.write(f"PyTorch version: {torch.__version__}")
    st.sidebar.write(f"FAISS version: {faiss.__version__}")


# Load and prepare data
def load_local_data(data):
    questions, contexts = [], []
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
    embeddings = model.encode(texts, convert_to_tensor=True, batch_size=batch_size)
    return embeddings.cpu().numpy().astype(np.float32)

# Embedding model initialization
@st.cache_resource
def initialize_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    return model

# Load and process local dataset
def load_data_and_create_index(dataset_path='local_dataset.json'):
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    return load_local_data(data)

# Create FAISS index if it doesn't exist
@st.cache_resource
def create_or_load_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    # Explicitly use CPU index
    index = faiss.IndexFlatL2(dimension)
    if faiss.get_num_gpus() > 0:
        # If GPU is available, make it a CPU to GPU index
        index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)
    index.add(embeddings)
    return index

# Retrieve contexts using FAISS
def retrieve_contexts(query, index, model, contexts, k=12):  # Increased k for better coverage
    query_embedding = generate_embeddings([query], model)
    distances, indices = index.search(np.array(query_embedding, dtype=np.float32), k)
    
    # Filter and rank contexts based on semantic similarity
    relevant_contexts = []
    for idx, distance in zip(indices[0], distances[0]):
        if idx != -1 and distance < 1.5:  # Stricter threshold for better relevance
            relevant_contexts.append((contexts[idx], distance))
    
    # Sort by relevance and take top 5 most relevant contexts
    relevant_contexts.sort(key=lambda x: x[1])
    return [ctx for ctx, _ in relevant_contexts[:5]] if relevant_contexts else None

# Generate response using Ollama's llama model
def generate_response_with_ollama(query, contexts, model_name='deepseek-r1'):
    system_prompt = """You are a precise and direct AI assistant. Format your response as follows:
    1. If you can answer the question from the context, provide the answer clearly and directly
    2. If you cannot answer the question completely, start with "Based on the available information:" and share what is known
    3. Use bullet points for listing multiple points
    4. Keep responses concise and factual
    DO NOT include phrases like "I think", "it seems", or "the context suggests". Just state the facts directly."""
    
    formatted_contexts = "\n\n".join([f"{ctx.strip()}" for ctx in contexts])
    input_text = f"Question: {query}\n\nContext: {formatted_contexts}"
    
    # Generate response using Ollama
    response = ollama.chat(
        model=model_name,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': input_text}
        ]
    )
    return response['message']['content']

# Streamlit UI
def main():
    # Set up the page configuration
    st.set_page_config(
        page_title="Precision AnswerHub",
        page_icon="🔍",
        layout="wide"
    )

    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
            max-width: 1200px;
            margin: 0 auto;
        }
        .stTextInput > div > div > input {
            padding: 15px;
            border-radius: 10px;
            border: 2px solid #e0e0e0;
            font-size: 16px;
        }
        .stTextInput > div > div > input:focus {
            border-color: #7792E3;
            box-shadow: 0 0 10px rgba(119, 146, 227, 0.2);
        }
        .stButton>button {
            width: 100%;
            padding: 12px;
            border-radius: 10px;
            background-color: #7792E3;
            color: white;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #5A73B3;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .css-1544g2n.e1fqkh3o4 {
            padding: 2rem;
            border-radius: 15px;
            background-color: #f8f9fa;
        }
        </style>
    """, unsafe_allow_html=True)

    # Create two columns for the header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🔍 Precision AnswerHub")
        st.markdown("### AI-Enhanced Retrieval & Generation")
    with col2:
        check_versions()

    # Initialize models and load data with a spinner
    with st.spinner("Initializing models and loading data..."):
        model = initialize_model()
        questions, contexts = load_data_and_create_index('local_dataset.json')
        context_embeddings = generate_embeddings(contexts, model)
        index = create_or_load_faiss_index(context_embeddings)

    # Create three columns for better layout
    left_col, center_col, right_col = st.columns([1, 2, 1])

    with center_col:
        # Input query form with better styling
        st.markdown("### Ask your question")
        with st.form(key="query_form"):
            query = st.text_input(
                label="Question Input",
                placeholder="Enter your question here...",
                help="Type your question and press Enter or click Submit",
                label_visibility="collapsed"
            )
            submit_button = st.form_submit_button(label="🔎 Submit", use_container_width=True)

    if submit_button and query:
        with st.spinner("🔍 Searching for relevant information..."):
            retrieved_contexts = retrieve_contexts(query, index, model, contexts)
        
        if retrieved_contexts:
            # Display contexts and response in tabs
            tab1, tab2 = st.tabs(["📝 Answer", "🔍 Related Contexts"])
            
            with tab1:
                with st.spinner("🔍 Finding answer..."):
                    raw_response = generate_response_with_ollama(query, retrieved_contexts)
                    # Clean and format the response
                    response = raw_response.replace("think:", "").replace("Context:", "")
                    response = "\n".join([line for line in response.split("\n") 
                                        if not line.strip().startswith(("<", "think", "Context"))
                                        and not "analyzing" in line.lower()])
                    st.markdown("### Answer")
                    st.markdown(response.strip())
                    
            with tab2:
                st.markdown("### Retrieved Contexts")
                for i, context in enumerate(retrieved_contexts, 1):
                    with st.expander(f"Context {i}"):
                        st.markdown(context)
        else:
            st.error("😕 No relevant information found. Please try rephrasing your question.")

if __name__ == "__main__":
    main()
