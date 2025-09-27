import json
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import faiss

def load_local_data(data):
    """Loads questions and contexts from the local dataset in SQuAD format."""
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

def save_data(data, filename):
    """Saves a list of strings to a file."""
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(item + '\n')

def generate_embeddings(texts, tokenizer, model, device, max_length=512, batch_size=4):
    """Generates embeddings for a list of texts using a transformer model."""
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        inputs = tokenizer(batch, return_tensors='pt', truncation=True, max_length=max_length, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to GPU if available
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.extend(outputs.last_hidden_state[:, 0, :].cpu().numpy())  # Move to CPU before saving
        del inputs, outputs
        torch.cuda.empty_cache()
    return embeddings

if __name__ == "__main__":
    # Load the local data
    with open('local_dataset.json', 'r') as f:
        data = json.load(f)

    questions, contexts = load_local_data(data)

    # Save the chunked data
    save_data(questions, 'questions.txt')
    save_data(contexts, 'contexts.txt')

    # Print some examples to verify data loading
    print("Verifying data loading...")
    for j in range(5):
        print(f"Question {j+1}: {questions[j]}")
        print(f"Context {j+1}: {contexts[j]}")
        print("-" * 20)

    # Load the embedding model
    print("Loading the embedding model...")
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Generate and save embeddings
    print("Generating embeddings...")
    question_embeddings = generate_embeddings(questions, tokenizer, model, device)
    context_embeddings = generate_embeddings(contexts, tokenizer, model, device)
    np.save('question_embeddings.npy', question_embeddings)
    np.save('context_embeddings.npy', context_embeddings)

    # Combine all embeddings for FAISS
    print("Creating FAISS index...")
    all_embeddings = np.concatenate((question_embeddings, context_embeddings), axis=0)

    # Create and save the FAISS index
    index = faiss.IndexFlatL2(all_embeddings.shape[1])
    index.add(all_embeddings)
    faiss.write_index(index, "local_index.bin")

    # Print completion message
    print("FAISS index created and saved to 'local_index.bin'")
    print(f"Total Question Embeddings shape: {len(question_embeddings)}")
    print(f"Total Context Embeddings shape: {len(context_embeddings)}")
