import json
import os
from sentence_transformers import SentenceTransformer
import numpy as np

# Path to the bibliography JSON
BIB_PATH = "example-bib.json"

# Output filenames
OUTPUT_FILES = {
    "titles": "titles_embeddings.npy",
    "journals": "journals_embeddings.npy",
    "abstracts": "abstracts_embeddings.npy",
    "keywords": "keywords_embeddings.npy",
}


def load_references(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("references", [])


def text_lists_from_refs(refs):
    titles = [ref.get("title", "") for ref in refs]
    journals = [ref.get("journal", "") for ref in refs]
    abstracts = [ref.get("abstract", "") for ref in refs]
    # Join keywords list into a single string per reference
    keywords = [", ".join(ref.get("keywords", [])) for ref in refs]
    return {
        "titles": titles,
        "journals": journals,
        "abstracts": abstracts,
        "keywords": keywords,
    }


def main():
    if not os.path.exists(BIB_PATH):
        raise FileNotFoundError(f"Bibliography file not found: {BIB_PATH}")

    refs = load_references(BIB_PATH)
    if not refs:
        raise ValueError("No references found in bibliography file.")

    texts = text_lists_from_refs(refs)

    print(f"Loading model: all-MiniLM-L6-v2")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    for key, text_list in texts.items():
        print(f"Encoding {key} ({len(text_list)} items)")
        embeddings = model.encode(text_list, normalize_embeddings=True)
        out_file = OUTPUT_FILES[key]
        np.save(out_file, embeddings)
        print(f"Saved {key} embeddings to {out_file} with shape {embeddings.shape}")


if __name__ == "__main__":
    main()

    @st.cache_resource
    def load_model():
        return SentenceTransformer("all-MiniLM-L6-v2")

    # Cache embeddings
    @st.cache_data
    def load_embeddings():
        return np.load("embeddings.npy")
    
    model = load_model()
    embeddings = load_embeddings()

    query_embedding = model.encode([query], normalize_embeddings=True)

        # Cosine similarity
    scores = cosine_similarity(query_embedding, embeddings)[0]

        # Sort
    top_k = 3
    top_indices = np.argsort(scores)[::-1][:top_k]

    for idx in top_indices:
        print(f"Score: {scores[idx]:.3f}")
        print(docs[idx])