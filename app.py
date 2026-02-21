import json
import os
import streamlit as st
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

# --- Cached model & embeddings loaders ---
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_data
def load_field_embeddings() -> Dict[str, Optional[np.ndarray]]:
    files = {
        "title": "titles_embeddings.npy",
        "journal": "journals_embeddings.npy",
        "abstract": "abstracts_embeddings.npy",
        "keywords": "keywords_embeddings.npy",
    }
    embs: Dict[str, Optional[np.ndarray]] = {}
    for k, path in files.items():
        if os.path.exists(path):
            embs[k] = np.load(path)
        else:
            embs[k] = None
    return embs


def search_semantic(references: List[Dict[str, Any]], query: str, fields: List[str], year_min: Optional[int], year_max: Optional[int], top_k: int = 50) -> List[Dict[str, Any]]:
    """Perform semantic search across selected fields using precomputed embeddings.
    Returns top_k references sorted by aggregated similarity score."""
    if not query:
        return []

    # Load embeddings and model
    embs = load_field_embeddings()
    model = load_model()

    # Check availability
    available = [f for f in fields if embs.get(f) is not None]
    if not available:
        return []

    # Encode query
    q_emb = model.encode([query], normalize_embeddings=True)[0]

    # Aggregate similarity scores (average across fields)
    scores = None
    for f in available:
        field_emb = embs[f]
        # Use dot product because embeddings are normalized (cosine)
        s = field_emb.dot(q_emb) if field_emb is not None else np.zeros(len(references))
        if scores is None:
            scores = s
        else:
            scores = scores + s
    scores = scores / len(available)

    # Collect and filter by year
    scored = []
    for i, r in enumerate(references):
        if year_min is not None and r.get("year") < year_min:
            continue
        if year_max is not None and r.get("year") > year_max:
            continue
        scored.append((scores[i], r))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Dynamic threshold: keep only results with score greater than the
    # average score among the (year-filtered) candidates and return up to 10 items
    if not scored:
        return []
    scores_list = [s for s, _ in scored]
    avg_score = float(np.mean(scores_list))
    filtered = [(s, r) for s, r in scored if s > avg_score*1.1]
    return [r for s, r in filtered[:top_k]]

DATA_PATH = "example-bib.json"

@st.cache_data
def load_references(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("references", [])


def clear_selection():
    st.session_state.selected_ref = None


def select_ref(ref_id: int):
    st.session_state.selected_ref = ref_id


def log_search(query: str, fields: List[str], result_ids: List[int], ai_semantic: bool) -> None:
    """Append a human-readable log entry for a search."""
    from datetime import datetime

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mode = "AI semantic" if ai_semantic else "lexical"
    entry = (
        f"[{ts}] {mode} search | query='{query}' | fields={fields} | results={result_ids}\n"
    )
    with open("search.log", "a", encoding="utf-8") as f:
        f.write(entry)


def search_refs(references: List[Dict[str, Any]], query: str, fields: List[str], year_min: int | None, year_max: int | None) -> List[Dict[str, Any]]:
    if not query and year_min is None and year_max is None:
        return references

    q = query.lower() if query else ""
    results = []
    for r in references:
        matched = False
        if q:
            for f in fields:
                if f in r:
                    val = r[f]
                    if isinstance(val, list):
                        joined = " ".join(map(str, val)).lower()
                        if q in joined:
                            matched = True
                            break
                    else:
                        if q in str(val).lower():
                            matched = True
                            break
        else:
            matched = True

        if matched:
            if year_min is not None and r.get("year") < year_min:
                continue
            if year_max is not None and r.get("year") > year_max:
                continue
            results.append(r)
    return results


# --- Initialize session state ---
if "selected_ref" not in st.session_state:
    st.session_state.selected_ref = None

# --- App UI ---
st.set_page_config(page_title="Research Reference Manager", layout="wide")
st.title("📚 Research Reference Manager")
#st.write("Load references from `example-bib.json`. Click any title to view full details on the right.")

references = load_references(DATA_PATH)

left_col, mid_col, right_col = st.columns([1, 2, 2])

# LEFT COLUMN: list of all references (title boxes)
with left_col:
    st.header("All references")
    st.write("Click a title to view full details on the right.")
    with st.container(height=1000):
        # Render a scrollable list of titles as buttons
        st.markdown('<div style="max-height:420px; overflow-y:auto; padding:4px;">', unsafe_allow_html=True)
        for r in references:
            if st.button(r["title"], key=f"left_{r['id']}", use_container_width=True):
                select_ref(r["id"])
        st.markdown('</div>', unsafe_allow_html=True)

# MIDDLE COLUMN: search bar + results
with mid_col:
    st.header("Search")
    # When search inputs change, clear current selection (clicking elsewhere should remove focus)
    query = st.chat_input("Search query")

    ai_semantic = st.checkbox("AI Semantic Search", key="ai_semantic", on_change=clear_selection)

    if ai_semantic:
        # For semantic search limit fields to title/journal/abstract/keywords
        fields = st.multiselect(
            "Search in fields (AI)",
            options=["title", "journal", "abstract", "keywords"],
            default=["title", "abstract"],
            key="search_fields_ai",
            on_change=clear_selection,
        )
        if not any([field for field in fields]):
            st.info("Select at least one field for AI Semantic Search")
    else:
        fields = st.multiselect(
            "Search in fields",
            options=["title", "abstract", "authors", "journal", "keywords", "year"],
            default=["title", "abstract"],
            key="search_fields",
            on_change=clear_selection,
        )

    st.write("Optional year filter")
    col_a, col_b = st.columns(2)
    year_min = col_a.number_input("Year min", min_value=1900, max_value=2100, value=1900, key="year_min", on_change=clear_selection)
    year_max = col_b.number_input("Year max", min_value=1900, max_value=2100, value=2100, key="year_max", on_change=clear_selection)

    # Choose search mode (lexical vs semantic)
    results = []
    if query is not None:
        if ai_semantic:
            # Map UI field names to embeddings keys (they match here)
            # Check embeddings availability first
            embs = load_field_embeddings()
            available = [f for f in fields if embs.get(f) is not None]
            if not available:
                st.warning("Embeddings not found for selected fields. Run `embed.py` to generate embeddings or uncheck AI Semantic Search.")
                results = search_refs(references, query.strip(), fields, year_min if year_min != 1900 else None, year_max if year_max != 2100 else None)
            else:
                results = search_semantic(references, query.strip(), fields, year_min if year_min != 1900 else None, year_max if year_max != 2100 else None, 10)
        else:
            results = search_refs(references, query.strip(), fields, year_min if year_min != 1900 else None, year_max if year_max != 2100 else None)

        # log the search event
        try:
            result_ids = [r["id"] for r in results] if results else []
            log_search(query.strip(), fields, result_ids, ai_semantic)
        except Exception as e:
            # If logging fails, just ignore to not break app
            st.error(f"Logging error: {e}")

    st.write(f"Found {len(results)} result(s)")

    # Render results as a scrollable list of clickable boxes
    #st.markdown('<div style="max-height:420px; overflow-y:auto; padding:4px;">', unsafe_allow_html=True)
    with st.container(height=600):
        st.markdown("""
            <style>
            div.stButton > button {
                white-space: pre-line;
            }
            </style>
            """, unsafe_allow_html=True)
        for r in results:
            st.markdown('<div class="special-btn">', unsafe_allow_html=True)
            content = f'***{r["title"]}***\n**Journal:** {r.get("journal", "")}    **Year:** {r.get("year", "")}\n{r.get("abstract", "")}'
            if st.button(content, key=f"mid_{r['id']}", use_container_width=True):
                select_ref(r["id"])
                #st.write(f"**Journal:** {r.get('journal', '')}    **Year:** {r.get('year', '')}")
                #st.write(r.get("abstract", ""))
            st.markdown('</div>', unsafe_allow_html=True)

# RIGHT COLUMN: detailed view of the selected reference
with right_col:
    st.header("Reference details")
    if st.session_state.selected_ref is None:
        st.write("No reference selected.")
    else:
        sel = next((x for x in references if x["id"] == st.session_state.selected_ref), None)
        if sel:
            st.subheader(sel["title"])
            st.write(f"**Authors:** {', '.join(sel.get('authors', []))}")
            st.write(f"**Journal:** {sel.get('journal', '')}")
            st.write(f"**Year:** {sel.get('year', '')}")
            st.write(f"**Volume/Issue/Pages:** {sel.get('volume', '')}/{sel.get('issue', '')} {sel.get('pages', '')}")
            st.write(f"**DOI:** {sel.get('doi', '')}")
            st.write("---")
            st.write(sel.get("abstract", ""))
            if sel.get("keywords"):
                st.write(f"**Keywords:** {', '.join(sel.get('keywords'))}")
        else:
            st.write("Selected reference could not be found.")

# Footer / Tips
st.markdown("---")
st.caption("Tip: Click a title on the left or click a result title in the middle to view details. Changing search inputs will hide the right column details.")
