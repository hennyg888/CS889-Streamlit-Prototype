import json
import streamlit as st
from typing import List, Dict, Any

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
    query = st.text_input("Search query", key="search_query", on_change=clear_selection)
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

    results = search_refs(references, query.strip(), fields, year_min if year_min != 1900 else None, year_max if year_max != 2100 else None)

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
            content = f"***{r["title"]}***\n**Journal:** {r.get('journal', '')}    **Year:** {r.get('year', '')}\n{r.get("abstract", "")}"
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
