import spacy
import streamlit as st
import pandas as pd
from collections import defaultdict
import re
import subprocess
from spacy.tokens import Span


# Ensure spaCy model is loaded (download if needed)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Load spaCy model and configure it with custom rules
@st.cache_resource
def load_nlp_model_with_custom_rules():
    nlp = spacy.load("en_core_web_sm")
    
    # Add entity ruler with custom rules - using the correct method for newer spaCy versions
    ruler = nlp.add_pipe("entity_ruler", before="ner")
    
    # Add patterns for AI companies and technologies (commonly misclassified)
    patterns = [
        {"label": "ORG", "pattern": "OpenAI"},
        {"label": "ORG", "pattern": "OpenAI Inc"},
        {"label": "ORG", "pattern": "OpenAI Inc."},
        {"label": "ORG", "pattern": "OpenAI Corporation"},
        {"label": "PRODUCT", "pattern": "GPT"},
        {"label": "PRODUCT", "pattern": "GPT-3"},
        {"label": "PRODUCT", "pattern": "GPT-4"},
        {"label": "PRODUCT", "pattern": "ChatGPT"},
        {"label": "ORG", "pattern": "Google DeepMind"},
        {"label": "ORG", "pattern": "DeepMind"},
        {"label": "ORG", "pattern": "Anthropic"},
        {"label": "PRODUCT", "pattern": "Claude"},
        {"label": "ORG", "pattern": "Meta AI"},
        {"label": "ORG", "pattern": "Facebook AI Research"},
        {"label": "ORG", "pattern": "FAIR"},
    ]
    ruler.add_patterns(patterns)
    
    # Custom component to fix "AI" and other tech terms being classified as GPE
    @spacy.Language.component("fix_tech_entities")
    def fix_tech_entities(doc):
        ai_terms = ["AI", "Artificial Intelligence", "Machine Learning", "ML", "NLP"]
        tech_orgs = ["AI Lab", "AI Research", "AI Alliance"]
        
        new_ents = []
        for ent in doc.ents:
            # Fix AI as standalone term
            if ent.text in ai_terms and ent.label_ == "GPE":
                new_ents.append(Span(doc, ent.start, ent.end, label="PRODUCT"))
            # Fix AI as part of organization name
            elif any(term in ent.text for term in tech_orgs) and ent.label_ == "GPE":
                new_ents.append(Span(doc, ent.start, ent.end, label="ORG"))
            else:
                new_ents.append(ent)
        
        doc.ents = new_ents
        return doc
    
    # Add custom component to pipeline
    nlp.add_pipe("fix_tech_entities", after="ner")
    
    return nlp

nlp = load_nlp_model_with_custom_rules()

# App title
st.title("Keyword and Entity Detection App")
st.markdown("""
Paste your text, highlight specific words of your choice, and extract named entities. 
Identify people, organizations, locations, technologies, and much more with enhanced accuracy—all in one tool.
""")

# User input
text = st.text_area("Enter your text:", height=150)

# Advanced options in sidebar
st.sidebar.header("Settings")
highlight_word = st.sidebar.text_input("Input a keyword to highlight (optional)")
highlight_entities = st.sidebar.checkbox("Highlight named entities", value=True)
show_entity_count = st.sidebar.checkbox("Show entity count statistics", value=True)

# Define colors for common NER labels with better contrast
ENTITY_COLORS = {
    "PERSON": "#ffadad",      # red - people
    "ORG": "#ffd6a5",         # orange - companies, agencies, institutions
    "GPE": "#caffbf",         # green - countries, cities, states
    "LOC": "#ffc6ff",         # pink - non-GPE locations, mountain ranges, bodies of water
    "FAC": "#bdb2ff",         # light purple - buildings, airports, highways, bridges
    "PRODUCT": "#a0c4ff",     # light blue - products
    "EVENT": "#fdffb6",       # yellow - named events
    "WORK_OF_ART": "#9bf6ff", # cyan - titles of books, songs, etc.
    "DATE": "#d8f8ff",        # pale blue - dates
    "TIME": "#a0c4ff",        # blue - times
    "MONEY": "#bdb2ff",       # purple - monetary values
    "QUANTITY": "#ffc6ff",    # pink - measurements
    "PERCENT": "#caffbf",     # green - percentage
    "CARDINAL": "#fffffc",    # white - numbers
    "ORDINAL": "#fffffc",     # white - ordinal numbers
    "LANGUAGE": "#ffd6a5",    # orange - languages
    "NORP": "#fdffb6",        # yellow - nationalities, religious or political groups
}

ENTITY_DESCRIPTIONS = {
    "PERSON": "People, including fictional",
    "ORG": "Organizations, companies, agencies",
    "GPE": "Countries, cities, states (Geopolitical Entities)",
    "LOC": "Non-GPE locations, mountain ranges, bodies of water",
    "FAC": "Buildings, airports, highways, bridges",
    "PRODUCT": "Objects, vehicles, foods, tech products",
    "EVENT": "Named events like wars, sports events, hurricanes",
    "WORK_OF_ART": "Titles of books, songs, etc.",
    "DATE": "Absolute or relative dates or periods",
    "TIME": "Times smaller than a day",
    "MONEY": "Monetary values, including unit",
    "QUANTITY": "Measurements, as of weight or distance",
    "PERCENT": "Percentage",
    "CARDINAL": "Numerals that do not fall under another type",
    "ORDINAL": "Ordinal numbers like 'first', 'second'",
    "LANGUAGE": "Any named language",
    "NORP": "Nationalities, religious or political groups",
}

# Add example text
with st.expander("Show example text to try"):
    example_text = """
    Microsoft announced a $10 billion investment in OpenAI on March 24, 2023. 
    The deal was signed by Satya Nadella, CEO of Microsoft. 
    OpenAI, based in San Francisco, California, is known for developing ChatGPT.
    The company plans to use these funds to enhance their AI research capabilities.
    Google DeepMind and Anthropic are also major players in the AI industry.
    """
    st.markdown(example_text)
    if st.button("Use this example"):
        st.session_state.text = example_text
        st.rerun()

# Load example if in session state
if 'text' in st.session_state:
    text = st.session_state.text
    # Clear the session state to avoid looping
    del st.session_state.text

# Processing
if text:
    doc = nlp(text)
    st.toast("Entities extracted below!")

    # 1. Collect word highlight spans
    highlights = []
    if highlight_word:
        pattern = re.compile(r'\b' + re.escape(highlight_word) + r'\b', re.IGNORECASE)
        for match in pattern.finditer(text):
            highlights.append((match.start(), match.end(), "#ffff00"))  # yellow

    # 2. Add entity highlight spans if enabled
    entities = []
    entity_counts = defaultdict(int)
    
    if highlight_entities:
        for ent in doc.ents:
            color = ENTITY_COLORS.get(ent.label_, "#e0e0e0")  # default light gray
            highlights.append((ent.start_char, ent.end_char, color))
            entities.append((ent.text, ent.label_))
            entity_counts[ent.label_] += 1

    # 3. Sort and build highlighted HTML
    highlights.sort(key=lambda x: x[0])  # sort by start index
    highlighted_text = ""
    last_idx = 0

    for start, end, color in highlights:
        # Handle overlapping spans by taking the later one
        if start < last_idx:
            continue
        highlighted_text += text[last_idx:start]
        highlighted_text += f"<span style='background-color: {color}; padding: 1px 3px; border-radius: 3px;'>{text[start:end]}</span>"
        last_idx = end

    highlighted_text += text[last_idx:]  # Add remaining text

    # Display results
    col1, col2 = st.columns([2, 1]) 
    
    with col1:
        st.markdown("### Highlighted Text")
        st.markdown(highlighted_text, unsafe_allow_html=True)
    
    with col2:
        # Entity legend with descriptions
        st.markdown("### Entity Types")
        for label, count in sorted(entity_counts.items(), key=lambda x: x[1], reverse=True):
            if label in ENTITY_COLORS:
                color = ENTITY_COLORS[label]
                description = ENTITY_DESCRIPTIONS.get(label, "")
                st.markdown(f"<div><span style='background-color: {color}; padding:2px 8px; margin-right:5px; border-radius:3px;'>{label}</span> {description} ({count})</div>", unsafe_allow_html=True)

    # Display extracted entities in a table with better formatting
    if entities:
        st.markdown("### Extracted Entities")
        entity_df = pd.DataFrame(entities, columns=["Entity", "Label"])
        st.dataframe(entity_df, use_container_width=True)

        # Add export options
        csv = entity_df.to_csv(index=False)
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name="extracted_entities.csv",
            mime="text/csv"
        )
        
    # Show entity distribution if enabled
    if show_entity_count and entity_counts:
        st.markdown("### Entity Distribution")
        entity_count_df = pd.DataFrame(list(entity_counts.items()), columns=['Entity Type', 'Count'])
        entity_count_df = entity_count_df.sort_values('Count', ascending=False)
        
        # Display as bar chart
        st.bar_chart(entity_count_df.set_index('Entity Type'))

# Add explanation of custom rules
with st.expander("About custom entity rules"):
    st.markdown("""
    ### Custom Entity Recognition Rules
    
    This application includes custom rules to improve the recognition of technology-related entities:
    
    1. **AI Companies**: Custom patterns ensure companies like OpenAI, DeepMind, and Anthropic are correctly labeled as Organizations (ORG).
    
    2. **AI Products**: Products like ChatGPT, GPT-4, and Claude are properly categorized as Products.
    
    3. **Tech Concepts**: Terms like "AI" and "Machine Learning" are properly categorized instead of being mistaken for locations.
    
    These rules help overcome limitations in the base spaCy model when dealing with newer technology organizations and concepts.
    """)