
import streamlit as st
from generate_response import generate_response
import time
import os
import webbrowser

# Set page config
st.set_page_config(page_title="Telecom AI Assistant", layout="wide")

# Initialize session state
if "response" not in st.session_state:
    st.session_state.response = ""
if "relevant_doc" not in st.session_state:
    st.session_state.relevant_doc = ""
if "doc_type" not in st.session_state:
    st.session_state.doc_type = "report"
if "processing" not in st.session_state:
    st.session_state.processing = False

# Custom CSS styling
st.markdown("""
    <style>
        .stApp {
            background-color: white;
            font-family: 'Segoe UI', sans-serif;
        }
        .title-header {
            font-size: 36px;
            font-weight: bold;
            text-align: center;
            color: #2e7bcf;
            margin-bottom: 20px;
        }
        .mode-box {
            background-color: #F5F5DC;
            padding: 1rem;
            border-radius: 10px;
        }
        .query-box, .output-box {
            background-color: #f9f9f9;
            padding: 1.5rem;
            border-radius: 12px;
            margin-top: 1rem;
        }
        .stTextArea textarea {
            font-size: 16px !important;
        }
        .stButton > button {
            background-color: #FAF3E0;
            color: Black;
            border: Black;
            font-size: 16px;
            font-weight: bold;
            padding: 8px 20px;
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<div class='title-header'>📡 Telecom AI Assistant 🤖</div>", unsafe_allow_html=True)

# Two-column layout: Mode left, content right
col_left, col_right = st.columns([1, 4], gap="large")

with col_left:
    with st.container():
        st.markdown("### ⚙️ Mode")
        mode = st.radio(
            "",
            ("Knowledge Search", "Generate Document"),
            key="mode",
            label_visibility="collapsed"
        )

with col_right:
    def clear_output():
        st.session_state.response = ""
        st.session_state.relevant_doc = ""
        st.session_state.doc_type = "report"
        st.session_state.processing = False

    if mode == "Knowledge Search":
        st.markdown("#### 🔍 Knowledge Search", unsafe_allow_html=True)
        with st.container():
            query = st.text_area("Enter your telecom-related query:", "", key="query_search")

            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Get Response"):
                    if query.strip():
                        st.session_state.processing = True
                        st.rerun()
                    else:
                        st.warning("Please enter a query.")
            with col2:
                st.button("Clear", on_click=clear_output)

            if st.session_state.processing:
                with st.spinner("Generating response..."):
                    time.sleep(1.5)
                    response, relevant_doc, _ = generate_response(query)
                    st.session_state.response = response
                    st.session_state.relevant_doc = relevant_doc
                    st.session_state.doc_type = None
                    st.session_state.processing = False
                    st.rerun()

    elif mode == "Generate Document":
        st.markdown("#### 📝 Generate Document", unsafe_allow_html=True)
        with st.container():
            query = st.text_area("Enter your telecom-related query:", "", key="query_doc")

            doc_type = st.radio(
                "Select Document Type",
                ["Report", "SOP", "Summary"],
                horizontal=True
            )

            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Generate Document"):
                    if query.strip():
                        st.session_state.processing = True
                        st.rerun()
                    else:
                        st.warning("Please enter a query.")
            with col2:
                st.button("Clear", on_click=clear_output)

            if st.session_state.processing:
                with st.spinner("Generating document..."):
                    time.sleep(1.5)
                    response, relevant_doc, selected_doc_type = generate_response(query, doc_type.lower())
                    st.session_state.response = response
                    st.session_state.relevant_doc = relevant_doc
                    st.session_state.doc_type = selected_doc_type
                    st.session_state.processing = False
                    st.rerun()

    # Output section
    if st.session_state.response and not st.session_state.processing:
        st.markdown("#### 📋 Output", unsafe_allow_html=True)
        with st.container():
            # Relevant document
            if st.session_state.relevant_doc and st.session_state.relevant_doc != "No relevant document found.":
                processed_file_name = st.session_state.relevant_doc
                raw_file_base_name = os.path.splitext(processed_file_name)[0]
                raw_dir = "data/raw"
                possible_extensions = ['.pdf', '.docx', '.txt']
                raw_file_path = None
                for ext in possible_extensions:
                    potential_path = os.path.join(raw_dir, raw_file_base_name + ext)
                    if os.path.exists(potential_path):
                        raw_file_path = potential_path
                        break
                st.markdown("**📄 Relevant Document:**")
                if raw_file_path:
                    st.write(os.path.basename(raw_file_path))
                    if st.button("Open Document"):
                        try:
                            webbrowser.open(f"file://{os.path.abspath(raw_file_path)}")
                            st.success(f"Opened {os.path.basename(raw_file_path)}")
                        except Exception as e:
                            st.error(f"Failed to open document: {str(e)}")
                else:
                    st.warning("⚠ Original document not found.")
            else:
                st.info("No relevant document found.")

            # AI Response
            if st.session_state.doc_type:
                st.markdown(f"**Generated {st.session_state.doc_type.capitalize()}:**")
            else:
                st.markdown("**AI Response:**")
            st.write(st.session_state.response)

    else:
        st.info("ℹ️ Enter a query and click a button to see results here.")

    st.markdown("---")


