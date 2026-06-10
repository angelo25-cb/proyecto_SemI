import streamlit as st
from pathlib import Path

def load_css():

    css_file = Path(__file__).parent / "style.css"

    with open(
        css_file,
        encoding="utf-8"
    ) as f:

        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )