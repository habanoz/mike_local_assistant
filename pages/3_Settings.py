import streamlit as st

from Home import show_sidebar

if __name__ == '__main__':
    st.set_page_config(page_title="Settings", page_icon="⚙️")

    st.write("No settings here")

    show_sidebar()