import logging
import random

import streamlit as st

logger = logging.getLogger(__name__)


def main():
    st.set_page_config(page_title="Home", page_icon="🏠")

    i = random.Random().randint(0, 1)
    st.markdown(
        '<img src="./app/static/assistant-' + str(i) + '.png" height="512" style="border: 5px solid white">',
        unsafe_allow_html=True,
    )


def show_sidebar():
    st.sidebar.markdown('''
        <style>
            .spacer {
                display: flex;
                flex-direction: column;
                height: 50vh;  # Adjust the height percentage as needed
            }
        </style>
        <div class="spacer"></div>
    ''', unsafe_allow_html=True)

    show_trademark()


def show_trademark():
    st.sidebar.markdown("Assistant is delivered by `habanoz.io`")


if __name__ == "__main__":
    main()
    show_sidebar()
