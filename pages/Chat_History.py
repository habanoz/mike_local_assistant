import uuid

import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from lib.db.model import Utterance
from lib.llm.kllm import Kllm
from lib.service.chat_history_service import ChatHistoryService
from lib.st.cached import db_manager, config


@st.cache_resource
def chat_history_service():
    return ChatHistoryService(db_manager())


@st.experimental_dialog("File")
def display_file(name, content, file):
    st.title(f"File {name}")
    if file: st.write("**File Name**: " + str(file))
    st.write(content)


@st.experimental_dialog("Debug Info", width="large")
def show_debug(debug):
    tabs = st.tabs([d['name'].replace("prompt", "prmp")[:11] for d in debug])
    for tab, d in zip(tabs, debug):
        with tab:
            st.subheader(d['name'])
            st.write(d['content'])


def validate_chat_id(chat_id):
    try:
        uuid.UUID(chat_id)
        return True
    except:
        return False


def show_no_owner():
    st.title(f"😿 No one here to chat!")
    st.write("You need to follow a public Steward link to chat with Steward!")
    st.write("Steward links are typically shared publicly in social media profile pages e.g. linkedin, "
             "twitter, instagram.")
    pass


def main():
    if "historical_chat_id" not in st.session_state:
        st.session_state["historical_chat_id"] = ""

    if "utterances" not in st.session_state:
        st.session_state["utterances"] = []

    chat_id_str = st.text_input("Chat ID", st.session_state["historical_chat_id"])

    if st.button("Fetch"):
        if not chat_id_str:
            st.warning("Please specify your email address!")
        elif not validate_chat_id(chat_id_str):
            st.warning("Please specify a valid chat id!")
        else:
            with st.spinner("Fetching..."):
                try:
                    utterances = chat_history_service().fetch_by_chat_id(uuid.UUID(chat_id_str))
                    st.session_state["utterances"] = utterances
                    st.session_state["historical_chat_id"] = chat_id_str

                    if not utterances:
                        st.warning("No history found!")
                except Exception as e:
                    print(e)
                    st.warning("Fetching failed with exception:" + str(e))

    for mi, utterance in enumerate(st.session_state["utterances"]):
        with st.chat_message(utterance.type):
            msg_content = utterance.message
            msg_content = transform_quotes(msg_content)

            if utterance.alternatives:
                names = [alt['name'] for alt in utterance.alternatives]
                tabs = st.tabs(["main"] + names)
                with tabs[0]:
                    st.markdown(msg_content)
                for tab, alternative in zip(tabs[1:], utterance.alternatives):
                    with tab:
                        alt_content = dequote(transform_quotes(alternative['content']))
                        st.markdown(alt_content)
            else:
                st.markdown(msg_content)

            show_buttons(utterance, 10, "files", mi)


def transform_quotes(msg_content):
    return msg_content.replace("[[", "`[").replace("]]", "]`").replace("\\n", "\n")


def show_buttons(utterance, max_cols, parent_name, parent_index):
    cols = st.columns(max_cols)
    if utterance.files:
        for i in range(len(cols) - 1):
            with cols[i]:
                if len(utterance.files) > i:
                    file = utterance.files[i]
                    if st.button(file['name'], key=parent_name + str(parent_index) + str(i)):
                        display_file(file['name'], file['content'], file['file'] if 'file' in file else None)
                else:
                    st.empty()

    if utterance.debug:
        with cols[-1]:
            if st.button("🪲", key=parent_name + str(parent_index) + str("d")):
                show_debug(utterance.debug)
        with cols[-2]:
            if st.button("♻️", key=parent_name + str(parent_index) + str("r")):
                re_generate(utterance)


@st.experimental_dialog("Re-generate")
def re_generate(utterance: Utterance):
    items: list[dict] = utterance.debug
    answer_prompt = next(filter(lambda item: 'prompt' in item['name'] and 'answer' in item['name'], items), None)

    messages = answer_prompt['content']
    prompt = ChatPromptTemplate.from_messages([(msg['role'], dequote(msg['content'])) for msg in messages])

    model_name, model = get_re_generate_model()
    if model:
        answer_stream = (prompt | model | StrOutputParser()).stream({})
        result = st.write_stream(answer_stream)

        if utterance.alternatives is None:
            utterance.alternatives = []

        utterance.alternatives.append({'name': model_name, 'content': dequote(result)})
        chat_history_service().update_utterance_alternatives(utterance)


def dequote(s):
    s = s.replace("\\n", "\n")
    if (len(s) >= 2 and s[0] == s[-1]) and s.startswith(("'", '"')):
        return s[1:-1]
    return s


def get_re_generate_model():
    model_names = "llama3-8b"
    model_name = st.selectbox("Select a model", [model_names])
    if st.button("Re-generate"):
        if model_name == "llama3-8b":
            return model_name, Kllm(config()['llms']).get_answer_llm()
    return None, None


if __name__ == '__main__':
    st.set_page_config(page_title="History", page_icon="📜")

    with st.sidebar:
        "Chat History!"

        for chat in chat_history_service().fetch_recent_chats():
            if st.button(str(chat.id) + " " + str(chat.created.strftime("%m-%d %H:%M")),
                         key="link_button" + str(chat.id)):
                st.session_state["historical_chat_id"] = str(chat.id)
                st.rerun()

    main()
