import uuid
from collections.abc import Callable

import streamlit as st

from Home import show_sidebar
from lib.chain.chains import get_chain
from lib.service.chat_history_service import ChatHistoryService
from lib.st.cached import db_manager, config, prompts_registry


@st.cache_resource
def chat_history_service():
    return ChatHistoryService(db_manager())


def get_source_files():
    if "source_files" not in st.session_state:
        st.session_state["source_files"] = []

    return st.session_state["source_files"]


def get_session_chain() -> Callable:
    if "session_chain" not in st.session_state:
        with st.spinner("Building..."):
            print("Building chat session...")
            st.session_state["session_chain"] = get_chain(config(), prompts_registry(), get_source_files())

    return st.session_state["session_chain"]


@st.experimental_dialog("File")
def display_file(name, content, file):
    st.title(f"File {name}")
    if file: st.write("**File Name**: " + str(file))
    st.write(content)


@st.experimental_dialog("Please wait...")
def wait_failure():
    st.markdown(
        "Due to limitations of the platform, engaging with UI buttons"
        " while generating an answer cancels current request. ")
    st.markdown("Thanks for your understanding.")
    st.markdown("Please refresh the page to start again!")

    if st.button("Close"):
        st.rerun()


def main():
    if "busy" not in st.session_state:
        st.session_state["busy"] = False

    st.markdown(f"### 💬 Chat with your assistant!")

    if "messages" not in st.session_state:
        ui_introduction_message = prompts_registry().instructions["ui_introduction_message"]
        st.session_state["messages"] = [
            {"role": "ai", "content": ui_introduction_message}]
        st.session_state["busy"] = False

    for mi, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            msg_content = message["content"]
            msg_content = msg_content.replace("[[", "`[").replace("]]", "]`")
            st.markdown(msg_content)

            if "files" in message:
                files = message["files"]
                intents = [file for file in files if file['name'] in {"rephrased_question"}]
                quotes = [file for file in files if file['name'] not in {"rephrased_question"}]

                if intents: horizontal_file_buttons(intents, 3, "intents", mi)
                if quotes: horizontal_file_buttons(quotes, 15, "files", mi)

    if prompt := st.chat_input():

        st.session_state["busy"] = True

        if "chat_id" not in st.session_state:
            _chat_id = uuid.uuid4()
            st.session_state["chat_id"] = _chat_id
            chat_history_service().add_chat(_chat_id)

        chat_history_service().add_utterance(st.session_state["chat_id"], "human", prompt)
        st.session_state.messages.append({"role": "human", "content": prompt})

        with st.chat_message("human"):
            st.markdown(prompt)

        with st.chat_message('ai'):
            messages = st.session_state.messages
            question = messages[-1]['content']
            chat_history = messages[1:-1]

            answer_chain = get_session_chain()(question, chat_history)

            with st.spinner("Generating response..."):
                first_token = next(answer_chain)

            def generator(initial, answer_generator):
                yield initial
                for chunk in answer_generator:
                    yield chunk

            result = st.write_stream(generator(first_token, answer_chain))

            msg = result
            if isinstance(result, dict):
                if "output" in result:
                    msg = result["output"]
                elif "content" in result:
                    msg = result["content"]
                else:
                    msg = str(result)
            files = get_source_files()
            files_to_keep = []
            for file in files:
                name = file["name"]
                if name in {"rephrased_question"} or f"[[{name}]]" in msg:
                    files_to_keep.append(file)

            chat_history_service().add_utterance(st.session_state["chat_id"], "ai", msg)
            st.session_state.messages.append({"role": "ai", "content": msg, "files": files_to_keep})

            st.session_state["busy"] = False
            st.rerun()


def horizontal_file_buttons(files, max_cols, parent_name, parent_index):
    cols = st.columns(max_cols)
    for i in range(len(cols)):
        with cols[i]:
            if len(files) > i:
                file = files[i]
                if st.button(file['name'], key=parent_name + str(parent_index) + str(i),
                             disabled=st.session_state["busy"]):
                    if st.session_state["busy"]:
                        wait_failure()
                    else:
                        display_file(file['name'], file['content'], file['file'] if 'file' in file else None)
            else:
                st.empty()


if __name__ == '__main__':
    st.set_page_config(page_title="Chat", page_icon="💬")

    with st.sidebar:
        "Chat with your AI assistant!"
        "AI assistant makes use of Llama 3 family of models in its pipeline."

    show_sidebar()

    main()
