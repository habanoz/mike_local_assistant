import streamlit as st

from Home import show_sidebar
from lib.service.file_chunk_service import FileChunkService
from lib.service.session_service import SessionService
from lib.service.user_file_service import UserFileService
from lib.st.cached import db_manager, user_file_vector_store


def remove_file(file):
    file_chunk_service = FileChunkService(db_manager())
    chunks = file_chunk_service.find_file_chunks(file.name)
    chunk_ids = [chunk.chunk_id for chunk in chunks]

    if chunk_ids:
        vector_store = user_file_vector_store()
        vector_store.delete(chunk_ids)

        file_chunk_service.delete_by_file_name(file.name)

    user_file_service = UserFileService(db_manager())
    user_file_service.delete_by_id(file.id)

    SessionService.remove_session_file(file)

    st.rerun()


# Main app
def show():
    st.title('Files')
    st.write('Files are used to answer user questions. '
             'Remove old files using the `X` button. Add new files using `upload` page.')

    if 'removal_message' not in st.session_state:
        st.session_state['removal_message'] = ''

    user_files = SessionService.get_session_files()

    for index, user_file in enumerate(user_files):
        col1, col2 = st.columns([0.8, 0.2], gap="small")
        with col1:
            st.text(user_file.name)
        with col2:
            if st.button('❌', key=user_file.name):
                remove_file(user_file)

        if len(user_files) > 1 and index < len(user_files) - 1:
            st.markdown('<hr style="margin-top: 0px; margin-bottom: 0px;">', unsafe_allow_html=True)

    with st.container():
        if st.session_state['removal_message']:
            st.info(st.session_state['removal_message'])
            st.session_state['removal_message'] = ''


if __name__ == "__main__":
    st.set_page_config(page_title="Files", page_icon="📁")

    show()
    show_sidebar()
