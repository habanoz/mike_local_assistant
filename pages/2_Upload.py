import uuid
from typing import List

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from Home import show_sidebar
from lib.chain.summary_chain import summarize
from lib.db.model import FileChunk
from lib.db.model.user_files import UserFile
from lib.ingest.user_file_index_builders import UserFileIndexBuilder
from lib.service.file_chunk_service import FileChunkService
from lib.st.session_service import SessionService
from lib.service.user_file_service import UserFileService
from lib.st.cached import db_manager, user_file_vector_store


def is_duplicate(file):
    for user_file in SessionService.get_session_files():
        if user_file.name == file.name:
            return True
    return False


def show():
    uploaded_files: List[UploadedFile] = st.file_uploader(
        "Upload your files.", accept_multiple_files=True,
        type=['md', 'pdf', "txt", "html"]
    )

    if uploaded_files:
        files_to_add = []

        for file in uploaded_files:
            if is_duplicate(file):
                st.error(f"Duplicate files are not allowed: {file.name}. "
                         "To re-upload a file try removing the file first.")
                files_to_add = []
                break
            else:
                files_to_add.append(file)

        if files_to_add:
            builder = UserFileIndexBuilder(user_file_vector_store())

            file_service = UserFileService(db_manager())
            chunk_service = FileChunkService(db_manager())

            progress_text = "Processing files. Please wait."
            file_progress_bar = st.progress(0, text=progress_text)

            for i, file in enumerate(files_to_add):
                content = file.getvalue().decode("utf-8", errors="replace")
                summary = summarize(content)

                user_file = UserFile(
                    id=uuid.uuid4(), name=file.name,
                    content=content,
                    summary=summary
                )

                doc_ids, docs = builder.create_index_user_file(file)
                chunk_service.add_all(
                    [FileChunk(id=uuid.uuid4(), name=file.name, idx=idx, chunk_id=doc_id,
                               content=docs[idx].page_content if docs else None)
                     for idx, doc_id in enumerate(doc_ids)]
                )

                user_file = file_service.save(user_file)

                SessionService.add_to_session_file(user_file)

                file_progress_bar.progress((i + 1) / len(files_to_add), text=progress_text)

            st.switch_page("pages/1_Files.py")


if __name__ == '__main__':
    st.set_page_config(page_title="Upload", page_icon="⬆️")

    st.title('Upload Files')
    st.write(
        'Uploaded files will be used to answer user questions. Make sure files do not contain personal information. '
        'Use `files` page to see list of uploaded files.')
    st.write("Markdown is recommended. Other methods are experimental!")

    show()
    show_sidebar()
