from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from lib.db.model.user_files import UserFile
from lib.st.cached import user_file_vector_store
from langchain_community.document_loaders import PyPDFLoader


class UserFileIndexBuilder:
    def __init__(self):
        self.vector_store = user_file_vector_store()

        self._add_start_index = True
        self._chunk_overlap = 0

    def create_index_user_file(self, file: UserFile) -> (List[str], List[Document]):
        if file.name.lower().endswith("md"):
            return self.create_index_user_file_md(file)

    def create_index_user_file_pdf(self, file: UserFile) -> (List[str], List[Document]):
        loader = PyPDFLoader(file.name)
        pages = loader.load_and_split()

    def create_index_user_file_txt(self, file: UserFile) -> (List[str], List[Document]):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False,
        )

        docs = text_splitter.create_documents([file.content])
        docs = [Document(page_content=doc.page_content, metadata={"file_name": file.name}) for doc in docs]

        return self.vector_store.add_documents(docs), docs

    def create_index_user_file_md(self, file: UserFile) -> (List[str], List[Document]):
        headers_to_split_on = [
            ("#", "H1"),
            ("##", "H2"),
            ("###", "H3"),
            ("####", "H4"),
        ]

        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        md_header_splits = markdown_splitter.split_text(file.content)

        def to_title(metadata):
            headers = [v for k, v in metadata.items() if k in ["H1", "H2", "H3", "H4"]]
            return "# " + " - ".join(headers)

        docs = [
            Document(
                page_content=to_title(split.metadata) + "\n" + split.page_content,
                metadata={"file_name": file.name})
            for split in md_header_splits
        ]

        return self.vector_store.add_documents(docs), docs
