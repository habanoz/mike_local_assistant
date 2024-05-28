from io import BytesIO
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters.html import HTMLHeaderTextSplitter
import pdfplumber


class UserFileIndexBuilder:
    def __init__(self, user_file_vector_store):
        self.vector_store = user_file_vector_store

        self._add_start_index = True
        self._chunk_overlap = 0

    def create_index_user_file(self, file: BytesIO) -> (List[str], List[Document]):
        if file.name.lower().endswith("md"):
            return self.create_index_user_file_md(file)
        elif file.name.lower().endswith("txt"):
            return self.create_index_user_file_txt(file)
        elif file.name.lower().endswith("pdf"):
            return self.create_index_user_file_pdf(file)
        elif file.name.lower().endswith("html") or file.name.lower().endswith("htm"):
            return self.create_index_user_file_html(file)
        else:
            raise Exception("Unsupported file type:" + str(file.name))

    def create_index_user_file_pdf(self, file: BytesIO) -> (List[str], List[Document]):
        file_name = file.name
        pdf = pdfplumber.open(file)

        ids, docs = [], []
        for page in pdf.pages:
            page_text = page.extract_text()
            page_ids, page_docs = self._create_index_user_file_txt_segment(page_text, file_name)

            ids.extend(page_ids)
            docs.extend(page_docs)

        return ids, docs

    def create_index_user_file_html(self, file: BytesIO) -> (List[str], List[Document]):
        file_name = file.name
        content = file.getvalue().decode("utf-8", errors="replace")

        headers_to_split_on = [
            ("h1", "Header 1"),
            ("h2", "Header 2"),
            ("h3", "Header 3"),
            ("h4", "Header 4"),
        ]

        html_splitter = HTMLHeaderTextSplitter(headers_to_split_on)

        html_header_splits = html_splitter.split_text(content)

        chunk_size = 400
        chunk_overlap = 30
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        docs = text_splitter.split_documents(html_header_splits)
        docs = [Document(page_content=doc.page_content, metadata={"file_name": file_name}) for doc in docs]

        return self.vector_store.add_documents(docs), docs

    def create_index_user_file_txt(self, file: BytesIO) -> (List[str], List[Document]):
        file_name = file.name
        content = file.getvalue().decode("utf-8", errors="replace")

        return self._create_index_user_file_txt_segment(content, file_name)

    def _create_index_user_file_txt_segment(self, text, file_name) -> (List[str], List[Document]):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False,
        )
        docs = text_splitter.create_documents([text])
        docs = [Document(page_content=doc.page_content, metadata={"file_name": file_name}) for doc in docs]

        return self.vector_store.add_documents(docs), docs

    def create_index_user_file_md(self, file: BytesIO) -> (List[str], List[Document]):
        headers_to_split_on = [
            ("#", "H1"),
            ("##", "H2"),
            ("###", "H3"),
            ("####", "H4"),
        ]

        content = file.getvalue().decode("utf-8", errors="replace")

        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        md_header_splits = markdown_splitter.split_text(content)

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
