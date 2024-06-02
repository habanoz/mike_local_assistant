import asyncio
import re
from typing import List

import aiohttp
import html2text
from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import TimeoutException
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_text_splitters import MarkdownHeaderTextSplitter

from lib.ingest.kembeddings import KEmbeddings
from lib.ingest.kvectorstore import KFaissTemporaryVectorStore

timeout: float = 1.99
max_results: int = 10
min_length_threshold: int = 200

headers_to_split_on = [
    ("#", "H1"),
    ("##", "H2"),
    ("###", "H3"),
    ("####", "H4"),
]

header_types = [header[1] for header in headers_to_split_on]


def doc_with_title(doc: Document) -> Document:
    header = resolve_title(doc)
    return Document(page_content=f"{header}\n{doc.page_content}", metadata=doc.metadata)


def resolve_title(doc):
    header_parts = [doc.metadata[type] for type in header_types if type in doc.metadata]
    header = "# " + " - ".join(header_parts)
    return header


def get_fetched_splits_in_original_order(list_of_split_docs, fetched_split_docs):
    fetched_split_id_set = {doc.metadata['split_id'] for doc in fetched_split_docs}

    list_of_selected_split_docs = []
    for split_docs in list_of_split_docs:
        selected_splits = [doc for doc in split_docs if doc.metadata['split_id'] in fetched_split_id_set]
        if selected_splits:
            list_of_selected_split_docs.append(selected_splits)
    return list_of_selected_split_docs


def split_docs_to_md_text(title: str, split_docs: List[Document]) -> str:
    split_docs_with_header = ["#" + resolve_title(doc) + "\n" + doc.page_content for doc in split_docs]
    return f"# {title}\n" + "\n".join(split_docs_with_header)


class WebSearchRetriever(BaseRetriever):
    embeddings: KEmbeddings

    async def _fetch(self, session, doc):
        try:
            async with session.get(doc['href'], timeout=timeout) as response:
                text = await response.text()

                #md_text = html2text.html2text(text)

                h2t = html2text.HTML2Text(bodywidth=0)
                h2t.ignore_links = True
                h2t.ignore_images = True
                md_text = h2t.handle(text)

                return doc | {"text": md_text}
        except asyncio.TimeoutError:
            print(f"Timeout > {timeout}s for URL: {doc['href']}")
            return doc | {"text": None}
        except Exception as e:
            print(f"{e} for URL: {doc['href']}")
            return doc | {"text": None}

    async def remove_excessive_white_spaces(self, visible_text):
        # remove leading spaces from each line
        visible_text = re.sub(r'^[^\S\n]+', '', visible_text, flags=re.MULTILINE)

        # remove excessive new lines e.g. more than 3 new lines.
        return re.sub('\n{4,}', '\n\n\n', visible_text)

    async def fetch_gather(self, docs):
        async with aiohttp.ClientSession() as session:
            tasks = [self._fetch(session, doc) for doc in docs]
            return await asyncio.gather(*tasks, return_exceptions=True)

    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        ddgs = DDGS()

        try:
            search_results = ddgs.text(query, max_results=max_results)
        except TimeoutException as e:
            print("DDGS Timeout :", str(e))
            return []

        resources_to_fetch = [{'title': result['title'], 'href': result['href']} for result in search_results]

        all_fetched_resources = asyncio.run(self.fetch_gather(resources_to_fetch))
        fetched_resources = [res for res in all_fetched_resources if self.is_valid_resource(res)]
        fetched_docs = [self.resource_to_doc(res) for res in fetched_resources]

        list_of_split_docs = self.split_md_docs(fetched_docs)
        list_of_relevant_split_docs = self._fetch_relevant(query, list_of_split_docs)

        md_docs = [Document(page_content=split_docs_to_md_text(split_docs[0].metadata['title'], split_docs),
                            metadata=split_docs[0].metadata) for
                   split_docs in list_of_relevant_split_docs]
        return md_docs

    def _fetch_relevant(self, query: str, list_of_split_docs: List[List[Document]]):
        vector_store = KFaissTemporaryVectorStore(self.embeddings)

        for splits_docs in list_of_split_docs:
            splits_docs_with_title = [doc_with_title(doc) for doc in splits_docs]
            vector_store.add_documents(splits_docs_with_title)

        fetched_split_docs = vector_store.as_retriever(20, 0.3).invoke(query)

        list_of_selected_split_docs = get_fetched_splits_in_original_order(list_of_split_docs, fetched_split_docs)

        return list_of_selected_split_docs

    def resource_to_doc(self, res: dict):
        return Document(page_content=res['text'], metadata={'url': res['href'], 'title': res['title']})

    def split_md_docs(self, docs: list[Document]):
        return [self.split_md_doc(doc) for doc in docs]

    def split_md_doc(self, doc: Document):

        md = doc.page_content

        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        md_header_splits = markdown_splitter.split_text(md)

        split_docs = []
        for i, split in enumerate(md_header_splits):
            split_docs.append(Document(page_content=split.page_content, metadata=split.metadata | doc.metadata | {
                'split_id': f"{doc.metadata['url']}-{i}"}))

        return split_docs

    def is_valid_resource(self, resource: dict):
        if 'text' not in resource:
            return False

        text = resource['text']

        if text is None:
            return False

        if len(text) < min_length_threshold:
            return False

        return True


def main():
    config = {"provider": "ollama",
              "model": "nomic-embed-text",
              "dimensions": 768,
              "base_url": "http://192.168.1.107:11434"
              }
    embeddings = KEmbeddings(config)
    docs = WebSearchRetriever(embeddings=embeddings).invoke("bbc trump conviction")
    for doc in docs:
        print(doc)


if __name__ == '__main__':
    main()
