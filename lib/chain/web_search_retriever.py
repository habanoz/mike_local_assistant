import asyncio
import re
from typing import List

import aiohttp
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import TimeoutException
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

timeout: float = 0.8
max_results: int = 10
min_length_threshold: int = 200


class WebSearchRetriever(BaseRetriever):
    async def _fetch(self, session, doc):
        timeout = 0.5
        try:
            async with session.get(doc['href'], timeout=timeout) as response:
                text = await response.text()

                soup = BeautifulSoup(text, features="lxml")
                [s.extract() for s in soup(['style', 'script', '[document]', 'head', 'title'])]

                visible_text = soup.getText().strip()
                cleared_visible_text = await self.remove_excessive_white_spaces(visible_text)

                return doc | {"text": cleared_visible_text}
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

        docs_to_fetch = [{'title': result['title'], 'href': result['href']} for result in search_results]

        fetched_docs = asyncio.run(self.fetch_gather(docs_to_fetch))
        fetched_docs = [doc for doc in fetched_docs if self.is_valid_doc(doc)]

        return [Document(page_content=doc['text'], metadata={'url': doc['href']}) for doc in fetched_docs]

    def is_valid_doc(self, doc):
        if 'text' not in doc:
            return False

        text = doc['text']

        if text is None:
            return False

        if len(text) < min_length_threshold:
            return False

        return True


if __name__ == '__main__':
    docs = WebSearchRetriever().search("who is huseyin abanoz")
    for doc in docs:
        print(doc)
