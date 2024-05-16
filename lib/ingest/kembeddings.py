from typing import List, Iterable

from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_core.embeddings import Embeddings


def ollama_embeddings(config) -> (Embeddings, int):
    from langchain_community.embeddings import OllamaEmbeddings

    model = config['model']
    dimensions = config['dimensions']
    base_url = config['base_url']

    return NormalizedEmbeddings(OllamaEmbeddings(model=model, base_url=base_url)), dimensions


def get_file_cached_embeddings(task, provider, model, backing_embeddings) -> Embeddings:
    store = LocalFileStore(f"data/embeddings/{task}/{provider}/{model}-")
    return CacheBackedEmbeddings.from_bytes_store(backing_embeddings, store, namespace=model)


class KEmbeddings(Embeddings):
    def __init__(self, task: str, config):
        provider = config['provider']
        if provider.lower() == "ollama":
            self.embeddings, self.dims = ollama_embeddings(config[provider])
        else:
            raise Exception("Unknown embeddings!:" + str(provider))

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.embeddings.embed_query(text)


class NormalizedEmbeddings(Embeddings):
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [norm_embed(emb) for emb in self.embeddings.embed_documents(texts)]

    def embed_query(self, text: str) -> List[float]:
        return norm_embed(self.embeddings.embed_query(text))


def norm_embed(emb: Iterable[float]):
    import numpy as np
    arr_emb = np.array(emb)
    magnitude = np.linalg.norm(arr_emb)
    return arr_emb / magnitude
