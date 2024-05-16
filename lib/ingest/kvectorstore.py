import os
from typing import List, Any

from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document

from lib.ingest.kembeddings import KEmbeddings


class KVectorStore:
    def __init__(self, embeddings: KEmbeddings, task, config):
        provider = config['provider']
        if provider == "faiss":
            self.vector_store = KFaissVectorStore(embeddings, task)
        else:
            raise Exception("Unknown provider:" + str(provider))

    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        return self.vector_store.add_documents(documents, **kwargs)

    def get_user_file_retriever(self, k=10):
        return self.vector_store.get_user_file_retriever(k)

    def delete(self, ids: List[str]):
        return self.vector_store.delete(ids)


class KFaissVectorStore:
    store_root = ".faiss"
    store_dir = ".faiss_{name}"

    def __init__(self, embeddings: KEmbeddings, name: str):

        from langchain_community.vectorstores.faiss import DistanceStrategy
        from langchain_community.vectorstores.faiss import FAISS
        import faiss
        import os

        self.embeddings = embeddings
        self.store_path = os.path.join(self.store_root, self.store_dir.format(name=name))

        if os.path.exists(self.store_path):
            self.vector_store = FAISS.load_local(self.store_path, self.embeddings, allow_dangerous_deserialization=True)
        else:
            self.vector_store = FAISS(
                embedding_function=self.embeddings, index=faiss.IndexFlatIP(self.embeddings.dims),
                docstore=InMemoryDocstore(), index_to_docstore_id={},
                distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
            )

            self.persist()

    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        ids = self.vector_store.add_documents(documents, **kwargs)
        self.persist()
        return ids

    def persist(self):
        if not os.path.exists(self.store_root):
            os.mkdir(self.store_root)

        self.vector_store.save_local(self.store_path)

    def get_user_file_retriever(self, k=10):
        return self.vector_store.as_retriever(
            # search_type="similarity_score_threshold",
            # search_type="mmr",
            search_kwargs={"k": k, "fetch_k": k * 5,
                           # score_threshold': 0.8,
                           # 'lambda_mult': 0.25 mmr diversity parameter
                           # "filter": {"file": str(file_name)}
                           }
        )

    def delete(self, ids: List[str]):
        self.vector_store.delete(ids)
