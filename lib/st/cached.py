import streamlit as st

from lib.db.db_manager import DatabaseManager
from lib.ingest.kembeddings import KEmbeddings
from lib.ingest.kvectorstore import KVectorStore
from lib.utils.yaml_utils import load_yaml_file


@st.cache_resource
def config():
    return load_yaml_file("config/config.yml")['config']


def is_dev():
    return 'profile' not in config() or config()['profile'] == 'dev'


@st.cache_resource
def user_file_embeddings() -> KEmbeddings:
    return KEmbeddings("user_files", config()['embeddings'])


@st.cache_resource
def user_file_vector_store() -> KVectorStore:
    print("Creating user file vector store")
    return KVectorStore(user_file_embeddings(), "user_files", config()['vector_store'])


@st.cache_resource(show_spinner=False)
def db_manager():
    with st.spinner("Initializing..."):
        print("creating DB Manager")
        connection_string = config()['db']['connection_string']
        if not connection_string:
            raise ValueError("db.connection_string not provided!")

        return DatabaseManager(database_url=connection_string)
