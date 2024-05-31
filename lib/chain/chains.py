from collections.abc import Callable
from typing import List

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate, )
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import (
    RunnablePassthrough, Runnable)

from lib.chain.answer_chain import get_answer_chain
from lib.chain.next_action_chain import get_next_action_chain
from lib.chain.prompt_registry import PromptRegistry
from lib.chain.utils import format_chat_history
from lib.db.model import UserFile
from lib.ingest.kvectorstore import KVectorStore
from lib.llm.kllm import Kllm
from lib.utils.chain_output_sink import ChainOutputSink

wrapper = DuckDuckGoSearchAPIWrapper(max_results=5)
search = DuckDuckGoSearchResults(api_wrapper=wrapper)


def get_standalone_question_chain(chain_sink, kllm, standalone_question):
    standalone_question_chain = (
        (
                ChatPromptTemplate.from_template(standalone_question)
                | kllm.get_deterministic_llm_runnable("standalone_question", chain_sink)
                | StrOutputParser()
        )
        .with_config(run_name="StandaloneQuestion")
    )
    return standalone_question_chain


def build_main_chain(kllm: Kllm, user_file_retriever: BaseRetriever, registry: PromptRegistry,
                     session_files: List[UserFile], chain_sink: ChainOutputSink
                     ) -> Runnable:
    standalone_question = registry.prompts['standalone_question']
    standalone_question_chain = get_standalone_question_chain(chain_sink, kllm, standalone_question)

    answer_chain = get_answer_chain(chain_sink, kllm, registry, user_file_retriever)

    return (
            RunnablePassthrough.assign(chat_history_str=lambda x: format_chat_history(x["chat_history"]))
            | RunnablePassthrough.assign(rephrased_question=standalone_question_chain)
            | RunnablePassthrough.assign(next_action=get_next_action_chain(kllm, registry, session_files, chain_sink))
            | answer_chain
    ).with_config(run_name="Assistant")


def get_main_chain_stream(config, registry: PromptRegistry, session_files: List[UserFile], user_file_vector_store: KVectorStore,
                          chain_sink: ChainOutputSink) -> Callable:
    user_file_retriever = user_file_vector_store.get_user_file_retriever()

    kllm = Kllm(config['llms'])

    chain = build_main_chain(kllm, user_file_retriever, registry, session_files, chain_sink)

    def add_message(question, chat_history):
        chat_history = chat_history[:10]
        answer = ""
        for chunk in chain.stream(
                {"question": question, "chat_history": chat_history}
        ):
            answer += chunk
            yield chunk
            chat_history.extend([HumanMessage(content=question), AIMessage(content=answer)])

    return add_message
