from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate, )
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough, RunnableBranch)

from lib.chain.code_chain import build_coding_chain
from lib.chain.prompt_registry import PromptRegistry
from lib.chain.utils import push_files_out, format_docs
from lib.chain.advanced_web_search_retriever import WebSearchRetriever
from lib.ingest.kembeddings import KEmbeddings
from lib.llm.kllm import Kllm
from lib.st.cached import user_file_embeddings
from lib.utils.chain_output_sink import ChainOutputSink


def get_grounded_answer_chain(registry: PromptRegistry, chain_sink, kllm):
    answer_system = registry.prompts['answer_system']
    answer_initial_ai = registry.prompts['answer_initial_ai']
    answer_grounded_system = registry.prompts['answer_grounded_system']

    grounded_answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", answer_system + "\n" + answer_grounded_system),
            ("ai", answer_initial_ai),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
    grounded_response = (
            grounded_answer_prompt
            | kllm.get_answer_llm_runnable("grounded_answer", chain_sink)
            | StrOutputParser()
    ).with_config(run_name="GroundedResponse")
    return grounded_response


def get_ungrounded_answer_chain(registry: PromptRegistry, chain_sink: ChainOutputSink, kllm: Kllm):
    answer_system = registry.prompts['answer_system']
    answer_initial_ai = registry.prompts['answer_initial_ai']

    ungrounded_answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", answer_system),
            ("ai", answer_initial_ai),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
    ungrounded_response = (
            ungrounded_answer_prompt
            | kllm.get_answer_llm_runnable("ungrounded_answer", chain_sink)
            | StrOutputParser()
    ).with_config(run_name="UngroundedResponse")
    return ungrounded_response


def get_answer_chain(chain_sink: ChainOutputSink, kllm: Kllm, registry: PromptRegistry, user_file_retriever):
    user_file_retrieval_chain = (
        (
                RunnableLambda(itemgetter("rephrased_question"))
                | user_file_retriever.with_config(run_name="UserFileRetriever")
        )
        .with_config(run_name="UserFileRetrievalChain")
    )

    web_search_chain = (
        (
                RunnableLambda(itemgetter("rephrased_question"))
                | WebSearchRetriever(embeddings=user_file_embeddings()).with_config(run_name="WebSearchRetriever")
        )
        .with_config(run_name="WebSearchChain")
    )

    grounded_answer = get_grounded_answer_chain(registry, chain_sink, kllm)

    ungrounded_answer = get_ungrounded_answer_chain(registry, chain_sink, kllm)

    return RunnableBranch(
        (lambda x: x['next_action'] == "db_lookup", (
                RunnablePassthrough.assign(docs=user_file_retrieval_chain)
                .assign(context=lambda x: format_docs(x["docs"]))
                | RunnableLambda(lambda x: push_files_out(x, chain_sink))
                | grounded_answer
        )),
        (lambda x: x['next_action'] == "web_search",
         (
                 RunnablePassthrough.assign(docs=web_search_chain)
                 .assign(context=lambda x: format_docs(x["docs"]))
                 | grounded_answer
         )),
        (lambda x: x['next_action'] == "code_assistant", build_coding_chain(kllm, registry, chain_sink)),
        ungrounded_answer  # default path
    )
