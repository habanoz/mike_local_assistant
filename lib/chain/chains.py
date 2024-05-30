from collections.abc import Callable
from operator import itemgetter
from typing import Sequence, List

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    MessagesPlaceholder, ChatPromptTemplate, )
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnablePassthrough, RunnableBranch)
from pydantic import BaseModel, Field

from lib.chain.prompt_registry import PromptRegistry
from lib.chain.web_search_retriever import WebSearchRetriever
from lib.db.model import UserFile
from lib.ingest.kvectorstore import KVectorStore
from lib.llm.kllm import Kllm
from lib.utils.chain_output_sink import ChainOutputSink

wrapper = DuckDuckGoSearchAPIWrapper(max_results=5)
search = DuckDuckGoSearchResults(api_wrapper=wrapper)


def push_files_out(x, chain_sink: ChainOutputSink):
    if "docs" in x:
        chain_sink.add_debug(name="files", content=[{
            "content": doc.page_content,
            "metadata": {k: str(v)[:100] for k, v in doc.metadata.items()}}
            for doc in x["docs"]])

    if "docs" in x:
        for i, doc in enumerate(x["docs"], start=1):
            chain_sink.add_file(
                name=str(i),
                file=doc.metadata['file_name'] if "file_name" in doc.metadata else None,
                content=doc.page_content)

    return x


def push_prompts_out(prompt, prompt_name, chain_sink: ChainOutputSink):
    chain_sink.add_debug(
        name="prompt_" + prompt_name,
        content=[{"role": msg.type, "content": msg.content} for msg in prompt.messages]
    )
    return prompt


def format_docs(docs: Sequence[Document]) -> str:
    formatted_docs = []
    for i, doc in enumerate(docs, start=1):
        doc_string = f"<doc id='{i}'>{doc.page_content}</doc>"
        formatted_docs.append(doc_string)
    return "\n".join(formatted_docs)


def format_session_user_files(files: List[UserFile]):
    return "\n".join([f"- Document {i} : {file.summary}" for i, file in enumerate(files, start=2)])


def get_next_action_chain(kllm: Kllm, registry: PromptRegistry, session_files: List[UserFile],
                          chain_sink: ChainOutputSink):
    select_next_action = registry.prompts['select_next_action']

    prompt = ChatPromptTemplate.from_template(select_next_action)

    next_action_chain = (
        (
                RunnablePassthrough.assign(uploaded_file_summaries=lambda x: format_session_user_files(session_files))
                | RunnablePassthrough.assign(question=itemgetter("rephrased_question"))
                | prompt
                | kllm.get_deterministic_llm_runnable("next_action", chain_sink)
                | StrOutputParser()
        )
        .with_config(run_name="next_action")
    )
    return next_action_chain


def build_coding_chain(kllm: Kllm, registry: PromptRegistry, chain_sink: ChainOutputSink):
    code_assistant_prompt = registry.prompts['coding_assistant_system']
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", code_assistant_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
    next_action_chain = (
        (
                prompt
                | kllm.get_code_llm_runnable("code_assistant", chain_sink)
                | StrOutputParser()
        )
        .with_config(run_name="code_assistant")
    )
    return next_action_chain


def create_chain(kllm: Kllm, user_file_retriever: BaseRetriever, registry: PromptRegistry,
                 session_files: List[UserFile], chain_sink: ChainOutputSink) -> Runnable:
    standalone_question = registry.prompts['standalone_question']
    generate_assistant_answer_main_system = registry.prompts['generate_assistant_answer_main_system']
    generate_assistant_answer_initial_ai = registry.prompts['generate_assistant_answer_initial_ai']
    generate_assistant_answer_documents_system = registry.prompts['generate_assistant_answer_documents_system']

    standalone_question_chain = (
        (
                ChatPromptTemplate.from_template(standalone_question)
                | kllm.get_deterministic_llm_runnable("standalone_question", chain_sink)
                | StrOutputParser()
        )
        .with_config(run_name="StandaloneQuestion")
    )

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
                | WebSearchRetriever().with_config(run_name="WebSearchRetriever")
        )
        .with_config(run_name="WebSearchChain")
    )

    grounded_answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", generate_assistant_answer_main_system + "\n" + generate_assistant_answer_documents_system),
            ("ai", generate_assistant_answer_initial_ai),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    ungrounded_answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", generate_assistant_answer_main_system),
            ("ai", generate_assistant_answer_initial_ai),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    grounded_response = (
            grounded_answer_prompt
            | kllm.get_answer_llm_runnable("grounded_answer", chain_sink)
            | StrOutputParser()
    ).with_config(run_name="GroundedResponse")

    ungrounded_response = (
            ungrounded_answer_prompt
            | kllm.get_answer_llm_runnable("ungrounded_answer", chain_sink)
            | StrOutputParser()
    ).with_config(run_name="UngroundedResponse")

    generate_answer_chain = RunnableBranch(
        (lambda x: x['next_action'] == "db_lookup", (
                RunnablePassthrough.assign(docs=user_file_retrieval_chain)
                .assign(context=lambda x: format_docs(x["docs"]))
                | RunnableLambda(lambda x: push_files_out(x, chain_sink))
                | grounded_response
        )),
        (lambda x: x['next_action'] == "web_search",
         (
                 RunnablePassthrough.assign(docs=web_search_chain)
                 .assign(context=lambda x: format_docs(x["docs"]))
                 | grounded_response
         )),
        (lambda x: x['next_action'] == "code_assistant", build_coding_chain(kllm, registry, chain_sink)),
        ungrounded_response  # default path
    )

    return (
            RunnablePassthrough.assign(chat_history_str=lambda x: format_chat_history(x["chat_history"]))
            | RunnablePassthrough.assign(rephrased_question=standalone_question_chain)
            | RunnablePassthrough.assign(next_action=get_next_action_chain(kllm, registry, session_files, chain_sink))
            | generate_answer_chain
    ).with_config(run_name="Assistant")


def format_chat_history(chat_history: list):
    if chat_history and isinstance(chat_history[0], dict):
        return "\n".join([f"- {turn['role']}: {turn['content']}" for turn in chat_history])
    return "\n".join([f"- {turn.type}: {turn.content}" for turn in chat_history])


def get_chain(config, registry: PromptRegistry, session_files: List[UserFile], user_file_vector_store: KVectorStore,
              chain_sink: ChainOutputSink) -> Callable:
    user_file_retriever = user_file_vector_store.get_user_file_retriever()

    kllm = Kllm(config['llms'])

    chain = create_chain(kllm, user_file_retriever, registry, session_files, chain_sink)

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
