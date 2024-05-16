from collections.abc import Callable
from operator import itemgetter
from typing import Sequence

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    MessagesPlaceholder, ChatPromptTemplate,
)
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnablePassthrough)

from lib.chain.prompt_registry import PromptRegistry
from lib.llm.kllm import Kllm
from lib.st.cached import user_file_vector_store, config, is_dev


def push_files_out(x, files_out: list):
    files_out.clear()

    if is_dev():
        if "rephrased_question" in x:
            files_out.append({"name": "rephrased_question", "content": x["rephrased_question"]})

    if "docs" in x:
        for i, doc in enumerate(x["docs"], start=1):
            files_out.append(
                {"name": str(i),
                 "file": doc.metadata['file_name'] if "file_name" in doc.metadata else None,
                 "content": doc.page_content})
    return x


def format_docs(docs: Sequence[Document]) -> str:
    formatted_docs = []
    for i, doc in enumerate(docs, start=1):
        doc_string = f"<doc id='{i}'>{doc.page_content}</doc>"
        formatted_docs.append(doc_string)
    return "\n".join(formatted_docs)


def create_chain(kllm: Kllm, user_file_retriever: BaseRetriever, registry: PromptRegistry, files_out: list) -> Runnable:
    standalone_question = registry.prompts['standalone_question']
    generate_assistant_answer_main_system = registry.prompts['generate_assistant_answer_main_system']
    generate_assistant_answer_initial_ai = registry.prompts['generate_assistant_answer_initial_ai']
    generate_assistant_answer_documents_system = registry.prompts['generate_assistant_answer_documents_system']

    standalone_question_chain = (
        (
                ChatPromptTemplate.from_template(standalone_question)
                | kllm.get_rephrase_llm()
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

    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", generate_assistant_answer_main_system + "\n" + generate_assistant_answer_documents_system),
            ("ai", generate_assistant_answer_initial_ai),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    answer_chain = (
            answer_prompt
            | kllm.get_answer_llm()
            | StrOutputParser()
    ).with_config(configurable={"llm_temperature": 0.4}, run_name="GenerateResponse")

    return (
            RunnablePassthrough.assign(chat_history_str=lambda x: format_chat_history(x["chat_history"]))
            | RunnablePassthrough.assign(rephrased_question=standalone_question_chain)
            | RunnablePassthrough.assign(docs=user_file_retrieval_chain)
            .assign(context=lambda x: format_docs(x["docs"]))
            | RunnableLambda(lambda x: push_files_out(x, files_out))
            | answer_chain
    ).with_config(run_name="Assistant")


def format_chat_history(chat_history: list):
    if chat_history and isinstance(chat_history[0], dict):
        return "\n".join([f"- {turn['role']}: {turn['content']}" for turn in chat_history])
    return "\n".join([f"- {turn.type}: {turn.content}" for turn in chat_history])


def get_chain(config, registry: PromptRegistry, files_out: list) -> Callable:
    user_file_retriever = user_file_vector_store().get_user_file_retriever()

    kllm = Kllm(config['llm'])

    chain = create_chain(kllm, user_file_retriever, registry, files_out)

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
