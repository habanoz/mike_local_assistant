from typing import Sequence

from langchain_core.documents import Document

from lib.utils.chain_output_sink import ChainOutputSink


def push_prompts_out(prompt, prompt_name, chain_sink: ChainOutputSink):
    chain_sink.add_debug(
        name="prompt_" + prompt_name,
        content=[{"role": msg.type, "content": msg.content} for msg in prompt.messages]
    )
    return prompt


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


def format_chat_history(chat_history: list):
    if chat_history and isinstance(chat_history[0], dict):
        return "\n".join([f"- {turn['role']}: {turn['content']}" for turn in chat_history])
    return "\n".join([f"- {turn.type}: {turn.content}" for turn in chat_history])


def format_docs(docs: Sequence[Document]) -> str:
    formatted_docs = []
    for i, doc in enumerate(docs, start=1):
        doc_string = f"<doc id='{i}'>{doc.page_content}</doc>"
        formatted_docs.append(doc_string)
    return "\n".join(formatted_docs)
