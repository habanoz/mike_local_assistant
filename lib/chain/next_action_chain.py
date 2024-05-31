from _operator import itemgetter
from typing import List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from lib.chain.prompt_registry import PromptRegistry
from lib.db.model import UserFile
from lib.llm.kllm import Kllm
from lib.utils.chain_output_sink import ChainOutputSink


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
