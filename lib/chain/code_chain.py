from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from lib.chain.prompt_registry import PromptRegistry
from lib.llm.kllm import Kllm
from lib.utils.chain_output_sink import ChainOutputSink


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
