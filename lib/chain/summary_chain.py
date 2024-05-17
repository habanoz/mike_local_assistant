from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from lib.chain.prompt_registry import PromptRegistry

from lib.llm.kllm import Kllm
from lib.st.cached import config, prompts_registry


def get_summary_chain(kllm: Kllm, registry: PromptRegistry):
    generate_file_summary = registry.prompts['generate_file_summary']

    summary_chain = (
        (
                ChatPromptTemplate.from_template(generate_file_summary)
                | kllm.get_deterministic_llm()
                | StrOutputParser()
        )
        .with_config(run_name="SummarizeUploadedFile")
    )

    return summary_chain


def summarize(content: str, max_words=400):
    words = content.split()
    truncated_content = ' '.join(words[:max_words])

    kllm = Kllm(config()['llm'])
    prompt_registry = prompts_registry()

    chain = get_summary_chain(kllm, prompt_registry)
    summary = chain.invoke({"uploaded_file_content": truncated_content})

    return summary
