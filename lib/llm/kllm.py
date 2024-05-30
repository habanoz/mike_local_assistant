from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from lib.utils.chain_output_sink import ChainOutputSink


class Kllm:
    def __init__(self, config):
        self.answer_llm = LLMFactory.build(config['answer'])
        self.code_llm = LLMFactory.build(config['code'])
        self.deterministic_llm = LLMFactory.build(config['deterministic'])

    def get_answer_llm(self):
        return self.answer_llm

    def get_code_llm(self):
        return self.code_llm

    def get_deterministic_llm(self):
        return self.deterministic_llm

    def get_code_llm_runnable(self, name, chain_sink: ChainOutputSink):
        return (RunnableLambda(lambda x: self.push_prompt(x, name, chain_sink))
                | self.code_llm)

    def get_deterministic_llm_runnable(self, name, chain_sink: ChainOutputSink):
        return (RunnableLambda(lambda x: self.push_prompt(x, name, chain_sink))
                | self.deterministic_llm
                | RunnableLambda(lambda x: self.push_output(x, name, chain_sink)))

    def get_answer_llm_runnable(self, name, chain_sink: ChainOutputSink):
        return (RunnableLambda(lambda x: self.push_prompt(x, name, chain_sink))
                | self.answer_llm)

    def push_prompt(self, prompt: ChatPromptTemplate, prompt_name, chain_sink: ChainOutputSink):
        chain_sink.add_debug(
            name="prompt_" + prompt_name,
            content=[{"role": msg.type, "content": msg.content} for msg in prompt.messages]
        )
        return prompt

    def push_output(self, output: AIMessage, action_name, chain_sink: ChainOutputSink):
        chain_sink.add_debug(name=action_name, content=output.content)
        return output


class LLMFactory:
    @classmethod
    def build(cls, config):
        provider = config['provider']
        if provider == "ollama":
            return LLMFactory.ollama(config)
        else:
            raise Exception("Unknown provider" + str(provider))

    @classmethod
    def ollama(cls, config):
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(model=config['model'],
                          base_url=config['base_url'],
                          temperature=config['temperature'],
                          stop=config['stop'])
