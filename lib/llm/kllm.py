from abc import ABC, abstractmethod

from langchain_core.runnables import ConfigurableField


class BaseKllm(ABC):

    @abstractmethod
    def get_creative_llm(self):
        pass

    @abstractmethod
    def get_deterministic_llm(self):
        pass


class Kllm:
    def __init__(self, config):
        provider = config['provider']
        if provider == "ollama":
            print("Ollama provider is selected. Make sure you are running in localhost!")
            self.provider = LocalKllmProvider(config[provider])
        else:
            raise Exception("Unknown provider" + str(provider))

    def get_creative_llm(self):
        return self.provider.get_creative_llm()

    def get_deterministic_llm(self):
        return self.provider.get_deterministic_llm()


class LocalKllmProvider(BaseKllm):
    def __init__(self, config):
        from langchain_community.chat_models import ChatOllama
        self.temperature = config['temperature']
        self.chat_llm = (
            ChatOllama(model=config['model'],
                       base_url=config['base_url'],
                       temperature=0.0,
                       stop=[config['stop']])
            .configurable_fields(
                temperature=ConfigurableField(
                    id="llm_temperature",
                    name="LLM Temperature",
                    description="The temperature of the LLM",
                )
            )
        )

    def get_creative_llm(self):
        return self.chat_llm.with_config(configurable={"llm_temperature": self.temperature})

    def get_deterministic_llm(self):
        return self.chat_llm
