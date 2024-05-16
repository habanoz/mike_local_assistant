import yaml


def _load_yaml_file(file):
    with open(file) as stream:
        try:
            return yaml.safe_load(stream)
        except Exception:
            raise


class PromptRegistry:
    def __init__(self):
        self.task_map = {}
        self._instructions = self._load_instructions()
        self._prompts = self._load_prompts(self._instructions)

    @property
    def prompts(self) -> dict[str:str]:
        return self._prompts

    @property
    def instructions(self) -> dict[str:str]:
        return self._instructions

    def _load_prompts(self, instructions: dict):
        prompts = _load_yaml_file("config/prompts.yml")['prompts']
        return {prompt['task']: self._replace_instruction(prompt['content'], instructions) for prompt in prompts}

    def _replace_instruction(self, text, instructions: dict) -> dict[str:str]:
        for name, content in instructions.items():
            text = text.replace(f"{{{{{name}}}}}", content)
        return text

    def _load_instructions(self) -> dict[str:str]:
        instructions = _load_yaml_file("config/instructions.yml")['instructions']
        return {instruction['type'].strip(): instruction['content'].strip() for instruction in
                instructions}
