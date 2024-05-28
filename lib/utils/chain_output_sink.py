class ChainOutputSink:
    def __init__(self):
        super().__init__()
        self.files: list[dict] = []
        self.debug: list[dict] = []

    def add_file(self, name, file, content):
        self.files.append({"name": name, "file": file, "content": content})

    def add_debug(self, name, content):
        self.debug.append({"name": name, "content": content})

    def add_files(self, files: list[dict]):
        self.files.extend(files)

    def get_and_clear_source_files_sink(self) -> list:
        copied = list(self.files)

        self.files.clear()

        return copied

    def get_and_clear_debug_sink(self) -> list:
        copied = list(self.debug)

        self.debug.clear()

        return copied
