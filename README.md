# Local Assistant

Local assistant, Mike, is an AI assistant chatbot designed to run in your personal PC.
Mike can help you to increase your productivity without any risk on your privacy.
Upload your files for answers grounded on your documents. For general knowledge and up-to-date information, the
assistant may make use of web search. For specialized coding tasks the assistant may make use of a code llm.

## Installation

Local assistant relies on Ollama to host a local language model. You need to install Ollama and download a language
model.

### Ollama Installation

Go to [Ollama Home Page](https://ollama.com/). Choose your platform to install. And follow the instructions.

Download a language model. Llama3 is recommended:

```bash
ollama pull llama3
```

Download a code model. eramax/nxcode-cq-7b-orpo:q6 is recommended:

```bash
ollama pull eramax/nxcode-cq-7b-orpo:q6
```

Download an embedding model.

```bash
ollama pull nomic-embed-text
``` 

Please note that language models run best at GPUs. If you do not have a GPU you can still use a language model, but it
will be slow.
A typical language model will require approximately 5GB of disk space.

For a list of available models you may refer to [Ollama Models Library](https://ollama.com/library)

### Python Installation

Go to [Official Python Downloads Page](https://www.python.org/downloads/).

### Installing Local Assistant

Use install scripts provided for your platform e.g. windows, linux.

For linux environment you may need to give execute permission to the scripts.

```bash
chmod +x install-linux.sh
```

### Configuration

Configuration file is `config/config.yml`. Edit it to configure options like models, database, etc.

Prompts are `config/prompts.yml` file. You may edit the file for providing your own prompts.

### Launching Local Assistant

Use launch script provided for your platform e.g. windows, linux.

For linux environment you may need to give execute permission to the scripts.

```bash
chmod +x launch-linux.sh
```

## Using Local Assistant

Open chat screen at [Assistant Chat](http://localhost:8501/Assistant_Chat).

You may upload your files using the upload screen at [Upload Page](http://localhost:8501/Upload).

## Provide Feedback

If you encounter an issue feel free to report by opening a GitHub issue.

