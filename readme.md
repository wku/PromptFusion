# PromptFusion: Intelligent Codebase Analysis

## Primary Purpose

PromptFusion is a specialized tool designed for deep exploration and rapid understanding of large codebases using Large Language Models (LLMs). The core purpose of this project is to significantly accelerate the process of analyzing large software projects consisting of numerous files, helping developers quickly find answers or identify issues in unfamiliar code.

The tool is particularly valuable in the following scenarios:
- Initial familiarization with a new codebase
- Finding specific functionality implementations in a large project
- Exploring relationships between components in a complex system
- Identifying potential problems or bottlenecks in the architecture
- Gaining detailed understanding of code behavior without manual analysis of each file

## About the Project

PromptFusion is a tool for interacting with codebases using Large Language Models (LLMs). The project allows you to analyze, navigate, and get consultations about codebases using generative language models.

### Key Features

- Analysis of project structure and file descriptions
- Semantic search across the codebase
- Interactive AI chat for code discussions
- Support for multiple programming languages
- API cost optimization

## Installation

1. Clone the repository
```bash
git clone https://github.com/wku/PromptFusion
cd PromptFusion
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root
```
OPENROUTER_API_KEY=your_api_key_here
```

## Usage

### Launch

```bash
python app.py
```

On first launch, you'll be prompted to specify the path to the project for analysis. The program will create an index of project files and their descriptions using LLM.

### Basic Chat Commands

- Ask questions about the structure and functionality of the project
- `/clear` - clear message history
- `/exit` - exit the chat

### Example Questions

- "Explain the main structure of the project"
- "How is function X implemented in file Y?"
- "Find all places where class Z is used"
- "Tell me about the interaction between modules A and B"
- "What architectural issues exist in the project?"
- "How is error handling implemented in the authentication module?"

## Project Architecture

### Key Modules

- **analyzers.py** - Code structure analyzers for various languages
- **app.py** - Application entry point
- **config.py** - Configuration management
- **core.py** - Core application logic
- **improved_chat_session.py** - Chat implementation with prompt tools
- **models.py** - Data models
- **prompt_tools.py** - Prompt tools for working with LLM

### Supported Programming Languages

- Python
- JavaScript/TypeScript
- Java
- C/C++
- Go
- Rust
- Solidity

## Prompt Tools

The project uses an innovative approach to interacting with LLMs through prompt engineering instead of standard API functions. This ensures compatibility with various LLM providers.

### How Prompt Tools Work

1. Instructions for using functions are added to the system prompt
2. LLM can call functions in the format `[FUNCTION: function_name(parameter="value")]`
3. Function calls are processed and replaced with results in the format `[RESULT: function_name]...result...[/RESULT]`

### Available Functions

- `get_file(path)` - Loads file content
- `find_files_semantic(query, page)` - Searches files using semantic search
- `find_in_files(query, is_case_sensitive, page)` - Searches for text in files
- `update_file(path, content)` - Updates or creates a file

## Analysis Modes

The project supports various file description modes:

- **MODE_DESC** - Full descriptions (standard mode)
- **MODE_DESC_NO** - No descriptions
- **MODE_DESC_2** - Brief descriptions

## Cost Optimization

The system automatically tracks:
- Number of tokens in requests/responses
- Cost of API requests
- File sizes

It warns about large files and projects that may lead to high costs.

## Project Advantages

- **Deep code analysis** - specialized parsers for different languages
- **Flexible settings** - control over file inclusion/exclusion
- **Semantic search** - using vector embeddings
- **Modular architecture** - clear separation of components
- **Local analysis** - minimizing data sent to LLM
- **Incremental updates** - updating only modified files
- **Universal compatibility** - works with various LLM APIs

## Limitations and Diagnostics

### Known Limitations
- Depends on the LLM's ability to correctly follow the format
- Large projects may require significant resources
- Increased prompt size due to inclusion of function results

### Diagnostic Tips
If the LLM incorrectly formats function calls:
1. Check the system prompt for clear instructions
2. Use a lower temperature (0-0.2)
3. Add examples of function usage

## Contributing to the Project

We welcome contributions to the project! If you have suggestions for improvement or have found a bug:

1. Create an issue
2. Submit a pull request with proposed changes

## License

The project is distributed under the MIT license