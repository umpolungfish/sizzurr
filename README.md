# SIZZURR

SIZZURR is a collection of powerful command-line utilities written in Python. The main goal is to combine these scripts into a single, globally installed CLI tool. The toolkit is designed for data ingestion, preparation, and management, with a particular focus on AI-related tasks.

## Features

- **`c2p2`**: A simple utility for moving or copying files from subdirectories to their parent directory.
- **`html2md`**: A sophisticated script for converting HTML files to Markdown, optimized for Retrieval-Augmented Generation (RAG) applications.
- **`pdf2md`**: An advanced script that uses a vision-language model (Qwen-VL) to convert PDFs to Markdown, preserving complex structures like code, formulas, and tables.
- **`relocate`**: A high-performance file relocation utility for moving or copying large amounts of data efficiently, using parallel processing and memory mapping.

## Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/your-username/sizzurr.git
    cd sizzurr
    ```

2.  Create a virtual environment and install the dependencies:

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -e .
    ```

## Usage

Once installed, the `sizzurr` command will be available in your terminal.

For detailed usage instructions, please see the `USAGE.md` file.
