## Project Overview

This project, named "SIZZURR", is a collection of powerful command-line utilities written in Python. The main goal is to combine these scripts into a single, globally installed CLI tool. The toolkit is designed for data ingestion, preparation, and management, with a particular focus on AI-related tasks.

The project consists of the following scripts:

*   **`c2p2.py`**: A simple utility for moving or copying files from subdirectories to their parent directory.
*   **`html_md_converter.py`**: A sophisticated script for converting HTML files to Markdown, optimized for Retrieval-Augmented Generation (RAG) applications.
*   **`pdf_to_md_converter.py`**: An advanced script that uses a vision-language model (Qwen-VL) to convert PDFs to Markdown, preserving complex structures like code, formulas, and tables.
*   **`power_relocator.py`**: A high-performance file relocation utility for moving or copying large amounts of data efficiently, using parallel processing and memory mapping.

## Building and Running

The project is a collection of Python scripts and does not have a central build system. Each script can be run individually using the Python interpreter.

To run any of the scripts, use the following command structure:

```bash
python <script_name>.py [arguments]
```

For example, to run the `html_md_converter.py` script:

```bash
python html_md_converter.py --input <input_file_or_directory> --output <output_file_or_directory>
```

Each script has its own set of command-line arguments, which can be viewed by running the script with the `--help` flag.

**Dependencies:**

The scripts have several Python dependencies, which are listed in the import sections of each file. The key dependencies include:

*   `beautifulsoup4`
*   `html2text`
*   `markdownify`
*   `PyMuPDF` (fitz)
*   `pdfplumber`
*   `torch`
*   `transformers`
*   `Pillow`
*   `opencv-python`
*   `pyyaml`
*   `psutil`

These dependencies can be installed using pip:

```bash
pip install beautifulsoup4 html2text markdownify PyMuPDF pdfplumber torch transformers Pillow opencv-python pyyaml psutil
```

## Development Conventions

*   The project consists of standalone Python scripts, each with its own command-line interface.
*   The scripts are well-documented with usage examples in their docstrings.
*   The `pdf_to_md_converter.py` script is designed to use a specific vision-language model and requires a CUDA-enabled GPU.
*   The `power_relocator.py` script is designed for high-performance file operations and uses multiprocessing and memory mapping.
*   The overall goal is to unify these scripts into a single CLI tool, which suggests that future development will involve creating a wrapper or a central entry point for all the functionalities.
