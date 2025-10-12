# SIZZURR Usage

This file provides detailed usage instructions for the `sizzurr` command and its subcommands.

## Global Options

-   `-h`, `--help`: Show the help message and exit.

## Subcommands

### `c2p2`

Move or copy files from subdirectories to the parent directory.

```
usage: sizzurr c2p2 [-h] --type TYPE [TYPE ...] [--copy] [--verbose]
                    [--dry-run] [--recursive]
                    parent_dir
```

**Positional Arguments:**

-   `parent_dir`: Parent directory path.

**Options:**

-   `--type`, `-t` TYPE [TYPE ...]: File extension(s) to search for (e.g., 'pdf', 'txt', 'docx').
-   `--copy`, `-c`: Copy files instead of moving them.
-   `--verbose`, `-v`: Enable verbose output.
-   `--dry-run`, `-d`: Simulate without actually moving/copying files.
-   `--recursive`, `-r`: Search recursively through all subdirectories (default: immediate subdirectories only).

### `html2md`

Convert HTML files to Markdown.

```
usage: sizzurr html2md [-h] [--output OUTPUT] [--preserve-structure] [--clean]
                       [--pattern PATTERN] [--combine] [--verbose]
                       input
```

**Positional Arguments:**

-   `input`: Input HTML file or directory.

**Options:**

-   `--output`, `-o` OUTPUT: Output file or directory.
-   `--preserve-structure`: Preserve document structure.
-   `--clean`: Clean unwanted elements.
-   `--pattern` PATTERN: File pattern for batch processing.
-   `--combine`, `-c`: Combine all files into a single markdown document.
-   `--verbose`, `-v`: Enable verbose output.

### `pdf2md`

Convert PDF files to Markdown using a vision-language model.

```
usage: sizzurr pdf2md [-h] -i INPUT -o OUTPUT [-c CONFIG] [--dpi DPI]
                      [-cs CHUNK_SIZE] [--no-images] [-mdl MODEL]
                      [-f {github,obsidian,standard}] [--no-metadata] [-v]
                      [-gi {0,1}] [-mr MAX_RETRIES] [-mt MAX_TOKENS]
```

**Options:**

-   `-i`, `--input` INPUT: Input PDF file.
-   `-o`, `--output` OUTPUT: Output Markdown file.
-   `-c`, `--config` CONFIG: Configuration YAML file.
-   `--dpi` DPI: Image rendering DPI (default: 300).
-   `-cs`, `--chunk-size` CHUNK_SIZE: Pages per processing chunk (default: 2).
-   `--no-images`: Skip image extraction.
-   `-mdl`, `--model` MODEL: Vision model to use (default: Qwen/Qwen2.5-VL-3B-Instruct).
-   `-f`, `--format` {github,obsidian,standard}: Output format (default: github).
-   `--no-metadata`: Skip document metadata.
-   `-v`, `--verbose`: Enable verbose output.
-   `-gi`, `--gpu-id` {0,1}: Primary GPU ID (0=RTX3060, 1=RTX2080S, default: 0).
-   `-mr`, `--max-retries` MAX_RETRIES: Maximum retries per page (default: 2).
-   `-mt`, `--max-tokens` MAX_TOKENS: Maximum tokens per generation (default: 2048).

### `relocate`

High-performance file relocation.

```
usage: sizzurr relocate [-h] [--recursive] [--preserve-structure]
                        [--include-dirs] [--copy] [--filter FILTER]
                        [--workers WORKERS] [--memory-budget MEMORY_BUDGET]
                        [--verify] [--verbose] [--dry-run]
                        [--parallel PARALLEL]
                        source dest
```

**Positional Arguments:**

-   `source`: Source directory path.
-   `dest`: Destination directory path.

**Options:**

-   `--recursive`, `-r`: Process files recursively.
-   `--preserve-structure`, `-ps`: Preserve directory structure when using recursive mode.
-   `--include-dirs`, `-id`: Include directories in addition to files.
-   `--copy`, `-c`: Copy files instead of moving.
-   `--filter`, `-f` FILTER: File patterns to match (e.g., '*.pdf').
-   `--workers`, `-w` WORKERS: Number of worker threads.
-   `--memory-budget`, `-mb` MEMORY_BUDGET: Memory budget in MB (default: 8192).
-   `--verify`, `-vf`: Verify file integrity with hash comparison.
-   `--verbose`, `-v`: Enable verbose output.
-   `--dry-run`, `-d`: Simulate without actually moving files.
-   `--parallel`, `-p` PARALLEL: Number of parallel processes (alias for --workers).
