# sizzurr Usage

This file provides detailed usage instructions for the `sizzurr` command and its subcommands.

## Global Options

-   `-h`, `--help`: Show the help message and exit.

## Subcommands

### `c2p2`

cut or copy files from subdirectories to the parent directory.

```
usage: sizzurr c2p2 [-h] --type TYPE [TYPE ...] [--copy] [--verbose]
                    [--dry-run] [--recursive]
                    parent_dir
```

**Positional Arguments:**

-   `parent_dir`: Parent directory path.

**Options:**

-   `--type`, `-t` TYPE [TYPE ...]: File extension(s) to search for (e.g., 'pdf', 'txt', 'docx').
-   `--copy`, `-c`: Copy files instead of cutting them.
-   `--verbose`, `-v`: Enable verbose output.
-   `--dry-run`, `-d`: Simulate without actually cutting/copying files.
-   `--recursive`, `-r`: Search recursively through all subdirectories (default: immediate subdirectories only).

### `relocate`

high-performance file relocation.

```
usage: sizzurr relocate [-h] [--recursive] [--preserve-structure]
                        [--include-dirs] [--copy] [--filter FILTER]
                        [--workers WORKERS] [--n-workers N_WORKERS]
                        [--memory-budget MEMORY_BUDGET] [--verify]
                        [--verbose] [--dry-run] [--parallel PARALLEL]
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
-   `--filter`, `-f` FILTER: Comma-separated file patterns to match (e.g., '*.pdf,*.txt').
-   `--workers`, `-w` WORKERS: Number of worker threads.
-   `--n-workers`, `-n` N_WORKERS: Number of parallel processes (alias for --workers).
-   `--memory-budget`, `-mb` MEMORY_BUDGET: Memory budget in MB (default: 8192).
-   `--verify`, `-vf`: Verify file integrity with hash comparison.
-   `--verbose`, `-v`: Enable verbose output.
-   `--dry-run`, `-d`: Simulate without actually moving files.
-   `--parallel`, `-p` PARALLEL: Number of parallel processes (alias for --workers).