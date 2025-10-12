#!/usr/bin/env python3

import argparse

from sizzurr import c2p2
from sizzurr import html_md_converter
from sizzurr import pdf_to_md_converter
from sizzurr import power_relocator

def main():
    parser = argparse.ArgumentParser(description="SIZZURR - A collection of powerful command-line utilities.")
    subparsers = parser.add_subparsers(dest="command")

    # c2p2 subcommand
    c2p2_parser = subparsers.add_parser(
        "c2p2",
        help="Move or copy files from subdirectories to parent directory.",
        epilog="""Examples:
  # Move all .pdf and .docx files from immediate subdirectories of the current directory into the current directory.
  sizzurr c2p2 . -t pdf docx

  # Copy all .jpg and .png files from all subdirectories of /path/to/images into that same directory.
  sizzurr c2p2 /path/to/images -t jpg png --copy --recursive

  # Perform a verbose dry run, showing which .txt files would be moved from all subdirectories of /path/to/documents.
  sizzurr c2p2 /path/to/documents -t txt --dry-run --verbose --recursive

  # Move all .py and .js files from only the immediate subdirectories of the project folder.
  sizzurr c2p2 /path/to/project -t py js

  # Copy all markdown files recursively with verbose output.
  sizzurr c2p2 . -t md --copy -r -v
""",
        formatter_class=argparse.RawTextHelpFormatter
    )
    c2p2.main_args(c2p2_parser)

    # html_md_converter subcommand
    html_md_converter_parser = subparsers.add_parser(
        "html2md",
        help="Convert HTML files to Markdown.",
        epilog="""
Examples:
  # Convert a single HTML file to a Markdown file with the same name.
  sizzurr html2md input.html

  # Convert a single HTML file and specify the output file.
  sizzurr html2md input.html -o output.md

  # Convert all .html files in a directory and its subdirectories.
  sizzurr html2md /path/to/html_files

  # Convert all .htm and .html files in a directory and combine them into a single file.
  sizzurr html2md /path/to/html_files --pattern "**/*.htm*" -c
""",
        formatter_class=argparse.RawTextHelpFormatter
    )
    html_md_converter.main_args(html_md_converter_parser)

    # pdf_to_md_converter subcommand
    pdf_to_md_converter_parser = subparsers.add_parser(
        "pdf2md",
        help="Convert PDF files to Markdown using a vision-language model.",
        epilog="""Examples:
  # Basic conversion with verbose output.
  sizzurr pdf2md -i input.pdf -o output.md -v

  # Convert a PDF using a specific, smaller model and a different GPU.
  sizzurr pdf2md -i technical_paper.pdf -o paper.md -mdl Qwen/Qwen2-VL-2B-Instruct --gpu-id 1

  # Convert a PDF with high-resolution images, but skip extracting them in the output.
  sizzurr pdf2md -i my_report.pdf -o report.md --dpi 600 --no-images

  # Use a custom configuration file for conversion settings.
  sizzurr pdf2md -i document.pdf -o document.md -c my_config.yaml

  # Faster conversion with a larger chunk size, but potentially higher memory usage.
  sizzurr pdf2md -i large_book.pdf -o book.md --chunk-size 4
""",
        formatter_class=argparse.RawTextHelpFormatter
    )
    pdf_to_md_converter.main_args(pdf_to_md_converter_parser)

    # power_relocator subcommand
    power_relocator_parser = subparsers.add_parser(
        "relocate",
        help="High-performance file relocation.",
        epilog="""Examples:
  # Move all files from /source/path to /dest/path.
  sizzurr relocate /source/path /dest/path

  # Copy all files and directories recursively from /source/path to /dest/path.
  sizzurr relocate /source/path /dest/path -r -c --include-dirs

  # Move only .jpg and .png files recursively, preserving the directory structure.
  sizzurr relocate /source/path /dest/path -r --preserve-structure -f "*.jpg,*.png"

  # Perform a verbose dry run of a copy operation with 16 worker threads.
  sizzurr relocate /source/path /dest/path -c -r -v -d -w 16

  # Move large video files and verify their integrity after the transfer.
  sizzurr relocate /video_source /video_dest -r -f "*.mp4,*.mkv" --verify
""",
        formatter_class=argparse.RawTextHelpFormatter
    )
    power_relocator.main_args(power_relocator_parser)

    args = parser.parse_args()

    if args.command == "c2p2":
        c2p2.main(args)
    elif args.command == "html2md":
        html_md_converter.main(args)
    elif args.command == "pdf2md":
        pdf_to_md_converter.main(args)
    elif args.command == "relocate":
        power_relocator.main(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
