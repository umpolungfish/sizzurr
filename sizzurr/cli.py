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
    c2p2_parser = subparsers.add_parser("c2p2", help="Move or copy files from subdirectories to parent directory.")
    c2p2.main_args(c2p2_parser)

    # html_md_converter subcommand
    html_md_converter_parser = subparsers.add_parser("html2md", help="Convert HTML files to Markdown.")
    html_md_converter.main_args(html_md_converter_parser)

    # pdf_to_md_converter subcommand
    pdf_to_md_converter_parser = subparsers.add_parser("pdf2md", help="Convert PDF files to Markdown using a vision-language model.")
    pdf_to_md_converter.main_args(pdf_to_md_converter_parser)

    # power_relocator subcommand
    power_relocator_parser = subparsers.add_parser("relocate", help="High-performance file relocation.")
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
