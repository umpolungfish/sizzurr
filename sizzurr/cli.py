#!/usr/bin/env python3

import argparse

from sizzurr import c2p2
from sizzurr import power_relocator

def main():
    parser = argparse.ArgumentParser(description="sizzurr - a pair of sharp cli scissors")
    subparsers = parser.add_subparsers(dest="command")

    # c2p2 subcommand
    c2p2_parser = subparsers.add_parser(
        "c2p2",
        help="cut or copy files from subdirectories to parent directory.",
        epilog="""Examples:
  # cut all .pdf and .docx files from immediate subdirectories of the current directory into the current directory.
  sizzurr c2p2 . -t pdf docx

  # Copy all .jpg and .png files from all subdirectories of /path/to/images into that same directory.
  sizzurr c2p2 /path/to/images -t jpg png --copy --recursive

  # Perform a verbose dry run, showing which .txt files would be cut from all subdirectories of /path/to/documents.
  sizzurr c2p2 /path/to/documents -t txt --dry-run --verbose --recursive

  # cut all .py and .js files from only the immediate subdirectories of the project folder.
  sizzurr c2p2 /path/to/project -t py js

  # Copy all markdown files recursively with verbose output.
  sizzurr c2p2 . -t md --copy -r -v
""",
        formatter_class=argparse.RawTextHelpFormatter
    )
    c2p2.main_args(c2p2_parser)





    # power_relocator subcommand
    power_relocator_parser = subparsers.add_parser(
        "relocate",
        help="high-performance file relocation.",
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
    elif args.command == "relocate":
        power_relocator.main(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
