#!/usr/bin/env python3
"""
Usage:
    python cut2parent.py /path/to/parent/directory -t pdf txt docx -v
    python cut2parent.py /path/to/parent/directory --type pdf -c --dry-run
    python cut2parent.py /path/to/parent/directory -t pdf txt -c -v -d -r
    python cut2parent.py /path/to/parent/directory -t pdf --recursive --verbose
"""

import os
import shutil
import argparse
from pathlib import Path
import sys

def process_files_in_directory(root_path, files, file_exts, parent_dir, operation, 
                              copy_mode, verbose, dry_run, processed_count, error_count):
    """Process files in a single directory."""
    for file in files:
        # Check if file matches any of the specified extensions
        if any(file.endswith(f".{ext}") for ext in file_exts):
            source_path = root_path / file
            target_path = parent_dir / file
            
            # Check if a file with the same name already exists in the parent directory
            if target_path.exists():
                if verbose:
                    print(f"Warning: File '{file}' already exists in the parent directory. Adding suffix.")
                
                # Add a suffix to the filename to make it unique
                filename, ext = os.path.splitext(file)
                counter = 1
                while target_path.exists():
                    new_filename = f"{filename}_{counter}{ext}"
                    target_path = parent_dir / new_filename
                    counter += 1
            
            if dry_run:
                print(f"Would {operation}: {source_path} -> {target_path}")
                processed_count += 1
            else:
                try:
                    if copy_mode:
                        # Copy the file (preserving metadata)
                        shutil.copy2(str(source_path), str(target_path))
                    else:
                        # Move the file
                        shutil.move(str(source_path), str(target_path))
                    
                    processed_count += 1
                    
                    if verbose:
                        print(f"{operation.capitalize()}d: {source_path} -> {target_path}")
                
                except Exception as e:
                    error_count += 1
                    if verbose:
                        print(f"Error {operation}ing {source_path}: {e}", file=sys.stderr)
    
    return processed_count, error_count

def main_args(parser):
    parser.add_argument("parent_dir", help="Parent directory path")
    parser.add_argument("--type", "-t", required=True, nargs='+', help="One or more file extensions to search for (e.g., pdf txt docx)")
    parser.add_argument("--copy", "-c", action="store_true", help="Copy files instead of moving them")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--dry-run", "-d", action="store_true", help="Simulate without actually moving/copying files")
    parser.add_argument("--recursive", "-r", action="store_true", help="Search recursively through all subdirectories (default: immediate subdirectories only)")

def main(args=None):
    """Main function to handle file operations."""
    if args is None:
        parser = argparse.ArgumentParser(description="Move or copy files of specified type(s) from subdirectories to parent directory.")
        main_args(parser)
        args = parser.parse_args()
    
    parent_dir = Path(args.parent_dir).resolve()
    file_exts = [ext.lstrip('.') for ext in args.type]  # Remove leading dots if present
    operation = "copy" if args.copy else "move"
    verbose = args.verbose
    dry_run = args.dry_run
    recursive = args.recursive
    
    if not parent_dir.is_dir():
        print(f"Error: {parent_dir} is not a valid directory", file=sys.stderr)
        return 1
    
    if verbose:
        extensions_str = ", ".join([f".{ext}" for ext in file_exts])
        search_mode = "recursively through all subdirectories" if recursive else "in immediate subdirectories only"
        print(f"Searching for {extensions_str} files {search_mode} of {parent_dir}")
        print(f"Operation: {operation.upper()}")
        if dry_run:
            print("DRY RUN: No files will be moved/copied")
    
    processed_count = 0
    error_count = 0
    
    # Walk through subdirectories (recursive or immediate only)
    if recursive:
        # Walk through all subdirectories recursively
        for root, dirs, files in os.walk(parent_dir):
            root_path = Path(root)
            
            # Skip the parent directory itself
            if root_path == parent_dir:
                continue
            
            # Process files in current directory
            processed_count, error_count = process_files_in_directory(
                root_path, files, file_exts, parent_dir, operation, 
                args.copy, verbose, dry_run, processed_count, error_count
            )
    else:
        # Only search immediate subdirectories (depth 1)
        for item in parent_dir.iterdir():
            if item.is_dir():
                try:
                    files = [f.name for f in item.iterdir() if f.is_file()]
                    processed_count, error_count = process_files_in_directory(
                        item, files, file_exts, parent_dir, operation,
                        args.copy, verbose, dry_run, processed_count, error_count
                    )
                except PermissionError:
                    if verbose:
                        print(f"Warning: Permission denied accessing {item}", file=sys.stderr)
    
    # Print summary
    if dry_run:
        print(f"Dry run complete: {processed_count} files would be {operation}d")
    else:
        action_past_tense = "copied" if args.copy else "moved"
        print(f"Operation complete: {processed_count} files {action_past_tense}, {error_count} errors")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
