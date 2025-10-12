#!/usr/bin/env python3
"""
Usage:
    python power_relocator.py /source/path /dest/path -v -p 8
    python power_relocator.py /source/path /dest/path -r -c -mb 1024 -w 16 -v
    python power_relocator.py /source/path /dest/path -r -ps -f "*.pdf" -f "*.txt" -d -v
    python power_relocator.py /source/path /dest/path -r -id -ps -c -v
"""

import os
import shutil
import argparse
import multiprocessing
import threading
from pathlib import Path
import sys
import time
import mmap
import fnmatch
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from queue import Queue
import psutil
from collections import defaultdict
import hashlib

class PowerRelocator:
    def __init__(self, workers=None, memory_budget_mb=8192, use_mmap_threshold=100*1024*1024):
        """
        Initialize the power relocator.
        
        Args:
            workers: Number of worker processes (default: CPU count)
            memory_budget_mb: Memory budget in MB (default: 8GB)
            use_mmap_threshold: File size threshold for memory mapping (default: 100MB)
        """
        self.workers = workers or multiprocessing.cpu_count()
        self.memory_budget = memory_budget_mb * 1024 * 1024  # Convert to bytes
        self.mmap_threshold = use_mmap_threshold
        self.stats = {
            'files_processed': 0,
            'bytes_transferred': 0,
            'errors': 0,
            'start_time': None,
            'skipped': 0
        }
        
    def get_files_to_process(self, source_path, patterns=None, recursive=False, include_dirs=False):
        """Efficiently gather files and optionally directories matching patterns."""
        items = []
        source = Path(source_path)
        
        if recursive:
            item_iter = source.rglob('*')
        else:
            item_iter = source.iterdir()
            
        for item in item_iter:
            # Skip the source directory itself to avoid infinite recursion
            if item == source:
                continue
                
            item_type = None
            if item.is_file():
                item_type = 'file'
            elif item.is_dir() and include_dirs:
                item_type = 'dir'
            else:
                continue
                
            # Apply pattern matching
            if patterns:
                if any(fnmatch.fnmatch(item.name, pattern) for pattern in patterns):
                    items.append((item, item.relative_to(source), item_type))
            else:
                items.append((item, item.relative_to(source), item_type))
                    
        return items
    
    def calculate_file_hash(self, filepath, chunk_size=8192):
        """Calculate MD5 hash for verification."""
        hash_md5 = hashlib.md5()
        try:
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(chunk_size), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return None
    
    def smart_copy_file(self, source_path, dest_path, verify=False):
        """
        Intelligently copy file using different strategies based on size.
        """
        try:
            source_size = source_path.stat().st_size
            
            # Strategy 1: Small files - direct copy
            if source_size < 1024 * 1024:  # < 1MB
                shutil.copy2(source_path, dest_path)
                
            # Strategy 2: Medium files - chunked copy with larger buffer
            elif source_size < self.mmap_threshold:
                self._chunked_copy(source_path, dest_path, chunk_size=1024*1024)  # 1MB chunks
                
            # Strategy 3: Large files - memory mapped copy
            else:
                self._mmap_copy(source_path, dest_path)
            
            # Verification if requested
            if verify:
                source_hash = self.calculate_file_hash(source_path)
                dest_hash = self.calculate_file_hash(dest_path)
                if source_hash != dest_hash:
                    raise Exception("Hash verification failed")
                    
            return True, source_size
            
        except Exception as e:
            return False, str(e)
    
    def smart_copy_directory(self, source_path, dest_path, copy_mode=True):
        """
        Copy or move directory with all contents.
        """
        try:
            if copy_mode:
                # Copy entire directory tree
                shutil.copytree(source_path, dest_path, dirs_exist_ok=True)
                total_size = sum(f.stat().st_size for f in source_path.rglob('*') if f.is_file())
            else:
                # Move directory
                shutil.move(str(source_path), str(dest_path))
                total_size = 0  # Can't calculate size after move
                
            return True, total_size
            
        except Exception as e:
            return False, str(e)
    
    def _chunked_copy(self, source_path, dest_path, chunk_size=1024*1024):
        """Copy file in chunks with optimized buffer size."""
        with open(source_path, 'rb') as src, open(dest_path, 'wb') as dst:
            while True:
                chunk = src.read(chunk_size)
                if not chunk:
                    break
                dst.write(chunk)
        
        # Preserve metadata
        shutil.copystat(source_path, dest_path)
    
    def _mmap_copy(self, source_path, dest_path):
        """Memory-mapped copy for large files."""
        with open(source_path, 'rb') as src, open(dest_path, 'wb') as dst:
            with mmap.mmap(src.fileno(), 0, access=mmap.ACCESS_READ) as mmapped_file:
                dst.write(mmapped_file)
        
        # Preserve metadata
        shutil.copystat(source_path, dest_path)
    
    def process_file_batch(self, item_batch, dest_base, copy_mode=False, verify=False, verbose=False, preserve_structure=False):
        """Process a batch of files and directories in a single worker."""
        local_stats = {'processed': 0, 'bytes': 0, 'errors': 0, 'skipped': 0, 'dirs': 0, 'files': 0}
        
        for item_info in item_batch:
            try:
                # Handle different tuple formats for backward compatibility
                if len(item_info) == 3:
                    source_item, relative_path, item_type = item_info
                elif len(item_info) == 2:
                    source_item, relative_path = item_info
                    item_type = 'file'  # Default for backward compatibility
                else:
                    source_item = item_info
                    relative_path = Path(source_item.name)
                    item_type = 'file'
                
                # Determine destination path
                if preserve_structure:
                    dest_item = Path(dest_base) / relative_path
                else:
                    dest_item = Path(dest_base) / source_item.name
                
                # Handle name conflicts
                counter = 1
                original_dest = dest_item
                while dest_item.exists():
                    if item_type == 'dir':
                        dest_item = original_dest.parent / f"{original_dest.name}_{counter}"
                    else:
                        stem, suffix = original_dest.stem, original_dest.suffix
                        dest_item = original_dest.parent / f"{stem}_{counter}{suffix}"
                    counter += 1
                
                # Ensure parent directory exists
                if item_type == 'file':
                    dest_item.parent.mkdir(parents=True, exist_ok=True)
                else:
                    dest_item.parent.mkdir(parents=True, exist_ok=True)
                
                # Process based on item type
                if item_type == 'file':
                    # Process file
                    success, result = self.smart_copy_file(source_item, dest_item, verify)
                    
                    if success:
                        bytes_transferred = result
                        local_stats['processed'] += 1
                        local_stats['files'] += 1
                        local_stats['bytes'] += bytes_transferred
                        
                        # Remove source if moving
                        if not copy_mode:
                            source_item.unlink()
                        
                        if verbose:
                            action = "Copied" if copy_mode else "Moved"
                            print(f"✔ {action} file: {source_item} -> {dest_item}")
                            
                    else:
                        local_stats['errors'] += 1
                        if verbose:
                            print(f"❌ Error processing file {source_item}: {result}")
                
                elif item_type == 'dir':
                    # Process directory
                    success, result = self.smart_copy_directory(source_item, dest_item, copy_mode)
                    
                    if success:
                        bytes_transferred = result if copy_mode else 0
                        local_stats['processed'] += 1
                        local_stats['dirs'] += 1
                        local_stats['bytes'] += bytes_transferred
                        
                        if verbose:
                            action = "Copied" if copy_mode else "Moved"
                            print(f"✔ {action} directory: {source_item} -> {dest_item}")
                            
                    else:
                        local_stats['errors'] += 1
                        if verbose:
                            print(f"❌ Error processing directory {source_item}: {result}")
                        
            except Exception as e:
                local_stats['errors'] += 1
                if verbose:
                    print(f"❌ Error processing {source_item}: {e}")
        
        return local_stats
    
    def create_batches(self, items, batch_size_mb=256):
        """Create intelligent batches based on memory budget and file sizes."""
        batches = []
        current_batch = []
        current_size = 0
        batch_size_bytes = batch_size_mb * 1024 * 1024
        
        # Handle different tuple formats and calculate sizes
        items_with_sizes = []
        for item_info in items:
            if len(item_info) == 3:
                item_path, _, item_type = item_info
            elif len(item_info) == 2:
                item_path, _ = item_info
                item_type = 'file'
            else:
                item_path = item_info
                item_type = 'file'
            
            # Calculate size based on item type
            if item_type == 'file':
                size = item_path.stat().st_size
            else:  # directory
                # For directories, calculate total size of all files within
                try:
                    size = sum(f.stat().st_size for f in item_path.rglob('*') if f.is_file())
                except:
                    size = 0  # Fallback if we can't calculate
            
            items_with_sizes.append((item_info, size))
        
        # Sort items by size (process small items first for better parallelization)
        items_with_sizes.sort(key=lambda x: x[1])
        
        for item_info, size in items_with_sizes:
            if current_size + size > batch_size_bytes and current_batch:
                batches.append(current_batch)
                current_batch = [item_info]
                current_size = size
            else:
                current_batch.append(item_info)
                current_size += size
        
        if current_batch:
            batches.append(current_batch)
            
        return batches
    
    def relocate(self, source_path, dest_path, patterns=None, recursive=False, 
                copy_mode=False, verify=False, verbose=False, dry_run=False, preserve_structure=False, include_dirs=False):
        """Main relocation function with parallel processing."""
        
        source = Path(source_path)
        dest = Path(dest_path)
        
        if not source.exists():
            print(f"❌ Source path does not exist: {source}")
            return False
            
        if not dry_run:
            dest.mkdir(parents=True, exist_ok=True)
        
        # Gather items to process (files and optionally directories)
        item_type_str = "files" + (" and directories" if include_dirs else "")
        print(f"🔍 Scanning {item_type_str}...")
        items_to_process = self.get_files_to_process(source, patterns, recursive, include_dirs)
        
        if not items_to_process:
            print("⚠ No items found matching criteria")
            return True
            
        # Calculate total size and separate counts
        total_size = 0
        file_count = 0
        dir_count = 0
        
        for item_info in items_to_process:
            if len(item_info) == 3:
                item_path, _, item_type = item_info
                if item_type == 'file':
                    file_count += 1
                    total_size += item_path.stat().st_size
                else:
                    dir_count += 1
                    try:
                        total_size += sum(f.stat().st_size for f in item_path.rglob('*') if f.is_file())
                    except:
                        pass  # Skip if we can't calculate directory size
            else:
                # Backward compatibility
                item_path = item_info[0] if isinstance(item_info, tuple) else item_info
                file_count += 1
                total_size += item_path.stat().st_size
                
        count_str = f"{file_count} files"
        if dir_count > 0:
            count_str += f" and {dir_count} directories"
            
        print(f"📊 Found {count_str} ({total_size / (1024**3):.2f} GB)")
        
        if dry_run:
            action = "copy" if copy_mode else "move"
            print(f"🧪 DRY RUN: Would {action} {len(items_to_process)} items")
            for i, item_info in enumerate(items_to_process[:10]):  # Show first 10
                if len(item_info) == 3:
                    item_path, relative_path, item_type = item_info
                    type_icon = "📁" if item_type == 'dir' else "📄"
                else:
                    item_path, relative_path = item_info if isinstance(item_info, tuple) else (item_info, item_info.name)
                    type_icon = "📄"
                
                if preserve_structure:
                    dest_preview = dest / relative_path
                else:
                    dest_preview = dest / item_path.name
                print(f"   {type_icon} {item_path} -> {dest_preview}")
            if len(items_to_process) > 10:
                print(f"   ... and {len(items_to_process) - 10} more items")
            return True
        
        # Create batches for parallel processing
        batches = self.create_batches(items_to_process)
        print(f"🔄 Created {len(batches)} batches for {self.workers} workers")
        
        # Start timing
        self.stats['start_time'] = time.time()
        
        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            future_to_batch = {
                executor.submit(
                    self.process_file_batch, 
                    batch, dest, copy_mode, verify, verbose, preserve_structure
                ): i for i, batch in enumerate(batches)
            }
            
            total_dirs = 0
            total_files = 0
            
            for future in as_completed(future_to_batch):
                batch_num = future_to_batch[future]
                try:
                    batch_stats = future.result()
                    self.stats['files_processed'] += batch_stats['processed']
                    self.stats['bytes_transferred'] += batch_stats['bytes']
                    self.stats['errors'] += batch_stats['errors']
                    self.stats['skipped'] += batch_stats['skipped']
                    total_files += batch_stats.get('files', 0)
                    total_dirs += batch_stats.get('dirs', 0)
                    
                    if verbose:
                        progress = (batch_num + 1) / len(batches) * 100
                        print(f"🏃‍♂️ Batch {batch_num + 1}/{len(batches)} complete ({progress:.1f}%)")
                        
                except Exception as e:
                    print(f"❌ Batch {batch_num} failed: {e}")
                    self.stats['errors'] += 1
        
        # Print final statistics
        elapsed = time.time() - self.stats['start_time']
        throughput = self.stats['bytes_transferred'] / (1024**2) / elapsed if elapsed > 0 else 0  # MB/s
        
        action = "copied" if copy_mode else "moved"
        print(f"\n🎯 Operation complete!")
        
        # Build summary string
        summary_parts = []
        if total_files > 0:
            summary_parts.append(f"{total_files} files")
        if total_dirs > 0:
            summary_parts.append(f"{total_dirs} directories")
        
        summary = " and ".join(summary_parts) if summary_parts else f"{self.stats['files_processed']} items"
        
        print(f"   📁 Items {action}: {summary}")
        print(f"   💾 Data transferred: {self.stats['bytes_transferred'] / (1024**3):.2f} GB")
        print(f"   ⏱ Time elapsed: {elapsed:.2f} seconds")
        print(f"   🚀 Throughput: {throughput:.2f} MB/s")
        print(f"   ❌ Errors: {self.stats['errors']}")
        
        return self.stats['errors'] == 0

def main_args(parser):
    parser.add_argument("source", help="Source directory path")
    parser.add_argument("dest", help="Destination directory path")
    parser.add_argument("--recursive", "-r", action="store_true", help="Process files recursively")
    parser.add_argument("--preserve-structure", "-ps", action="store_true", help="Preserve directory structure when using recursive mode")
    parser.add_argument("--include-dirs", "-id", action="store_true", help="Include directories in addition to files")
    parser.add_argument("--copy", "-c", action="store_true", help="Copy files instead of moving")
    parser.add_argument("--filter", "-f", help="Comma-separated file patterns to match (e.g., '*.pdf,*.txt')")
    parser.add_argument("--workers", "-w", type=int, help="Number of worker threads")
    parser.add_argument("--n-workers", "-n", type=int, help="Number of parallel processes (alias for --workers)")
    parser.add_argument("--memory-budget", "-mb", type=int, default=8192, help="Memory budget in MB (default: 8192)")
    parser.add_argument("--verify", "-vf", action="store_true", help="Verify file integrity with hash comparison")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--dry-run", "-d", action="store_true", help="Simulate without actually moving files")
    parser.add_argument("--parallel", "-p", type=int, help="Number of parallel processes (alias for --workers)")

def main(args=None):
    """Main function with argument parsing."""
    if args is None:
        parser = argparse.ArgumentParser(
            description="High-performance file relocator leveraging memory and parallel processing"
        )
        main_args(parser)
        args = parser.parse_args()
    
    # Handle parallel alias
    workers = args.workers or args.parallel or args.n_workers

    # Process patterns
    patterns = [p.strip() for p in args.filter.split(',')] if args.filter else None
    
    # Initialize relocator
    relocator = PowerRelocator(
        workers=workers,
        memory_budget_mb=args.memory_budget
    )
    
    # Run relocation
    success = relocator.relocate(
        source_path=args.source,
        dest_path=args.dest,
        patterns=patterns,
        recursive=args.recursive,
        copy_mode=args.copy,
        verify=args.verify,
        verbose=args.verbose,
        dry_run=args.dry_run,
        preserve_structure=args.preserve_structure,
        include_dirs=args.include_dirs
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())