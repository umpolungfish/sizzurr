#!/usr/bin/env python3
"""
Advanced HTML to Markdown Converter
Optimized for RAG database creation with intelligent content extraction
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import json
from datetime import datetime

try:
    from bs4 import BeautifulSoup, NavigableString, Tag
    import html2text
    import markdownify
except ImportError as e:
    print(f"⚠️  Missing dependency: {e}")
    print("Install with: pip install beautifulsoup4 html2text markdownify")
    exit(1)

@dataclass
class ConversionResult:
    markdown_content: str
    metadata: Dict
    word_count: int
    code_blocks: int
    links: int

class HTMLToMarkdownConverter:
    def __init__(self, preserve_structure: bool = True, clean_output: bool = True):
        self.preserve_structure = preserve_structure
        self.clean_output = clean_output
        self.setup_html2text()
    
    def setup_html2text(self):
        """Configure html2text converter with code-friendly settings"""
        self.h2t = html2text.HTML2Text()
        self.h2t.ignore_links = False
        self.h2t.ignore_images = False
        self.h2t.ignore_emphasis = False
        self.h2t.body_width = 0  # No line wrapping (CRITICAL for code)
        self.h2t.unicode_snob = True
        self.h2t.skip_internal_links = False
        self.h2t.inline_links = False  # Use reference-style links
        self.h2t.mark_code = False  # Don't let html2text handle code - we do it manually
        self.h2t.wrap_links = False
        self.h2t.wrap_list_items = False
        self.h2t.decode_errors = 'ignore'
        self.h2t.default_image_alt = ''
        # Preserve spacing in preformatted text
        self.h2t.pad_tables = True
    
    def extract_metadata(self, soup: BeautifulSoup, file_path: Path) -> Dict:
        """Extract metadata from HTML document"""
        metadata = {
            'source_file': str(file_path),
            'title': '',
            'description': '',
            'keywords': [],
            'url': '',
            'processed_date': datetime.now().isoformat(),
        }
        
        # Extract title
        title_tag = soup.find('title')
        if title_tag:
            metadata['title'] = title_tag.get_text().strip()
        
        # Extract meta tags
        meta_tags = soup.find_all('meta')
        for meta in meta_tags:
            name = meta.get('name', '').lower()
            content = meta.get('content', '')
            
            if name == 'description':
                metadata['description'] = content
            elif name == 'keywords':
                metadata['keywords'] = [k.strip() for k in content.split(',') if k.strip()]
            elif name == 'author':
                metadata['author'] = content
        
        # Extract canonical URL
        canonical = soup.find('link', {'rel': 'canonical'})
        if canonical:
            metadata['url'] = canonical.get('href', '')
        
        return metadata
    
    def clean_html_structure(self, soup: BeautifulSoup) -> BeautifulSoup:
        """Remove unwanted elements and clean structure"""
        # Remove unwanted elements
        unwanted_tags = [
            'script', 'style', 'noscript', 'iframe', 'embed', 'object',
            'nav', 'header', 'footer', 'aside', 'advertisement', 'ads'
        ]
        
        for tag_name in unwanted_tags:
            for tag in soup.find_all(tag_name):
                tag.decompose()
        
        # Remove elements with common ad/navigation classes
        unwanted_classes = [
            'nav', 'navigation', 'navbar', 'header', 'footer', 'sidebar',
            'ad', 'advertisement', 'ads', 'banner', 'promo', 'social',
            'breadcrumb', 'pagination', 'comment', 'related'
        ]
        
        for class_name in unwanted_classes:
            for element in soup.find_all(class_=re.compile(class_name, re.I)):
                element.decompose()
        
        # Remove elements with common ad IDs
        unwanted_ids = ['header', 'footer', 'nav', 'sidebar', 'ads']
        for id_name in unwanted_ids:
            element = soup.find(id=re.compile(id_name, re.I))
            if element:
                element.decompose()
        
        return soup
    
    def find_main_content(self, soup: BeautifulSoup) -> Tag:
        """Identify and extract main content area"""
        # Try common content containers in order of preference
        content_selectors = [
            'main',
            'article', 
            '[role="main"]',
            '.content',
            '.main-content',
            '.post-content',
            '.entry-content',
            '#content',
            '#main',
            '.container .content',
            'body'
        ]
        
        for selector in content_selectors:
            content = soup.select_one(selector)
            if content and len(content.get_text().strip()) > 100:  # Ensure substantial content
                return content
        
        # Fallback to body
        return soup.find('body') or soup
    
    def preserve_code_blocks(self, soup: BeautifulSoup) -> BeautifulSoup:
        """Enhanced code block preservation with proper formatting"""
        import html
        
        # First, protect all code content by encoding it
        code_blocks = []
        
        # Handle pre tags (with or without code children)
        for i, pre in enumerate(soup.find_all('pre')):
            placeholder = f"CODEBLOCK_PLACEHOLDER_{i}"
            
            # Get the original text content preserving whitespace
            if pre.find('code'):
                code_element = pre.find('code')
                # Extract language from classes
                lang = self._extract_language_from_classes(code_element.get('class', []))
                # Get raw text content, preserving all whitespace
                raw_content = self._get_raw_text_content(code_element)
            else:
                lang = self._extract_language_from_classes(pre.get('class', []))
                raw_content = self._get_raw_text_content(pre)
            
            # Store the code block info
            code_blocks.append({
                'placeholder': placeholder,
                'content': raw_content,
                'language': lang
            })
            
            # Replace with placeholder
            pre.replace_with(soup.new_string(placeholder))
        
        # Handle standalone code tags (inline code)
        for i, code in enumerate(soup.find_all('code')):
            # Skip if this was already handled as part of a pre tag
            if code.find_parent('pre'):
                continue
                
            placeholder = f"INLINECODE_PLACEHOLDER_{i}"
            raw_content = self._get_raw_text_content(code)
            
            code_blocks.append({
                'placeholder': placeholder,
                'content': raw_content,
                'language': 'inline'
            })
            
            code.replace_with(soup.new_string(placeholder))
        
        # Store code blocks for later restoration
        soup._preserved_code_blocks = code_blocks
        
        return soup
    
    def _extract_language_from_classes(self, classes: list) -> str:
        """Extract programming language from CSS classes"""
        if not classes:
            return ''
            
        for cls in classes:
            cls = str(cls).lower()
            # Common patterns for language specification
            if cls.startswith('language-'):
                return cls.replace('language-', '')
            elif cls.startswith('lang-'):
                return cls.replace('lang-', '')
            elif cls.startswith('highlight-'):
                return cls.replace('highlight-', '')
            elif cls in ['cpp', 'c++', 'cxx']:
                return 'cpp'
            elif cls in ['js', 'javascript']:
                return 'javascript'
            elif cls in ['py', 'python']:
                return 'python'
            elif cls in ['cs', 'csharp', 'c#']:
                return 'csharp'
            elif cls in ['sh', 'bash', 'shell']:
                return 'bash'
            elif cls in ['asm', 'assembly', 'nasm']:
                return 'asm'
            elif cls in ['ps1', 'powershell']:
                return 'powershell'
            elif cls in ['sql']:
                return 'sql'
            elif cls in ['xml', 'html']:
                return cls
        
        return ''
    
    def _get_raw_text_content(self, element) -> str:
        """Extract text content preserving all whitespace and formatting"""
        # Get all text nodes, preserving whitespace
        text_parts = []
        
        for node in element.descendants:
            if isinstance(node, NavigableString) and node.parent.name not in ['script', 'style']:
                text_parts.append(str(node))
        
        # Join without modifying whitespace
        raw_text = ''.join(text_parts)
        
        # Decode HTML entities but preserve structure
        import html
        raw_text = html.unescape(raw_text)
        
        # Normalize line endings but preserve indentation
        lines = raw_text.splitlines()
        
        # Remove common leading whitespace while preserving relative indentation
        if lines:
            # Find minimum indentation (excluding empty lines)
            non_empty_lines = [line for line in lines if line.strip()]
            if non_empty_lines:
                min_indent = min(len(line) - len(line.lstrip()) for line in non_empty_lines)
                # Remove common indentation
                lines = [line[min_indent:] if len(line) > min_indent else line for line in lines]
        
        return '\n'.join(lines).strip()
    
    def restore_code_blocks(self, markdown_content: str, code_blocks: list) -> str:
        """Restore properly formatted code blocks in markdown"""
        for block in code_blocks:
            placeholder = block['placeholder']
            content = block['content']
            language = block['language']
            
            if language == 'inline':
                # Restore inline code
                replacement = f"`{content}`"
            else:
                # Restore code blocks
                if language:
                    replacement = f"```{language}\n{content}\n```"
                else:
                    replacement = f"```\n{content}\n```"
            
            markdown_content = markdown_content.replace(placeholder, replacement)
        
        return markdown_content
    
    def enhance_tables(self, soup: BeautifulSoup) -> BeautifulSoup:
        """Improve table conversion to markdown"""
        for table in soup.find_all('table'):
            # Add table classes for better markdown conversion
            if not table.get('class'):
                table['class'] = ['markdown-table']
        
        return soup
    
    def convert_file(self, html_file: Path, output_file: Optional[Path] = None) -> ConversionResult:
        """Convert a single HTML file to markdown"""
        print(f"🔄 Converting: {html_file.name}")
        
        # Read HTML file
        try:
            with open(html_file, 'r', encoding='utf-8', errors='ignore') as f:
                html_content = f.read()
        except Exception as e:
            print(f"❌ Error reading {html_file}: {e}")
            return None
        
        # Parse HTML
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract metadata
        metadata = self.extract_metadata(soup, html_file)
        
        if self.clean_output:
            # Clean unwanted elements
            soup = self.clean_html_structure(soup)
            
            # Focus on main content
            main_content = self.find_main_content(soup)
            soup = BeautifulSoup(str(main_content), 'html.parser')
        
        # CRITICAL: Preserve code blocks BEFORE any other processing
        preserved_code_blocks = []
        if self.preserve_structure:
            soup = self.preserve_code_blocks(soup)
            preserved_code_blocks = getattr(soup, '_preserved_code_blocks', [])
            soup = self.enhance_tables(soup)
        
        # Convert to markdown using html2text
        markdown_content = self.h2t.handle(str(soup))
        
        # CRITICAL: Restore code blocks BEFORE post-processing
        if preserved_code_blocks:
            markdown_content = self.restore_code_blocks(markdown_content, preserved_code_blocks)
        
        # Post-process markdown (but avoid breaking restored code)
        markdown_content = self.post_process_markdown(markdown_content)
        
        # Add metadata header
        if metadata['title'] or metadata['description']:
            header_parts = ['---']
            if metadata['title']:
                header_parts.append(f"title: {metadata['title']}")
            if metadata['description']:
                header_parts.append(f"description: {metadata['description']}")
            if metadata['url']:
                header_parts.append(f"source_url: {metadata['url']}")
            header_parts.append(f"source_file: {metadata['source_file']}")
            header_parts.append(f"processed_date: {metadata['processed_date']}")
            header_parts.append('---\n')
            
            markdown_content = '\n'.join(header_parts) + '\n' + markdown_content
        
        # Calculate stats
        word_count = len(markdown_content.split())
        code_blocks = markdown_content.count('```')
        links = markdown_content.count('[')
        
        # Write output
        if not output_file:
            output_file = html_file.with_suffix('.md')
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            print(f"✅ Saved: {output_file}")
        except Exception as e:
            print(f"❌ Error writing {output_file}: {e}")
            return None
        
        return ConversionResult(
            markdown_content=markdown_content,
            metadata=metadata,
            word_count=word_count,
            code_blocks=code_blocks // 2,  # Divide by 2 since we count opening and closing
            links=links
        )
    
    def post_process_markdown(self, markdown: str) -> str:
        """Clean up and optimize markdown output while preserving code blocks"""
        # First, protect code blocks from post-processing
        import re
        
        # Find all code blocks and replace with placeholders
        code_block_pattern = r'```[\s\S]*?```'
        code_blocks = re.findall(code_block_pattern, markdown)
        
        # Replace code blocks with placeholders
        protected_markdown = markdown
        for i, block in enumerate(code_blocks):
            placeholder = f"PROTECTED_CODE_BLOCK_{i}"
            protected_markdown = protected_markdown.replace(block, placeholder, 1)
        
        # Now safe to do post-processing on non-code content
        
        # Remove excessive blank lines (but not in code blocks)
        protected_markdown = re.sub(r'\n\s*\n\s*\n', '\n\n', protected_markdown)
        
        # Clean up list formatting
        protected_markdown = re.sub(r'\n(\s*[-*+]\s)', r'\n\1', protected_markdown)
        
        # Fix header spacing (avoid affecting code)
        protected_markdown = re.sub(r'(#{1,6})\s*([^\n]+)', r'\1 \2', protected_markdown)
        
        # Remove trailing whitespace from lines
        lines = [line.rstrip() for line in protected_markdown.split('\n')]
        protected_markdown = '\n'.join(lines)
        
        # Restore code blocks
        for i, block in enumerate(code_blocks):
            placeholder = f"PROTECTED_CODE_BLOCK_{i}"
            protected_markdown = protected_markdown.replace(placeholder, block)
        
        return protected_markdown.strip()

class BatchConverter:
    def __init__(self, converter: HTMLToMarkdownConverter):
        self.converter = converter
        self.stats = {
            'processed': 0,
            'successful': 0,
            'failed': 0,
            'total_words': 0,
            'total_code_blocks': 0,
        }
    
    def convert_directory(self, input_dir: Path, output_dir: Optional[Path] = None, 
                         pattern: str = "*.html", combine: bool = False) -> Dict:
        """Convert all HTML files in a directory"""
        print(f"🔥 Starting batch conversion: {input_dir}")
        
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
        
        html_files = list(input_dir.rglob(pattern))
        
        if not html_files:
            print(f"⚠️  No HTML files found in {input_dir}")
            return self.stats
        
        print(f"📄 Found {len(html_files)} HTML files")
        
        if combine:
            return self._convert_combined(html_files, input_dir, output_dir)
        else:
            return self._convert_separate(html_files, input_dir, output_dir)
    
    def _convert_separate(self, html_files: List[Path], input_dir: Path, output_dir: Optional[Path]) -> Dict:
        """Convert files to separate markdown files"""
        for html_file in html_files:
            self.stats['processed'] += 1
            
            # Determine output file
            if output_dir:
                # Preserve directory structure
                rel_path = html_file.relative_to(input_dir)
                output_file = output_dir / rel_path.with_suffix('.md')
                output_file.parent.mkdir(parents=True, exist_ok=True)
            else:
                output_file = html_file.with_suffix('.md')
            
            # Convert file
            result = self.converter.convert_file(html_file, output_file)
            
            if result:
                self.stats['successful'] += 1
                self.stats['total_words'] += result.word_count
                self.stats['total_code_blocks'] += result.code_blocks
            else:
                self.stats['failed'] += 1
        
        self.print_summary()
        return self.stats
    
    def _convert_combined(self, html_files: List[Path], input_dir: Path, output_dir: Optional[Path]) -> Dict:
        """Convert all files into one combined markdown file"""
        print("📚 Creating combined markdown file...")
        
        # Determine output file location
        if output_dir:
            combined_file = output_dir / "combined_documentation.md"
        else:
            combined_file = input_dir / "combined_documentation.md"
        
        # Sort files for consistent ordering
        html_files.sort()
        
        combined_content = []
        toc_entries = []
        
        # Add header and metadata
        combined_content.append("---")
        combined_content.append(f"title: Combined Documentation")
        combined_content.append(f"source_directory: {input_dir}")
        combined_content.append(f"total_files: {len(html_files)}")
        combined_content.append(f"generated_date: {datetime.now().isoformat()}")
        combined_content.append("---")
        combined_content.append("")
        combined_content.append("# Combined Documentation")
        combined_content.append("")
        combined_content.append(f"This document combines {len(html_files)} HTML files from `{input_dir}`")
        combined_content.append("")
        
        # Add table of contents placeholder
        toc_placeholder = "<!-- TABLE_OF_CONTENTS -->"
        combined_content.append("## Table of Contents")
        combined_content.append("")
        combined_content.append(toc_placeholder)
        combined_content.append("")
        combined_content.append("---")
        combined_content.append("")
        
        # Process each file
        for i, html_file in enumerate(html_files, 1):
            print(f"🔄 Processing [{i}/{len(html_files)}]: {html_file.name}")
            self.stats['processed'] += 1
            
            try:
                # Convert file to get content and metadata
                result = self.converter.convert_file(html_file)
                
                if result:
                    self.stats['successful'] += 1
                    self.stats['total_words'] += result.word_count
                    self.stats['total_code_blocks'] += result.code_blocks
                    
                    # Extract title from metadata or filename
                    title = result.metadata.get('title', html_file.stem.replace('_', ' ').replace('-', ' ').title())
                    
                    # Create section anchor
                    anchor = self._create_anchor(title)
                    toc_entries.append(f"- [{title}](#{anchor})")
                    
                    # Add document separator
                    combined_content.append(f"## {title} {{#{anchor}}}")
                    combined_content.append("")
                    combined_content.append(f"**Source:** `{html_file.relative_to(input_dir)}`")
                    
                    # Add metadata if available
                    if result.metadata.get('description'):
                        combined_content.append(f"**Description:** {result.metadata['description']}")
                    if result.metadata.get('url'):
                        combined_content.append(f"**URL:** {result.metadata['url']}")
                    
                    combined_content.append("")
                    
                    # Add the actual content (strip YAML frontmatter if present)
                    content = result.markdown_content
                    if content.startswith('---'):
                        # Remove YAML frontmatter
                        lines = content.split('\n')
                        start_idx = 0
                        end_idx = 0
                        for i, line in enumerate(lines[1:], 1):
                            if line.strip() == '---':
                                end_idx = i
                                break
                        if end_idx > 0:
                            content = '\n'.join(lines[end_idx + 1:]).strip()
                    
                    combined_content.append(content)
                    combined_content.append("")
                    combined_content.append("---")
                    combined_content.append("")
                    
                    # Clean up the temporary file if it was created
                    temp_md = html_file.with_suffix('.md')
                    if temp_md.exists():
                        temp_md.unlink()
                        
                else:
                    self.stats['failed'] += 1
                    print(f"❌ Failed to convert: {html_file}")
                    
            except Exception as e:
                self.stats['failed'] += 1
                print(f"❌ Error processing {html_file}: {e}")
        
        # Replace TOC placeholder with actual TOC
        final_content = '\n'.join(combined_content)
        toc_content = '\n'.join(toc_entries)
        final_content = final_content.replace(toc_placeholder, toc_content)
        
        # Write combined file
        try:
            with open(combined_file, 'w', encoding='utf-8') as f:
                f.write(final_content)
            print(f"✅ Combined file saved: {combined_file}")
            print(f"📊 File size: {combined_file.stat().st_size / 1024 / 1024:.2f} MB")
        except Exception as e:
            print(f"❌ Error writing combined file: {e}")
        
        self.print_summary()
        return self.stats
    
    def _create_anchor(self, title: str) -> str:
        """Create URL-safe anchor from title"""
        # Convert to lowercase, replace spaces and special chars with hyphens
        anchor = re.sub(r'[^\w\s-]', '', title.lower())
        anchor = re.sub(r'[-\s]+', '-', anchor)
        return anchor.strip('-')
    
    def print_summary(self):
        """Print conversion summary"""
        print(f"\n🎯 Conversion Summary:")
        print(f"   📄 Files processed: {self.stats['processed']}")
        print(f"   ✅ Successful: {self.stats['successful']}")
        print(f"   ❌ Failed: {self.stats['failed']}")
        print(f"   📝 Total words: {self.stats['total_words']:,}")
        print(f"   💻 Code blocks: {self.stats['total_code_blocks']}")

def main_args(parser):
    parser.add_argument('input', type=Path, help="Input HTML file or directory")
    parser.add_argument('--output', '-o', type=Path, help="Output file or directory")
    parser.add_argument('--preserve-structure', action='store_true', default=True,
                       help="Preserve document structure")
    parser.add_argument('--clean', action='store_true', default=True,
                       help="Clean unwanted elements")
    parser.add_argument('--pattern', default="*.html",
                       help="File pattern for batch processing")
    parser.add_argument('--combine', '-c', action='store_true',
                       help="Combine all files into single markdown document")
    parser.add_argument('--verbose', '-v', action='store_true',
                       help="Enable verbose output")

def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser(description="Convert HTML files to Markdown for RAG")
        main_args(parser)
        args = parser.parse_args()
    
    # Create converter
    converter = HTMLToMarkdownConverter(
        preserve_structure=args.preserve_structure,
        clean_output=args.clean
    )
    
    # Process input
    if args.input.is_file():
        # Single file conversion
        result = converter.convert_file(args.input, args.output)
        if result and args.verbose:
            print(f"📊 Words: {result.word_count}, Code blocks: {result.code_blocks}")
    
    elif args.input.is_dir():
        # Batch conversion
        batch_converter = BatchConverter(converter)
        batch_converter.convert_directory(args.input, args.output, args.pattern, args.combine)
    
    else:
        print(f"❌ Input path not found: {args.input}")

if __name__ == "__main__":
    main()
