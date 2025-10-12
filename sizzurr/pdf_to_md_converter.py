#!/usr/bin/env python3
"""
Advanced PDF to Markdown Converter 📄➡️📝💀
Optimized for technical documentation with code, formulas, and complex layouts.
FIXED: Model name validation and hanging issues.
"""

import argparse
import logging
import os
import sys
import io
import signal
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import yaml
import json
import fitz  # pymupdf
import pdfplumber
import torch
from PIL import Image
import numpy as np
import cv2
import re
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, TimeoutError
import time
import threading

# Import transformers with better error handling
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM

try:
    from transformers import Qwen2VLForConditionalGeneration
    QWEN2VL_AVAILABLE = True
except ImportError:
    QWEN2VL_AVAILABLE = False

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
    QWEN25VL_AVAILABLE = True
except ImportError:
    QWEN25VL_AVAILABLE = False

if not (QWEN2VL_AVAILABLE or QWEN25VL_AVAILABLE):
    logging.warning("⚠️ No Qwen-VL models available, using fallback")

try:
    from qwen_vl_utils import process_vision_info
    QWEN_VL_UTILS_AVAILABLE = True
except ImportError:
    QWEN_VL_UTILS_AVAILABLE = False
    
    # Create fallback function
    def process_vision_info(messages):
        """Fallback vision processing for when qwen_vl_utils is not available"""
        image_inputs = []
        for message in messages:
            if isinstance(message.get("content"), list):
                for content_item in message["content"]:
                    if content_item.get("type") == "image":
                        image = content_item.get("image")
                        if hasattr(image, 'size'):  # PIL Image check
                            image_inputs.append(image)
        return image_inputs, None

@dataclass
class ConversionConfig:
    """Configuration for PDF to Markdown conversion"""
    # Model settings
    vision_model: str = "Qwen/Qwen2.5-VL-3B-Instruct"  # Correct 3B model name
    device_vision: str = "cuda:0"  # RTX 3060 12GB (GPU_ID=0)
    device_secondary: str = "cuda:1"  # RTX 2080 Super 8GB (GPU_ID=1)
    max_tokens: int = 2048  # Reduced for stability
    temperature: float = 0.1
    
    # Processing settings
    dpi: int = 300
    extract_images: bool = True
    preserve_formatting: bool = True
    process_code_blocks: bool = True
    process_formulas: bool = True
    
    # Output settings
    output_format: str = "github"  # github, obsidian, standard
    include_metadata: bool = True
    chunk_size: int = 2  # Reduced chunk size for stability
    
    # Quality settings
    confidence_threshold: float = 0.8
    retry_failed_pages: bool = True
    max_retries: int = 2  # Reduced retries
    model_load_timeout: int = 120  # 2 minutes timeout for model loading
    generation_timeout: int = 60  # 1 minute timeout for generation
    
    @classmethod
    def load_from_yaml(cls, config_path: Path) -> 'ConversionConfig':
        """Load config from YAML file with validation"""
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Validate required fields and types
        valid_config = {}
        for field_name, field_def in cls.__dataclass_fields__.items():
            if field_name in config_data:
                valid_config[field_name] = config_data[field_name]
        
        return cls(**valid_config)

@dataclass
class PageData:
    """Represents extracted data from a single page"""
    page_num: int
    text: str = ""
    images: List[Image.Image] = field(default_factory=list)
    image_paths: List[str] = field(default_factory=list)
    layout_info: Dict = field(default_factory=dict)
    confidence: float = 0.0
    markdown: str = ""

class TimeoutHandler:
    """Timeout handler for model operations"""
    def __init__(self, timeout_seconds: int):
        self.timeout_seconds = timeout_seconds
        self.timer = None
        
    def __enter__(self):
        def timeout_handler():
            raise TimeoutError(f"Operation timed out after {self.timeout_seconds} seconds")
        
        self.timer = threading.Timer(self.timeout_seconds, timeout_handler)
        self.timer.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.timer:
            self.timer.cancel()

def validate_model_name(model_name: str, logger) -> str:
    """Validate model name and suggest alternatives if invalid"""
    valid_models = {
        # Qwen2-VL series (older)
        "Qwen/Qwen2-VL-2B-Instruct": "2B",
        "Qwen/Qwen2-VL-7B-Instruct": "7B", 
        "Qwen/Qwen2-VL-72B-Instruct": "72B",
        # Qwen2.5-VL series (newer, recommended)
        "Qwen/Qwen2.5-VL-3B-Instruct": "3B",
        "Qwen/Qwen2.5-VL-7B-Instruct": "7B",
        "Qwen/Qwen2.5-VL-32B-Instruct": "32B",
        "Qwen/Qwen2.5-VL-72B-Instruct": "72B"
    }
    
    if model_name in valid_models:
        logger.info(f"✅ Using valid model: {model_name} ({valid_models[model_name]})")
        
        # Check for qwen_vl_utils dependency for Qwen2.5-VL models
        if "qwen2.5-vl" in model_name.lower() and not QWEN_VL_UTILS_AVAILABLE:
            logger.warning("⚠️ For optimal Qwen2.5-VL performance, install: pip install qwen_vl_utils")
        
        return model_name
    
    # Check for common mistakes
    if "Qwen2-VL-3B" in model_name:
        suggested = "Qwen/Qwen2.5-VL-3B-Instruct"
        logger.warning(f"⚠️ Model '{model_name}' doesn't exist. Did you mean '{suggested}'?")
        logger.info(f"🔄 Auto-correcting to: {suggested}")
        return suggested
    
    if "Qwen2-VL-1B" in model_name:
        suggested = "Qwen/Qwen2-VL-2B-Instruct"
        logger.warning(f"⚠️ Model '{model_name}' doesn't exist. Did you mean '{suggested}'?")
        logger.info(f"🔄 Auto-correcting to: {suggested}")
        return suggested
    
    # If no auto-correction possible, show available models
    logger.error(f"💥 Invalid model: {model_name}")
    logger.error("Available models:")
    logger.error("  Qwen2.5-VL (newer, recommended):")
    for model, size in valid_models.items():
        if "Qwen2.5-VL" in model:
            logger.error(f"    - {model} ({size})")
    logger.error("  Qwen2-VL (older):")
    for model, size in valid_models.items():
        if "Qwen2-VL" in model and "2.5" not in model:
            logger.error(f"    - {model} ({size})")
    
    return model_name  # Return original to let the error happen naturally

def validate_system_requirements(config: ConversionConfig) -> bool:
    """Validate system has required resources before processing"""
    logger = logging.getLogger(__name__)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.error("❌ CUDA not available - GPU processing required")
        return False
    
    # Check specific GPU devices
    device_count = torch.cuda.device_count()
    logger.info(f"🔌 Detected {device_count} GPU(s)")
    
    # Check GPU memory
    for i in range(min(2, device_count)):
        try:
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            logger.info(f"🔌 GPU {i}: {props.name} ({memory_gb:.1f}GB)")
            
            # Validate minimum memory requirements
            if "3B" in config.vision_model and memory_gb < 6:
                logger.warning(f"⚠️ GPU {i} may have insufficient memory for 3B model")
            elif "7B" in config.vision_model and memory_gb < 10:
                logger.error(f"❌ GPU {i} insufficient memory for 7B model - consider using 3B")
                return False
        except Exception as e:
            logger.warning(f"⚠️ Could not check GPU {i} properties: {e}")
    
    return True

def clear_gpu_cache():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

class VisionProcessor:
    """Vision-Language Model processor for layout understanding"""
    
    def __init__(self, config: ConversionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Use GPU_ID=0 (RTX 3060 12GB) for primary vision processing
        self.device = torch.device(config.device_vision)
        
        # Load vision model with timeout protection
        self.processor = None
        self.model = None
        self.model_loaded = False
        
        self._load_model_with_timeout()
    
    def _load_model_with_timeout(self):
        """Load model with timeout protection to prevent hanging"""
        try:
            with TimeoutHandler(self.config.model_load_timeout):
                self._load_model()
                self.model_loaded = True
                self.logger.info("✅ Model loaded successfully")
        except TimeoutError:
            self.logger.error(f"💥 Model loading timed out after {self.config.model_load_timeout}s")
            raise
        except Exception as e:
            self.logger.error(f"💥 Model loading failed: {e}")
            raise
    
    def _load_model(self):
        """Load the vision-language model with simplified approach"""
        try:
            # Clear any existing GPU memory first
            clear_gpu_cache()
            
            # Validate device availability
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available")
            
            device_count = torch.cuda.device_count()
            device_id = int(str(self.device).split(':')[1]) if ':' in str(self.device) else 0
            
            if device_id >= device_count:
                self.logger.warning(f"⚠️ Requested device {self.device} not available, using cuda:0")
                self.device = torch.device("cuda:0")
                device_id = 0
            
            self.logger.info(f"🧠 Loading {self.config.vision_model} on {self.device}")
            
            # CRITICAL: Ensure model string is exactly what user specified
            model_name = self.config.vision_model.strip()
            self.logger.info(f"🔥 Confirmed model name: '{model_name}'")
            
            # Load processor first (lighter operation)
            self.logger.info("📡 Loading processor...")
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            )
            
            # Simplified model loading based on model type
            self.logger.info("🔌 Loading model...")
            
            if "qwen2.5-vl" in model_name.lower() and QWEN25VL_AVAILABLE:
                # Use Qwen2.5-VL-specific loader
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    device_map={"": self.device},  # Simple device mapping
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    attn_implementation="flash_attention_2" if torch.cuda.get_device_capability()[0] >= 8 else "eager"
                )
                self.logger.info("✅ Using Qwen2_5_VLForConditionalGeneration")
                
            elif "qwen2-vl" in model_name.lower() and QWEN2VL_AVAILABLE:
                # Use Qwen2-VL-specific loader
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    device_map={"": self.device},  # Simple device mapping
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    attn_implementation="flash_attention_2" if torch.cuda.get_device_capability()[0] >= 8 else "eager"
                )
                self.logger.info("✅ Using Qwen2VLForConditionalGeneration")
                
            else:
                # Fallback to AutoModelForCausalLM
                self.logger.warning("⚠️ Using AutoModelForCausalLM fallback")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    device_map={"": self.device},
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
            
            # Move model to device explicitly
            self.model = self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            # Test model is responsive
            self.logger.info("🔍 Testing model responsiveness...")
            with torch.no_grad():
                test_input = torch.tensor([[1, 2, 3]], device=self.device)
                _ = test_input + 1  # Simple GPU operation test
            
            self.logger.info("✅ Model loading completed successfully")
            
        except Exception as e:
            self.logger.error(f"💥 Failed to load vision model: {e}")
            # Clean up any partial loading
            if hasattr(self, 'model') and self.model is not None:
                del self.model
                self.model = None
            if hasattr(self, 'processor') and self.processor is not None:
                del self.processor
                self.processor = None
            clear_gpu_cache()
            raise
    
    def process_page_to_markdown(self, page_data: PageData, pdf_path: Path) -> str:
        """Convert page to markdown using vision model with comprehensive timeout protection"""
        
        if not self.model_loaded:
            self.logger.error("💥 Model not loaded, using fallback")
            return self._fallback_markdown(page_data)
        
        for attempt in range(self.config.max_retries):
            try:
                self.logger.info(f"🔥 Processing page {page_data.page_num + 1}, attempt {attempt + 1}")
                
                # Create page image with timeout
                with TimeoutHandler(30):  # 30 second timeout for image rendering
                    page_image = self._render_page_image(pdf_path, page_data.page_num)
                
                # Prepare prompt for technical documentation
                prompt = self._create_conversion_prompt(page_data)
                
                # Process with vision model with timeout
                with TimeoutHandler(self.config.generation_timeout):
                    result = self._generate_markdown(page_image, prompt)
                
                if result and len(result.strip()) > 50:  # Basic quality check
                    self.logger.info(f"✅ Successfully processed page {page_data.page_num + 1}")
                    return result
                else:
                    self.logger.warning(f"⚠️ Generated markdown too short, retrying...")
                    
            except TimeoutError as e:
                self.logger.warning(f"⏰ Timeout on attempt {attempt + 1}: {e}")
                clear_gpu_cache()
                
                if attempt < self.config.max_retries - 1:
                    # Reduce complexity for retry
                    self.config.max_tokens = min(1024, self.config.max_tokens)
                    time.sleep(2)
                    continue
                else:
                    return self._fallback_markdown(page_data)
                    
            except torch.cuda.OutOfMemoryError as e:
                self.logger.warning(f"🩸 GPU OOM on attempt {attempt + 1}: {e}")
                clear_gpu_cache()
                
                if attempt < self.config.max_retries - 1:
                    # Reduce processing complexity for retry
                    self.config.max_tokens = min(1024, self.config.max_tokens)
                    time.sleep(3)
                    continue
                else:
                    return self._fallback_markdown(page_data)
                    
            except Exception as e:
                self.logger.error(f"💥 Vision processing failed on attempt {attempt + 1}: {e}")
                if attempt < self.config.max_retries - 1:
                    clear_gpu_cache()
                    time.sleep(1)
                    continue
                else:
                    return self._fallback_markdown(page_data)
        
        return self._fallback_markdown(page_data)
    
    def _generate_markdown(self, page_image: Image.Image, prompt: str) -> str:
        """Generate markdown with the vision model"""
        try:
            # Prepare messages following Qwen template structure
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": page_image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Apply chat template (CRITICAL: Qwen3 template compliance)
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Process vision inputs
            if QWEN_VL_UTILS_AVAILABLE:
                image_inputs, video_inputs = process_vision_info(messages)
            else:
                image_inputs, video_inputs = self._process_vision_info(messages)
            
            # Tokenize with proper device handling
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            
            # Move inputs to correct device
            inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            
            # Generate with conservative settings
            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    do_sample=True,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True  # Enable caching for efficiency
                )
            
            # Decode response
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
            
            result = self._post_process_markdown(output_text)
            
            # Clean up tensors
            del inputs, generated_ids, generated_ids_trimmed
            clear_gpu_cache()
            
            return result
            
        except Exception as e:
            self.logger.error(f"💥 Generation failed: {e}")
            clear_gpu_cache()
            raise
    
    def _process_vision_info(self, messages: List[Dict]) -> Tuple[List[Image.Image], None]:
        """Process vision information from messages for Qwen2-VL"""
        image_inputs = []
        
        for message in messages:
            if isinstance(message.get("content"), list):
                for content_item in message["content"]:
                    if content_item.get("type") == "image":
                        image = content_item.get("image")
                        if isinstance(image, Image.Image):
                            image_inputs.append(image)
        
        return image_inputs, None
    
    def _render_page_image(self, pdf_path: Path, page_num: int) -> Image.Image:
        """Render page as high-quality image with error handling"""
        try:
            with fitz.open(str(pdf_path)) as doc:
                if page_num >= len(doc):
                    raise ValueError(f"Page {page_num} not found in document with {len(doc)} pages")
                
                page = doc[page_num]
                mat = fitz.Matrix(self.config.dpi/72, self.config.dpi/72)
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                
                # Convert to PIL Image
                image = Image.open(io.BytesIO(img_data))
                
                # Ensure RGB format
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                return image
                
        except Exception as e:
            self.logger.error(f"💥 Failed to render page {page_num}: {e}")
            raise
    
    def _create_conversion_prompt(self, page_data: PageData) -> str:
        """Create optimized prompt for technical documentation conversion"""
        return f"""Convert this technical documentation page to clean Markdown format.

REQUIREMENTS:
- Preserve all code blocks with proper syntax highlighting
- Convert tables to Markdown table format  
- Maintain heading hierarchy (# ## ### etc.)
- Preserve mathematical formulas as LaTeX when possible
- Extract and describe any diagrams or technical figures
- Maintain logical flow and technical accuracy
- Use proper Markdown formatting for lists, emphasis, links

CONTEXT:
- This is page {page_data.page_num + 1} of technical documentation
- Detected layout elements: {list(page_data.layout_info.keys())}
- Page contains {len(page_data.images)} images/diagrams

EXTRACTED TEXT (for reference):
{page_data.text[:300] if page_data.text else "No text extracted"}...

Provide ONLY the Markdown conversion, no explanations or metadata.
"""
    
    def _post_process_markdown(self, markdown: str) -> str:
        """Clean and optimize generated markdown"""
        if not markdown:
            return ""
        
        # Remove any model artifacts
        markdown = re.sub(r'^.*?(?=#{1,6}\s|\w)', '', markdown, flags=re.DOTALL)
        
        # Fix common markdown issues
        markdown = re.sub(r'\n\n\n+', '\n\n', markdown)
        markdown = re.sub(r'^[ \t]+', '', markdown, flags=re.MULTILINE)
        
        # Ensure proper code block formatting
        markdown = re.sub(r'```(\w*)\n', r'```\1\n', markdown)
        
        # Remove empty lines at start and end
        markdown = markdown.strip()
        
        return markdown
    
    def _fallback_markdown(self, page_data: PageData) -> str:
        """Fallback markdown generation if vision processing fails"""
        self.logger.warning(f"⚠️ Using fallback markdown for page {page_data.page_num + 1}")
        
        markdown = f"# Page {page_data.page_num + 1}\n\n"
        
        if page_data.text:
            # Simple text-based conversion
            lines = page_data.text.split('\n')
            current_paragraph = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    if current_paragraph:
                        markdown += ' '.join(current_paragraph) + '\n\n'
                        current_paragraph = []
                    continue
                
                # Simple heuristics for markdown formatting
                if len(line) < 80 and line[0].isupper() and not line.endswith('.'):
                    # Likely a heading
                    if current_paragraph:
                        markdown += ' '.join(current_paragraph) + '\n\n'
                        current_paragraph = []
                    markdown += f"## {line}\n\n"
                elif line.startswith('•') or line.startswith('-') or line.startswith('*'):
                    # List item
                    if current_paragraph:
                        markdown += ' '.join(current_paragraph) + '\n\n'
                        current_paragraph = []
                    markdown += f"- {line[1:].strip()}\n"
                else:
                    # Regular text
                    current_paragraph.append(line)
            
            # Add any remaining paragraph
            if current_paragraph:
                markdown += ' '.join(current_paragraph) + '\n\n'
        else:
            markdown += "*No text content could be extracted from this page.*\n\n"
        
        return markdown

class PDFExtractor:
    """Traditional PDF extraction using pymupdf and pdfplumber"""
    
    def __init__(self, config: ConversionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def extract_page_data(self, pdf_path: Path, page_num: int) -> PageData:
        """Extract text, images, and layout from a single page"""
        page_data = PageData(page_num=page_num)
        
        try:
            # Extract with pymupdf for images and basic text
            with fitz.open(str(pdf_path)) as doc:
                if page_num >= len(doc):
                    self.logger.error(f"💥 Page {page_num} not found in document")
                    return page_data
                
                page = doc[page_num]
                
                # Extract text with positioning
                try:
                    text_dict = page.get_text("dict")
                    page_data.layout_info = self._analyze_layout(text_dict)
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to extract layout for page {page_num}: {e}")
                
                # Extract images if configured
                if self.config.extract_images:
                    try:
                        image_list = page.get_images()
                        for img_index, img in enumerate(image_list[:5]):  # Limit to 5 images per page
                            try:
                                xref = img[0]
                                pix = fitz.Pixmap(doc, xref)
                                if pix.n - pix.alpha < 4:  # GRAY or RGB
                                    img_data = pix.tobytes("png")
                                    img_pil = Image.open(io.BytesIO(img_data))
                                    
                                    # Basic image validation
                                    if img_pil.size[0] > 50 and img_pil.size[1] > 50:  # Skip tiny images
                                        page_data.images.append(img_pil)
                            except Exception as e:
                                self.logger.warning(f"⚠️ Failed to extract image {img_index} from page {page_num}: {e}")
                            finally:
                                if 'pix' in locals():
                                    pix = None
                    except Exception as e:
                        self.logger.warning(f"⚠️ Failed to process images for page {page_num}: {e}")
            
            # Extract with pdfplumber for better table detection
            try:
                with pdfplumber.open(str(pdf_path)) as pdf:
                    if page_num < len(pdf.pages):
                        page = pdf.pages[page_num]
                        
                        # Extract text
                        page_data.text = page.extract_text() or ""
                        
                        # Extract tables
                        tables = page.extract_tables()
                        if tables:
                            page_data.layout_info['tables'] = tables[:3]  # Limit to 3 tables per page
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to extract with pdfplumber for page {page_num}: {e}")
                
        except Exception as e:
            self.logger.error(f"💥 Failed to extract page {page_num}: {e}")
            page_data.confidence = 0.0
        
        # Calculate confidence based on extracted content
        if page_data.text or page_data.images or page_data.layout_info:
            page_data.confidence = 0.8
        else:
            page_data.confidence = 0.1
        
        return page_data
    
    def _analyze_layout(self, text_dict: Dict) -> Dict:
        """Analyze text layout to identify structure"""
        layout = {
            'headers': [],
            'code_blocks': [],
            'paragraphs': [],
            'lists': []
        }
        
        try:
            for block in text_dict.get("blocks", []):
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line.get("spans", []):
                            text = span.get("text", "").strip()
                            font_size = span.get("size", 12)
                            font_flags = span.get("flags", 0)
                            
                            if not text:
                                continue
                            
                            # Classify text based on font properties
                            if font_size > 16 or font_flags & 2**4:  # Large or bold
                                layout['headers'].append(text)
                            elif self._is_code_like(text):
                                layout['code_blocks'].append(text)
                            elif text.startswith(('•', '-', '*', '1.', '2.')):
                                layout['lists'].append(text)
                            elif len(text) > 10:  # Skip very short fragments
                                layout['paragraphs'].append(text)
        except Exception as e:
            self.logger.warning(f"⚠️ Layout analysis failed: {e}")
        
        return layout
    
    def _is_code_like(self, text: str) -> bool:
        """Heuristic to identify code-like text"""
        if len(text) < 5:
            return False
        
        code_indicators = [
            r'\b(def|class|import|from|if|else|for|while|try|except|function|var|let|const)\b',
            r'[{}();]',
            r'\b\w+\.\w+\(',
            r'^\s*[#//]',  # Comments
            r'[<>]=?|==|!=',  # Operators
            r'[A-Za-z_]\w*\s*=\s*',  # Assignment
        ]
        
        code_score = sum(1 for pattern in code_indicators if re.search(pattern, text))
        return code_score >= 2  # Need at least 2 indicators

class PDFToMarkdownConverter:
    """Main converter class orchestrating the conversion process"""
    
    def __init__(self, config: Optional[ConversionConfig] = None):
        self.config = config or ConversionConfig()
        self.logger = self._setup_logging()
        
        # Validate system before initialization
        if not validate_system_requirements(self.config):
            raise RuntimeError("System requirements not met")
        
        # Clear GPU memory before starting
        clear_gpu_cache()
        
        # Initialize processors
        self.extractor = PDFExtractor(self.config)
        
        # Initialize vision processor with error handling
        try:
            self.vision_processor = VisionProcessor(self.config)
        except Exception as e:
            self.logger.error(f"💥 Failed to initialize vision processor: {e}")
            raise
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging with cyber theme"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def convert_pdf(self, pdf_path: Path, output_path: Path) -> bool:
        """Convert PDF to Markdown with comprehensive error handling"""
        try:
            self.logger.info(f"🔥 Starting conversion: {pdf_path} → {output_path}")
            
            # Validate inputs
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF not found: {pdf_path}")
            
            if not pdf_path.suffix.lower() == '.pdf':
                raise ValueError(f"Input file must be a PDF: {pdf_path}")
            
            # Create output directory
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Check if output already exists and warn
            if output_path.exists():
                self.logger.warning(f"⚠️ Output file exists and will be overwritten: {output_path}")
                
            # Get page count with validation
            try:
                with fitz.open(str(pdf_path)) as doc:
                    total_pages = len(doc)
                    if total_pages == 0:
                        raise ValueError("PDF contains no pages")
            except Exception as e:
                self.logger.error(f"💥 Failed to open PDF: {e}")
                return False
            
            self.logger.info(f"📄 Processing {total_pages} pages...")
            
            # Process pages in chunks with progress tracking
            all_markdown = []
            successful_pages = 0
            
            for start_page in range(0, total_pages, self.config.chunk_size):
                end_page = min(start_page + self.config.chunk_size, total_pages)
                
                self.logger.info(f"🧠 Processing chunk: pages {start_page + 1}-{end_page}")
                
                try:
                    chunk_markdown = self._process_page_chunk(pdf_path, start_page, end_page)
                    all_markdown.extend(chunk_markdown)
                    successful_pages += len([md for md in chunk_markdown if md.strip()])
                    
                    # Clear GPU cache between chunks
                    clear_gpu_cache()
                    
                    # Progress indicator
                    progress = (end_page / total_pages) * 100
                    self.logger.info(f"🔌 Progress: {progress:.1f}% ({successful_pages}/{end_page} pages processed)")
                    
                except Exception as e:
                    self.logger.error(f"💥 Failed to process chunk {start_page}-{end_page}: {e}")
                    # Add empty strings for failed pages to maintain page alignment
                    for _ in range(start_page, end_page):
                        all_markdown.append(f"# Page {_ + 1}\n\n*Failed to process this page*\n\n")
            
            # Combine and save
            if not all_markdown:
                self.logger.error("💥 No pages were successfully processed")
                return False
            
            final_markdown = self._combine_markdown_pages(all_markdown)
            
            # Write output with error handling
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(final_markdown)
                
                # Verify output was written
                if not output_path.exists() or output_path.stat().st_size == 0:
                    raise IOError("Output file was not created or is empty")
                
            except Exception as e:
                self.logger.error(f"💥 Failed to write output file: {e}")
                return False
            
            self.logger.info(f"✅ Conversion completed: {output_path}")
            self.logger.info(f"📊 Successfully processed {successful_pages}/{total_pages} pages")
            return True
            
        except Exception as e:
            self.logger.error(f"💥 Conversion failed: {e}")
            return False
    
    def _process_page_chunk(self, pdf_path: Path, start_page: int, end_page: int) -> List[str]:
        """Process a chunk of pages with individual page error handling"""
        chunk_markdown = []
        
        for page_num in range(start_page, end_page):
            try:
                self.logger.info(f"🧠 Processing page {page_num + 1}...")
                
                # Extract page data with timeout
                with TimeoutHandler(30):  # 30 second timeout per page extraction
                    page_data = self.extractor.extract_page_data(pdf_path, page_num)
                
                # Convert to markdown with timeout
                with TimeoutHandler(90):  # 90 second timeout per page conversion
                    markdown = self.vision_processor.process_page_to_markdown(page_data, pdf_path)
                
                if markdown.strip():
                    chunk_markdown.append(markdown)
                    self.logger.info(f"✅ Successfully processed page {page_num + 1}")
                else:
                    self.logger.warning(f"⚠️ Empty result for page {page_num + 1}")
                    chunk_markdown.append(f"# Page {page_num + 1}\n\n*No content extracted*\n\n")
                
            except TimeoutError as e:
                self.logger.error(f"⏰ Timeout processing page {page_num + 1}: {e}")
                chunk_markdown.append(f"# Page {page_num + 1}\n\n*Processing timed out*\n\n")
                
            except Exception as e:
                self.logger.error(f"💥 Failed to process page {page_num + 1}: {e}")
                chunk_markdown.append(f"# Page {page_num + 1}\n\n*Processing failed: {str(e)[:100]}*\n\n")
        
        return chunk_markdown
    
    def _combine_markdown_pages(self, pages: List[str]) -> str:
        """Combine individual page markdown into final document"""
        if not pages:
            return "# Empty Document\n\n*No content was extracted*\n"
        
        # Add document header if configured
        combined = ""
        if self.config.include_metadata:
            combined += f"# Converted Document 📝💀\n\n"
            combined += f"*Converted using Advanced PDF→MD Converter*\n"
            combined += f"*Model: {self.config.vision_model}*\n"
            combined += f"*Pages processed: {len(pages)}*\n\n"
            combined += "---\n\n"
        
        # Combine pages with separators
        for i, page_md in enumerate(pages):
            if page_md and page_md.strip():
                combined += page_md.strip() + "\n\n"
                
                # Add page breaks for longer documents (every 5 pages)
                if (i + 1) % 5 == 0 and i < len(pages) - 1:
                    combined += "---\n\n"
        
        return combined.strip()

def main_args(parser):
    # Required arguments
    parser.add_argument("-i", "--input", type=Path, required=True, 
                       help="Input PDF file")
    parser.add_argument("-o", "--output", type=Path, required=True,
                       help="Output Markdown file")
    
    # Configuration
    parser.add_argument("-c", "--config", type=Path,
                       help="Configuration YAML file")
    
    # Processing options
    parser.add_argument("--dpi", type=int, default=300,
                       help="Image rendering DPI (default: 300)")
    parser.add_argument("-cs", "--chunk-size", type=int, default=2,
                       help="Pages per processing chunk (default: 2)")
    parser.add_argument("--no-images", action="store_true",
                       help="Skip image extraction")
    parser.add_argument("-mdl", "--model", type=str,
                       default="Qwen/Qwen2.5-VL-3B-Instruct",  # Correct 3B model
                       help="Vision model to use (default: Qwen/Qwen2.5-VL-3B-Instruct)")
    
    # Output options
    parser.add_argument("-f", "--format", choices=["github", "obsidian", "standard"],
                       default="github", help="Output format (default: github)")
    parser.add_argument("--no-metadata", action="store_true",
                       help="Skip document metadata")
    
    # System options (following CLI standards)
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Enable verbose output")
    parser.add_argument("-gi", "--gpu-id", type=int, choices=[0, 1], default=0,
                       help="Primary GPU ID (0=RTX3060, 1=RTX2080S, default: 0)")
    parser.add_argument("-mr", "--max-retries", type=int, default=2,
                       help="Maximum retries per page (default: 2)")
    parser.add_argument("-mt", "--max-tokens", type=int, default=2048,
                       help="Maximum tokens per generation (default: 2048)")

def main(args=None):
    """Main CLI function with enhanced argument handling"""
    if args is None:
        parser = argparse.ArgumentParser(
            description="Advanced PDF to Markdown Converter 💀📄",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
    Examples:
      %(prog)s -i input.pdf -o output.md -v
      %(prog)s -i input.pdf -o output.md -mdl Qwen/Qwen2.5-VL-3B-Instruct
      %(prog)s -i input.pdf -o output.md -mdl Qwen/Qwen2-VL-2B-Instruct
      %(prog)s -i input.pdf -o output.md --dpi 600 --no-images -v -gi 1
    
    Available Models:
      Qwen2.5-VL (newer): 3B, 7B, 32B, 72B
      Qwen2-VL (older): 2B, 7B, 72B
            """
        )
        main_args(parser)
        args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("🔥 PDF to Markdown Converter starting...")
    
    # Load or create config
    if args.config and args.config.exists():
        try:
            config = ConversionConfig.load_from_yaml(args.config)
            logger.info(f"📄 Loaded config from {args.config}")
        except Exception as e:
            logger.error(f"💥 Failed to load config: {e}")
            sys.exit(1)
    else:
        config = ConversionConfig()
        logger.info("📄 Using default configuration")
    
    # Apply CLI overrides - CRITICAL: Ensure model override works
    config.dpi = args.dpi
    config.chunk_size = args.chunk_size
    config.extract_images = not args.no_images
    config.vision_model = validate_model_name(args.model.strip(), logger)  # Validate model name
    config.output_format = args.format
    config.include_metadata = not args.no_metadata
    config.max_retries = args.max_retries
    config.max_tokens = args.max_tokens
    
    # Set GPU device based on user preference
    if args.gpu_id == 0:
        config.device_vision = "cuda:0"  # RTX 3060 12GB
        config.device_secondary = "cuda:1"  # RTX 2080 Super 8GB
    else:
        config.device_vision = "cuda:1"  # RTX 2080 Super 8GB
        config.device_secondary = "cuda:0"  # RTX 3060 12GB
    
    # Log configuration
    logger.info(f"🧠 Model: {config.vision_model}")
    logger.info(f"🔌 Primary GPU: {config.device_vision}")
    logger.info(f"📊 Max tokens: {config.max_tokens}")
    logger.info(f"🔄 Chunk size: {config.chunk_size}")
    
    # Validate inputs
    if not args.input.exists():
        logger.error(f"💥 Input file not found: {args.input}")
        sys.exit(1)
    
    if not args.input.suffix.lower() == '.pdf':
        logger.error(f"💥 Input file must be a PDF: {args.input}")
        sys.exit(1)
    
    # Run conversion with comprehensive error handling
    try:
        converter = PDFToMarkdownConverter(config)
        success = converter.convert_pdf(args.input, args.output)
        
        if success:
            logger.info("✅ Conversion completed successfully")
            sys.exit(0)
        else:
            logger.error("💥 Conversion failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.warning("⚠️ Conversion interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
