# BIZRA Agentic Cleaner Module
# Implements "Code-Synthesis" Data Cleaning Paradigm (IEEE TKDE 2025 Survey)
# Part of BIZRA Node0 Intelligence Layer
#
# SECURITY: This module now uses AST validation to prevent arbitrary code execution
# from LLM-generated cleaning functions. See _validate_cleaning_code().

import os
import json
import logging
import ast
import textwrap
import importlib.util
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
import pandas as pd
from tqdm import tqdm

# Import BIZRA ecosystem components
from bizra_config import RAW_PATH, PROCESSED_PATH, INTAKE_PATH
try:
    from pat_engine import AgentConfig, AgentRole, AgentMessage
    import httpx
except ImportError:
    logging.warning("PAT Engine not found. Running in standalone mode.")
    httpx = None

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | CLEANER | %(message)s'
)
logger = logging.getLogger("AGENTIC_CLEANER")


# ═══════════════════════════════════════════════════════════════════════════════
# SECURITY: AST-BASED CODE VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

# Allowlisted modules that cleaning functions may import
ALLOWED_IMPORTS: Set[str] = {"re", "datetime", "json", "string", "unicodedata"}

# Allowlisted built-in functions (restrictive set for data cleaning)
ALLOWED_BUILTINS: Set[str] = {
    "str", "int", "float", "bool", "len", "range", "enumerate", "zip",
    "list", "dict", "set", "tuple", "sorted", "reversed", "min", "max",
    "sum", "any", "all", "map", "filter", "isinstance", "type", "ord", "chr",
    "strip", "split", "join", "replace", "lower", "upper", "title",
    "startswith", "endswith", "isdigit", "isalpha", "isalnum",
    "True", "False", "None", "print"  # print for debugging only
}

# Forbidden AST node types (dangerous operations)
FORBIDDEN_NODES: Set[type] = {
    ast.Import,       # No arbitrary imports (we validate ImportFrom separately)
    ast.Global,       # No global state modification
    ast.Nonlocal,     # No closure escapes
    ast.Exec,         # No nested exec (Python 2 compat)
    ast.AsyncFunctionDef,  # Keep it synchronous
    ast.AsyncFor,
    ast.AsyncWith,
    ast.Await,
}

# Forbidden function calls (dangerous builtins)
FORBIDDEN_CALLS: Set[str] = {
    "eval", "exec", "compile", "open", "input", "__import__",
    "getattr", "setattr", "delattr", "hasattr",
    "globals", "locals", "vars", "dir",
    "breakpoint", "exit", "quit",
    "os", "sys", "subprocess", "shutil", "pathlib",
    "importlib", "pickle", "marshal", "shelve",
    "socket", "urllib", "requests", "httpx",
}


class CodeValidationError(Exception):
    """Raised when LLM-generated code fails security validation."""
    pass


def _validate_cleaning_code(code: str) -> ast.Module:
    """
    SECURITY: Validate LLM-generated cleaning code using AST analysis.
    
    This function:
    1. Parses the code into an AST
    2. Checks for forbidden node types (imports, exec, etc.)
    3. Validates function calls against allowlist
    4. Ensures only safe imports are used
    
    Returns the parsed AST if valid, raises CodeValidationError otherwise.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise CodeValidationError(f"Syntax error in generated code: {e}")
    
    for node in ast.walk(tree):
        # Check for forbidden node types
        if type(node) in FORBIDDEN_NODES:
            raise CodeValidationError(
                f"Forbidden construct: {type(node).__name__}. "
                "LLM-generated code may not use imports, global, or async."
            )
        
        # Validate Import statements (ast.Import is forbidden, but check ImportFrom)
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module not in ALLOWED_IMPORTS:
                raise CodeValidationError(
                    f"Forbidden import: '{module}'. "
                    f"Allowed imports: {ALLOWED_IMPORTS}"
                )
        
        # Validate function calls
        if isinstance(node, ast.Call):
            func_name = None
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                # For method calls like os.system, check the root
                if isinstance(node.func.value, ast.Name):
                    func_name = node.func.value.id
            
            if func_name and func_name in FORBIDDEN_CALLS:
                raise CodeValidationError(
                    f"Forbidden function call: '{func_name}'. "
                    "This function is not allowed in cleaning code."
                )
        
        # Check for attribute access on dangerous modules
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name):
                if node.value.id in {"os", "sys", "subprocess", "shutil"}:
                    raise CodeValidationError(
                        f"Forbidden module access: '{node.value.id}.{node.attr}'"
                    )
    
    # Verify clean_record function exists
    has_clean_record = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "clean_record":
            has_clean_record = True
            # Verify it takes exactly one argument
            if len(node.args.args) != 1:
                raise CodeValidationError(
                    "clean_record must take exactly one argument"
                )
            break
    
    if not has_clean_record:
        raise CodeValidationError(
            "Generated code must define a 'clean_record(input_data)' function"
        )
    
    logger.info("✅ Code validation passed (AST security check)")
    return tree


def _create_safe_execution_scope() -> Dict[str, Any]:
    """
    SECURITY: Create a restricted execution scope for cleaning functions.
    
    Only provides access to safe, pre-imported modules and a restricted
    set of builtins. No __builtins__ passthrough.
    """
    import re
    import datetime
    import json
    import string
    import unicodedata
    
    # Restricted builtins - only safe functions
    safe_builtins = {
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "len": len,
        "range": range,
        "enumerate": enumerate,
        "zip": zip,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "sorted": sorted,
        "reversed": reversed,
        "min": min,
        "max": max,
        "sum": sum,
        "any": any,
        "all": all,
        "map": map,
        "filter": filter,
        "isinstance": isinstance,
        "type": type,
        "ord": ord,
        "chr": chr,
        "True": True,
        "False": False,
        "None": None,
        "print": print,  # For debugging; can be removed in production
    }
    
    return {
        "re": re,
        "datetime": datetime,
        "json": json,
        "string": string,
        "unicodedata": unicodedata,
        "__builtins__": safe_builtins,  # Restricted, not full __builtins__
    }


# ═══════════════════════════════════════════════════════════════════════════════
# AGENTIC CLEANER CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class AgenticCleaner:
    """
    Implements the Code-Synthesis approach for data cleaning.
    1. Samples raw data.
    2. Uses LLM to synthesize a Python cleaning function.
    3. VALIDATES the function using AST security checks.
    4. Executes the function on the full dataset in a restricted scope.
    """
    
    def __init__(self, model_url: str = "http://192.168.56.1:1234/v1/chat/completions"):
        """
        Initialize the cleaner. Defaults to local LM Studio/Ollama endpoint standard in BIZRA.
        """
        self.model_url = model_url
        self.model_name = "liquid/lfm2.5-1.2b"  # Efficient model for code generation
        self.api_key = "lm-studio"
        self._error_counts: Dict[str, int] = {}  # Track errors for auditability
        
    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Low-level call to the local LLM inference engine."""
        if not httpx:
            logger.error("HTTP client not available.")
            return self._get_fallback_cleaning_code()

        try:
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.2,  # Low temperature for precise code generation
                "max_tokens": 1024,
                "stream": False
            }
            
            response = httpx.post(
                self.model_url, 
                json=payload, 
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
                timeout=60.0  # Code generation can take a moment
            )
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
            
        except Exception as e:
            logger.error(f"LLM Call Failed: {e}")
            return self._get_fallback_cleaning_code()

    def _get_fallback_cleaning_code(self) -> str:
        """Safe fallback cleaning function."""
        return textwrap.dedent("""
            def clean_record(text):
                # Basic standardization (fallback)
                text = str(text).strip()
                text = re.sub(r'\\s+', ' ', text)  # Collapse whitespace
                return text
        """)

    def sample_data(self, file_path: Path, n_lines: int = 5) -> str:
        """Extracts a sample from the file to context-load the LLM."""
        sample = []
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for _ in range(n_lines):
                    line = f.readline()
                    if not line:
                        break
                    sample.append(line.strip())
        except Exception as e:
            logger.error(f"Error sampling file: {e}")
            return ""
        return "\n".join(sample)

    def synthesize_cleaning_function(self, context_sample: str, file_type: str) -> str:
        """
        Survey Section III: Automatic Code-Synthesis Standardization.
        Instructs LLM to generate executable Python code.
        
        SECURITY: The prompt explicitly forbids dangerous operations.
        """
        system_prompt = textwrap.dedent("""
            You are a Data Engineering Agent specialized in writing Python cleaning functions.
            Your task is to write a single Python function named `clean_record(input_data)` that standardizes raw text or JSON data.
            
            Rules:
            1. Handle whitespace irregularities.
            2. Fix common encoding artifacts.
            3. Standardize dates to ISO 8601 if found.
            4. Return the cleaned string or object.
            5. OUTPUT ONLY THE PYTHON CODE. NO MARKDOWN. NO EXPLANATION.
            
            SECURITY CONSTRAINTS (MANDATORY):
            - You may ONLY use these modules: re, datetime, json, string, unicodedata
            - You may NOT use: os, sys, subprocess, open, eval, exec, __import__
            - You may NOT access files, network, or system resources
            - The function must be pure: input -> output, no side effects
        """)
        
        user_prompt = f"""
        Here is a sample of the raw data ({file_type}):
        ---
        {context_sample}
        ---
        
        Write the 'clean_record' function to clean this specific data format.
        Remember: Only use re, datetime, json, string modules. No file/network access.
        """
        
        logger.info("🤖 Synthesizing cleaning code based on sample...")
        code_response = self._call_llm(system_prompt, user_prompt)
        
        # Strip markdown fences if present
        code_response = code_response.replace("```python", "").replace("```", "").strip()
        logger.info(f"✅ Code synthesis complete. Generated:\n{code_response}")
        return code_response

    def process_file(self, file_path: str, output_dir: Optional[str] = None):
        """
        Main Agentic Loop:
        1. Read File
        2. Synthesize Cleaner
        3. VALIDATE (AST security check)
        4. Execute in restricted scope & Save
        """
        input_path = Path(file_path)
        if not input_path.exists():
            logger.error(f"File not found: {input_path}")
            return

        # Reset error tracking for this file
        self._error_counts = {}

        # 1. Sample
        logger.info(f"📂 Processing: {input_path.name}")
        sample = self.sample_data(input_path)
        file_ext = input_path.suffix.lower()

        # 2. Synthesize
        cleaning_code = self.synthesize_cleaning_function(sample, file_ext)
        
        # 3. SECURITY: Validate before execution
        clean_func = None
        validation_passed = False
        
        try:
            _validate_cleaning_code(cleaning_code)
            validation_passed = True
            
            # Create restricted execution scope (no full __builtins__)
            execution_scope = _create_safe_execution_scope()
            
            # Execute in restricted scope
            exec(cleaning_code, execution_scope, execution_scope)  # nosec B102 — AST-validated + restricted builtins
            clean_func = execution_scope.get('clean_record')
            
            # Sanity Check: Test on sample
            test_res = clean_func("  Test Data  ")
            if not test_res or not isinstance(test_res, str):
                logger.warning("⚠️ Synthesized function failed sanity check. Using fallback.")
                clean_func = None
                
        except CodeValidationError as e:
            logger.error(f"🛡️ SECURITY: Code validation failed: {e}")
            clean_func = None
        except Exception as e:
            logger.error(f"Failed to compile cleaning code: {e}")
            clean_func = None

        # Fallback with proper initialization
        if clean_func is None:
            logger.info("📎 Using fallback cleaning function")
            fallback_code = self._get_fallback_cleaning_code()
            local_scope = _create_safe_execution_scope()  # FIX: Initialize local_scope
            exec(fallback_code, local_scope, local_scope)  # nosec B102 — hardcoded safe fallback
            clean_func = local_scope.get('clean_record')

        # 4. Process Full File
        if output_dir:
            out_path = Path(output_dir) / input_path.name
        else:
            out_path = PROCESSED_PATH / "text" / "cleaned" / input_path.name

        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        success_count = 0
        error_count = 0
        
        with open(input_path, 'r', encoding='utf-8', errors='ignore') as infile, \
             open(out_path, 'w', encoding='utf-8') as outfile:
            
            for line_num, line in enumerate(tqdm(infile, desc="Cleaning"), 1):
                try:
                    cleaned = clean_func(line)
                    if cleaned and cleaned.strip(): 
                        outfile.write(str(cleaned) + "\n")
                        success_count += 1
                except Exception as loop_err:
                    # FIX: Track errors instead of silent pass
                    error_count += 1
                    error_type = type(loop_err).__name__
                    self._error_counts[error_type] = self._error_counts.get(error_type, 0) + 1
                    
                    # Log first occurrence of each error type
                    if self._error_counts[error_type] == 1:
                        logger.warning(f"⚠️ Line {line_num}: {error_type}: {loop_err}")
        
        # Auditability: Report error summary
        if error_count > 0:
            logger.warning(f"⚠️ {error_count} lines failed cleaning:")
            for err_type, count in self._error_counts.items():
                logger.warning(f"   - {err_type}: {count} occurrences")
        
        logger.info(f"✨ Cleaning Complete. {success_count} lines written, {error_count} errors. Output: {out_path}")
        
        return {
            "success_count": success_count,
            "error_count": error_count,
            "error_breakdown": self._error_counts.copy(),
            "output_path": str(out_path),
            "validation_passed": validation_passed,
        }


if __name__ == "__main__":
    # Test execution on intake folder
    cleaner = AgenticCleaner()
    
    # Target the staging area defined in DataLakeProcessor.ps1
    staging_area = INTAKE_PATH / "text_staging"
    
    if not staging_area.exists():
        logger.info(f"Staging area not found: {staging_area}")
    else:
        # Scan for supported text extensions
        intake_files = []
        for ext in ["*.txt", "*.md", "*.py", "*.js", "*.json", "*.csv"]:
            intake_files.extend(staging_area.glob(ext))
        
        logger.info(f"Found {len(intake_files)} files to process.")
        
        for f in intake_files:
            result = cleaner.process_file(str(f))
            if result:
                logger.info(f"Result: {json.dumps(result, indent=2)}")
