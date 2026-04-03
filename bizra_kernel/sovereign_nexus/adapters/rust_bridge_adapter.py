"""
Rust Bridge Adapter for BIZRA Sovereign Nexus

Provides an interface to connect with Rust FFI components,
particularly the FateEngine and other Rust-based systems.
"""

import ctypes
import os
from typing import Dict, Any, Optional, Union
from pathlib import Path


class RustBridgeAdapter:
    """
    Adapter to connect Python-based Nexus with Rust FFI components.
    
    Handles communication with the Rust-based FateEngine and other
    core systems implemented in Rust.
    """
    
    def __init__(self, lib_path: Optional[str] = None):
        """
        Initialize the Rust bridge adapter.
        
        Args:
            lib_path: Path to the Rust shared library. If None, attempts to find it automatically.
        """
        self.lib_path = lib_path or self._find_lib_path()
        self.lib = None
        self.connected = False
        
        if self.lib_path:
            self.connect()
    
    def _find_lib_path(self) -> Optional[str]:
        """Attempt to find the Rust shared library automatically."""
        possible_paths = [
            "./libbizra.so",  # Current directory
            "./target/release/libbizra.so",  # Standard Rust build path
            "./target/release/libmeta_alpha_dual_agentic.so", # Real Rust build path
            "/usr/local/lib/libbizra.so",  # System-wide installation
            "./lib/libbizra.so",  # Library subdirectory
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # Try to find with glob
        import glob
        libs = glob.glob("**/lib*bizra*.so", recursive=True)
        if libs:
            return libs[0]
        
        return None
    
    def connect(self) -> bool:
        """
        Establish connection to the Rust library.
        
        Returns:
            True if connection successful, False otherwise.
        """
        try:
            if not self.lib_path or not os.path.exists(self.lib_path):
                print(f"Rust library not found at: {self.lib_path}")
                return False
            
            self.lib = ctypes.CDLL(self.lib_path)
            self.connected = True
            
            # Define function signatures
            self._define_function_signatures()
            
            print(f"Connected to Rust library at: {self.lib_path}")
            return True
            
        except Exception as e:
            print(f"Failed to connect to Rust library: {e}")
            self.connected = False
            return False
    
    def _define_function_signatures(self):
        """Define the function signatures for Rust FFI calls."""
        # Example signatures - these would need to match the actual Rust library
        try:
            # fate_engine_process - takes a string and returns a processed string
            self.lib.fate_engine_process.argtypes = [ctypes.c_char_p]
            self.lib.fate_engine_process.restype = ctypes.c_char_p
            
            # validate_receipt - takes a receipt string and returns validation result
            self.lib.validate_receipt.argtypes = [ctypes.c_char_p]
            self.lib.validate_receipt.restype = ctypes.c_bool
            
            # get_ihsan_score - takes a string and returns a score
            self.lib.get_ihsan_score.argtypes = [ctypes.c_char_p]
            self.lib.get_ihsan_score.restype = ctypes.c_float
        except AttributeError:
            # Functions might not exist in the library yet
            pass
    
    def execute_fate_engine(self, input_data: str) -> Optional[str]:
        """
        Execute the FateEngine with the given input.
        
        Args:
            input_data: Input data to process
            
        Returns:
            Processed result from FateEngine, or None if failed
        """
        if not self.connected or not self.lib:
            print("Not connected to Rust library")
            return None
        
        try:
            # Convert Python string to C string
            c_input = ctypes.c_char_p(input_data.encode('utf-8'))
            
            # Call the Rust function
            result = self.lib.fate_engine_process(c_input)
            
            # Convert result back to Python string
            if result:
                return result.decode('utf-8')
            else:
                return None
                
        except Exception as e:
            print(f"Error executing FateEngine: {e}")
            return None
    
    def validate_receipt(self, receipt: str) -> bool:
        """
        Validate a cryptographic receipt using the Rust validator.
        
        Args:
            receipt: The receipt string to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not self.connected or not self.lib:
            print("Not connected to Rust library")
            return False
        
        try:
            c_receipt = ctypes.c_char_p(receipt.encode('utf-8'))
            result = self.lib.validate_receipt(c_receipt)
            return bool(result)
            
        except Exception as e:
            print(f"Error validating receipt: {e}")
            return False
    
    def get_ihsan_score(self, content: str) -> Optional[float]:
        """
        Get the Ihsan score for the given content.
        
        Args:
            content: Content to evaluate
            
        Returns:
            Ihsan score between 0.0 and 1.0, or None if failed
        """
        if not self.connected or not self.lib:
            print("Not connected to Rust library")
            return None
        
        try:
            c_content = ctypes.c_char_p(content.encode('utf-8'))
            result = self.lib.get_ihsan_score(c_content)
            return float(result)
            
        except Exception as e:
            print(f"Error getting Ihsan score: {e}")
            return None
    
    def disconnect(self):
        """Close the connection to the Rust library."""
        self.connected = False
        self.lib = None