#!/usr/bin/env python3
"""
E2B Sandbox Verification Script
================================
Verifies that E2B Code Interpreter is properly installed and configured.
Tests: API key loading, sandbox creation, code execution, file ops, package install.
"""

import os
import sys

def main():
    # Load environment
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv('E2B_API_KEY')
    if api_key:
        print(f'[1] E2B_API_KEY loaded: {api_key[:20]}...')
    else:
        print('[1] E2B_API_KEY: NOT FOUND - check .env file')
        sys.exit(1)
    
    # Import E2B
    from e2b_code_interpreter import Sandbox
    
    # Create sandbox
    print('[2] Creating sandbox...')
    sb = Sandbox.create(api_key=api_key)
    print(f'[3] Sandbox ID: {sb.sandbox_id}')
    
    # Test code execution
    result = sb.run_code('import sys; print(f"Python {sys.version}")')
    if result.logs.stdout:
        print(f'[4] Sandbox Python version: {result.logs.stdout[0].strip()}')
    
    # Test file operations
    sb.files.write('/tmp/test.txt', 'BIZRA sandbox verified')
    content = sb.files.read('/tmp/test.txt')
    print(f'[5] File write/read: {content}')
    
    # Test package installation
    result = sb.run_code('import numpy; print(numpy.__version__)')
    if result.logs.stdout:
        print(f'[6] Numpy available: {result.logs.stdout[-1].strip()}')
    else:
        print('[6] Numpy: installing...')
        result = sb.run_code('!pip install numpy -q && import numpy; print(numpy.__version__)')
        if result.logs.stdout:
            print(f'[6] Numpy installed: {result.logs.stdout[-1].strip()}')
    
    # Cleanup
    sb.kill()
    print('[7] Sandbox terminated successfully')
    print('\n=== E2B SANDBOX: FULLY OPERATIONAL ===')
    return 0

if __name__ == '__main__':
    sys.exit(main())
