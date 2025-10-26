#!/usr/bin/env python3
"""
Fix std::unexpected usage to std::vesper_unexpected throughout the codebase
"""

import os
import re
import sys

def fix_unexpected_in_file(filepath):
    """Replace std::unexpected with std::vesper_unexpected in a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace std::unexpected( with std::vesper_unexpected(
        # But NOT std::vesper_vesper_unexpected (avoid double replacement)
        original_content = content
        content = re.sub(r'(?<!vesper_)std::unexpected\(', r'std::vesper_unexpected(', content)
        
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

def main():
    """Main function to fix all C++ files."""
    src_dir = os.path.join(os.path.dirname(__file__), 'src')
    
    if not os.path.exists(src_dir):
        print(f"Source directory not found: {src_dir}")
        return 1
    
    fixed_files = []
    
    # Walk through all source files
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith(('.cpp', '.hpp', '.h')):
                filepath = os.path.join(root, file)
                if fix_unexpected_in_file(filepath):
                    fixed_files.append(filepath)
                    print(f"Fixed: {filepath}")
    
    if fixed_files:
        print(f"\nFixed {len(fixed_files)} files:")
        for f in fixed_files:
            print(f"  - {os.path.relpath(f, src_dir)}")
    else:
        print("No files needed fixing.")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())