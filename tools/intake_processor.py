#!/usr/bin/env python3
"""
BIZRA Intake Processor - ChatGPT Conversation JSON → Text Conversion
=====================================================================
Processes ChatGPT conversation exports from 00_INTAKE into queryable text
for the BIZRA Data Lake knowledge base.

Author: BIZRA Peak Masterpiece Engine
Version: 1.0.0
"""

import json
import re
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
INTAKE_DIR = Path("C:/BIZRA-DATA-LAKE/00_INTAKE")
OUTPUT_DIR = Path("C:/BIZRA-DATA-LAKE/02_PROCESSED/text/conversations")
MANIFEST_PATH = Path("C:/BIZRA-DATA-LAKE/02_PROCESSED/intake_manifest.jsonl")


def extract_text_from_message(message: Dict[str, Any]) -> Optional[str]:
    """Extract clean text content from a ChatGPT message node."""
    if not message or 'message' not in message:
        return None
    
    msg = message['message']
    if not msg:
        return None
    
    author = msg.get('author', {})
    role = author.get('role', 'unknown')
    
    content = msg.get('content', {})
    content_type = content.get('content_type', '')
    
    # Extract text parts
    parts = content.get('parts', [])
    if not parts:
        return None
    
    text_parts = []
    for part in parts:
        if isinstance(part, str) and part.strip():
            text_parts.append(part.strip())
    
    if not text_parts:
        return None
    
    combined_text = '\n'.join(text_parts)
    
    # Format with role prefix
    role_prefix = {
        'user': '## User',
        'assistant': '## Assistant', 
        'system': '## System',
        'tool': '## Tool Response'
    }.get(role, f'## {role.title()}')
    
    return f"{role_prefix}\n\n{combined_text}"


def extract_conversation_metadata(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract metadata from conversation JSON."""
    return {
        'title': data.get('title', 'Untitled Conversation'),
        'conversation_id': data.get('conversation_id', ''),
        'create_time': data.get('create_time'),
        'update_time': data.get('update_time'),
        'model_slug': data.get('default_model_slug', 'unknown'),
        'is_archived': data.get('is_archived', False),
    }


def traverse_conversation_tree(mapping: Dict[str, Any], current_node: str) -> List[str]:
    """Traverse the conversation tree from current node backwards to build message order."""
    messages = []
    visited = set()
    
    # Build the tree structure
    node_to_children = {}
    for node_id, node_data in mapping.items():
        parent = node_data.get('parent')
        if parent:
            if parent not in node_to_children:
                node_to_children[parent] = []
            node_to_children[parent].append(node_id)
    
    # BFS from root to current
    def find_path_to_current(root_candidates):
        """Find path from root to current_node."""
        queue = []
        for root in root_candidates:
            queue.append((root, [root]))
        
        while queue:
            node, path = queue.pop(0)
            if node == current_node:
                return path
            children = node_to_children.get(node, [])
            for child in children:
                if child not in visited:
                    visited.add(child)
                    queue.append((child, path + [child]))
        return []
    
    # Find roots (nodes with no parent or parent not in mapping)
    roots = []
    for node_id, node_data in mapping.items():
        parent = node_data.get('parent')
        if not parent or parent not in mapping:
            roots.append(node_id)
    
    path = find_path_to_current(roots)
    
    # Extract messages in order
    for node_id in path:
        node = mapping.get(node_id)
        if node:
            text = extract_text_from_message(node)
            if text:
                messages.append(text)
    
    return messages


def process_conversation_file(json_path: Path) -> Optional[Dict[str, Any]]:
    """Process a single conversation JSON file."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, dict):
            logger.warning(f"Skipping {json_path.name}: Not a valid conversation object")
            return None
        
        # Extract metadata
        metadata = extract_conversation_metadata(data)
        
        # Get mapping and current node
        mapping = data.get('mapping', {})
        current_node = data.get('current_node')
        
        if not mapping or not current_node:
            logger.warning(f"Skipping {json_path.name}: Missing mapping or current_node")
            return None
        
        # Extract messages in order
        messages = traverse_conversation_tree(mapping, current_node)
        
        if not messages:
            logger.warning(f"Skipping {json_path.name}: No extractable messages")
            return None
        
        # Build output document
        title = metadata['title']
        safe_title = re.sub(r'[<>:"/\\|?*]', '_', title)[:100]
        
        # Create content
        header = f"""# {title}

**Conversation ID:** {metadata['conversation_id']}
**Model:** {metadata['model_slug']}
**Created:** {datetime.fromtimestamp(metadata['create_time']).isoformat() if metadata['create_time'] else 'Unknown'}

---

"""
        content = header + '\n\n---\n\n'.join(messages)
        
        # Generate output filename
        conv_id = metadata['conversation_id'][:8] if metadata['conversation_id'] else hashlib.blake2b(title.encode(), digest_size=16).hexdigest()[:8]
        output_filename = f"{conv_id}-{safe_title}.md"
        
        return {
            'content': content,
            'filename': output_filename,
            'metadata': metadata,
            'source_file': str(json_path),
            'message_count': len(messages),
            'char_count': len(content),
        }
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in {json_path.name}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error processing {json_path.name}: {e}")
        return None


def process_all_intake() -> Dict[str, Any]:
    """Process all JSON files in the INTAKE directory."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    stats = {
        'total_files': 0,
        'processed': 0,
        'skipped': 0,
        'errors': 0,
        'total_messages': 0,
        'total_chars': 0,
    }
    
    manifest_entries = []
    
    # Find all JSON files
    json_files = list(INTAKE_DIR.rglob('*.json'))
    stats['total_files'] = len(json_files)
    
    logger.info(f"Found {len(json_files)} JSON files to process")
    
    for idx, json_path in enumerate(json_files, 1):
        if idx % 50 == 0:
            logger.info(f"Progress: {idx}/{len(json_files)} files processed")
        
        result = process_conversation_file(json_path)
        
        if result:
            # Write output file
            output_path = OUTPUT_DIR / result['filename']
            
            # Handle duplicates
            counter = 1
            while output_path.exists():
                stem = output_path.stem
                output_path = OUTPUT_DIR / f"{stem}_{counter}.md"
                counter += 1
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result['content'])
            
            stats['processed'] += 1
            stats['total_messages'] += result['message_count']
            stats['total_chars'] += result['char_count']
            
            # Add to manifest
            manifest_entries.append({
                'source': result['source_file'],
                'output': str(output_path),
                'title': result['metadata']['title'],
                'conversation_id': result['metadata']['conversation_id'],
                'message_count': result['message_count'],
                'char_count': result['char_count'],
                'model': result['metadata']['model_slug'],
                'processed_at': datetime.now().isoformat(),
            })
        else:
            stats['skipped'] += 1
    
    # Write manifest
    with open(MANIFEST_PATH, 'w', encoding='utf-8') as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry) + '\n')
    
    logger.info(f"\n=== Processing Complete ===")
    logger.info(f"Total files: {stats['total_files']}")
    logger.info(f"Processed: {stats['processed']}")
    logger.info(f"Skipped: {stats['skipped']}")
    logger.info(f"Total messages: {stats['total_messages']}")
    logger.info(f"Total characters: {stats['total_chars']:,}")
    logger.info(f"Manifest written to: {MANIFEST_PATH}")
    
    return stats


if __name__ == "__main__":
    print("=" * 60)
    print("BIZRA INTAKE PROCESSOR - ChatGPT Conversation Extractor")
    print("=" * 60)
    print()
    
    stats = process_all_intake()
    
    print()
    print("=" * 60)
    print("PROCESSING COMPLETE")
    print(f"  ✓ {stats['processed']} conversations converted")
    print(f"  ✓ {stats['total_messages']} messages extracted")
    print(f"  ✓ {stats['total_chars']:,} characters of knowledge")
    print("=" * 60)
