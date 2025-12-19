#!/usr/bin/env python3
"""
BIZRA JSON Conversation Processor for HyperGraphRAG
Processes conversations.json exports and feeds them into the hypergraph pipeline.
Supports streaming for large files (100MB+).
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Iterator, Dict, Any, List
import argparse
import ijson  # For streaming large JSON

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent))

from bizra_hypergraphrag_wrapper import BIZRAHyperGraphOffline


def stream_conversations_ijson(json_path: Path) -> Iterator[Dict[str, Any]]:
    """Stream conversations from large JSON file using ijson."""
    with open(json_path, 'rb') as f:
        # Try different structures
        try:
            # Most common: array at root
            parser = ijson.items(f, 'item')
            count = 0
            for conv in parser:
                count += 1
                yield conv
            if count > 0:
                return
        except Exception as e:
            pass
        
        # Try as object with nested array
        f.seek(0)
        try:
            for key in ['conversations', 'data', 'items', 'chats']:
                f.seek(0)
                parser = ijson.items(f, f'{key}.item')
                count = 0
                for conv in parser:
                    count += 1
                    yield conv
                if count > 0:
                    return
        except:
            pass


def load_conversations_standard(json_path: Path) -> List[Dict[str, Any]]:
    """Load conversations using standard JSON (for smaller files)."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return data.get('conversations', data.get('items', [data]))
        return [data]


def extract_conversation_text(conv: Dict[str, Any]) -> str:
    """Extract text content from a conversation object.
    Supports: ChatGPT, Claude/Anthropic, DeepSeek formats.
    """
    texts = []
    
    # Get title/name
    if title := conv.get('title', conv.get('name', '')):
        texts.append(f"# {title}")
    
    # Get creation time
    if created := conv.get('create_time', conv.get('created_at', conv.get('inserted_at', ''))):
        if isinstance(created, (int, float)):
            try:
                dt = datetime.fromtimestamp(created, tz=timezone.utc)
                texts.append(f"Date: {dt.isoformat()}")
            except:
                pass
        elif isinstance(created, str):
            texts.append(f"Date: {created}")
    
    # ============== Claude/Anthropic format ==============
    if chat_messages := conv.get('chat_messages', []):
        for msg in chat_messages:
            sender = msg.get('sender', 'unknown')
            content = msg.get('content', [])
            text_content = msg.get('text', '')
            
            # Get text from content array
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        texts.append(f"[{sender.upper()}]: {item.get('text', '')}")
                    elif isinstance(item, str):
                        texts.append(f"[{sender.upper()}]: {item}")
            elif isinstance(content, str) and content.strip():
                texts.append(f"[{sender.upper()}]: {content}")
            
            # Fallback to text field
            if text_content and text_content.strip():
                texts.append(f"[{sender.upper()}]: {text_content}")
        
        if texts:
            return '\n\n'.join(texts)
    
    # ============== DeepSeek/ChatGPT mapping format ==============
    if mapping := conv.get('mapping', {}):
        for node_id, node in mapping.items():
            if node_id == 'root':
                continue
            
            if msg := node.get('message'):
                # DeepSeek format with fragments
                if fragments := msg.get('fragments', []):
                    for frag in fragments:
                        frag_type = frag.get('type', 'unknown')
                        content = frag.get('content', '')
                        
                        # REQUEST = user, THINKING/RESPONSE = assistant
                        if frag_type == 'REQUEST':
                            role = 'USER'
                        elif frag_type in ('THINKING', 'RESPONSE'):
                            role = 'ASSISTANT'
                        else:
                            role = frag_type
                        
                        if content and isinstance(content, str) and len(content) > 1:
                            # Skip JSON-looking content (system prompts)
                            if not content.startswith('{'):
                                texts.append(f"[{role}]: {content[:5000]}")  # Limit length
                
                # ChatGPT format
                else:
                    role = msg.get('author', {}).get('role', msg.get('role', 'unknown'))
                    content = msg.get('content', {})
                    
                    if isinstance(content, dict):
                        parts = content.get('parts', [])
                        for part in parts:
                            if isinstance(part, str) and part.strip():
                                texts.append(f"[{role.upper()}]: {part}")
                    elif isinstance(content, str) and content.strip():
                        texts.append(f"[{role.upper()}]: {content}")
        
        if texts:
            return '\n\n'.join(texts)
    
    # ============== Direct messages list ==============
    messages = conv.get('messages', conv.get('turns', []))
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get('role', msg.get('author', msg.get('sender', 'unknown')))
                content = msg.get('content', msg.get('text', ''))
                
                if isinstance(content, str) and content.strip():
                    texts.append(f"[{role.upper()}]: {content}")
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, str) and part.strip():
                            texts.append(f"[{role.upper()}]: {part}")
                        elif isinstance(part, dict):
                            if text := part.get('text', part.get('content', '')):
                                texts.append(f"[{role.upper()}]: {text}")
    
    return '\n\n'.join(texts)


def process_json_conversations(
    json_paths: List[Path],
    output_dir: Path,
    use_streaming: bool = True,
    limit: int = 0
) -> Dict[str, Any]:
    """Process JSON conversation files and build hypergraph."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize hypergraph with output directory
    hg = BIZRAHyperGraphOffline(working_dir=str(output_dir))
    
    stats = {
        'files_processed': 0,
        'conversations_processed': 0,
        'total_text_chars': 0,
        'sources': []
    }
    
    for json_path in json_paths:
        if not json_path.exists():
            print(f"⚠️ File not found: {json_path}")
            continue
            
        file_size_mb = json_path.stat().st_size / (1024 * 1024)
        print(f"\n📂 Processing: {json_path.name} ({file_size_mb:.1f} MB)")
        stats['sources'].append(json_path.name)
        stats['files_processed'] += 1
        
        # Choose loading method based on size
        if use_streaming and file_size_mb > 10:
            print(f"   Using streaming parser (large file)")
            conv_iter = stream_conversations_ijson(json_path)
        else:
            print(f"   Loading full JSON...")
            conv_iter = iter(load_conversations_standard(json_path))
        
        conv_count = 0
        for conv in conv_iter:
            text = extract_conversation_text(conv)
            
            if text and len(text) > 100:  # Skip very short conversations
                # Create a virtual document path
                conv_id = conv.get('id', conv.get('conversation_id', f'conv_{conv_count}'))
                title = conv.get('title', 'Untitled')[:50].replace('/', '_')
                doc_path = f"{json_path.stem}/{conv_id}_{title}"
                
                # Process through hypergraph
                hg.insert(text, doc_path)
                
                stats['total_text_chars'] += len(text)
                conv_count += 1
                
                if conv_count % 100 == 0:
                    print(f"   Processed {conv_count} conversations...")
                
                if limit > 0 and conv_count >= limit:
                    print(f"   Reached limit of {limit} conversations")
                    break
        
        stats['conversations_processed'] += conv_count
        print(f"   ✅ {conv_count} conversations extracted")
    
    # Get extraction stats
    entity_count = len(hg.entities)
    hyperedge_count = len(hg.hyperedges)
    
    print(f"\n{'='*60}")
    print(f"📊 EXTRACTION RESULTS")
    print(f"{'='*60}")
    print(f"📄 Files: {stats['files_processed']}")
    print(f"💬 Conversations: {stats['conversations_processed']}")
    print(f"📝 Total text: {stats['total_text_chars']:,} chars ({stats['total_text_chars']//1000}K)")
    print(f"🧬 Entities: {entity_count}")
    print(f"🔗 HyperEdges: {hyperedge_count}")
    
    # Save outputs
    hg.save()
    
    # Save extended stats
    stats['entities'] = entity_count
    stats['hyperedges'] = hyperedge_count
    stats['timestamp'] = datetime.now().isoformat()
    
    with open(output_dir / 'json_processing_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n💾 Saved to: {output_dir}")
    return stats


def process_memories_file(
    memories_path: Path,
    output_dir: Path
) -> Dict[str, Any]:
    """Process ChatGPT memories.json file."""
    
    print(f"\n🧠 Processing memories: {memories_path.name}")
    
    with open(memories_path, 'r', encoding='utf-8') as f:
        memories = json.load(f)
    
    if not isinstance(memories, list):
        memories = memories.get('memories', memories.get('items', [memories]))
    
    # Initialize hypergraph with custom output dir
    mem_output = output_dir / 'memories_hypergraph'
    hg = BIZRAHyperGraphOffline(working_dir=str(mem_output))
    
    memory_texts = []
    for mem in memories:
        content = mem.get('content', mem.get('text', mem.get('memory', '')))
        if content:
            memory_texts.append(content)
    
    # Process as single document
    full_text = '\n\n---\n\n'.join(memory_texts)
    if full_text:
        hg.insert(full_text, "ChatGPT_Memories_Export")
    
    hg.save()
    
    stats = {
        'memories_count': len(memories),
        'entities': len(hg.entities),
        'hyperedges': len(hg.hyperedges)
    }
    
    print(f"   ✅ {len(memories)} memories → {len(hg.entities)} entities, {len(hg.hyperedges)} hyperedges")
    return stats


def main():
    parser = argparse.ArgumentParser(description='Process JSON conversation exports for HyperGraphRAG')
    parser.add_argument('paths', nargs='*', help='JSON files to process')
    parser.add_argument('--output', '-o', default='evidence/bizra_hypergraph_json',
                       help='Output directory')
    parser.add_argument('--limit', '-l', type=int, default=0,
                       help='Limit conversations per file (0=unlimited)')
    parser.add_argument('--no-stream', action='store_true',
                       help='Disable streaming parser')
    parser.add_argument('--all', action='store_true',
                       help='Process all JSON files in chat data sample folder')
    
    args = parser.parse_args()
    
    # Find the workspace root
    script_dir = Path(__file__).parent
    workspace = script_dir.parent
    
    output_dir = workspace / args.output
    
    if args.all:
        # Find main conversation JSON files
        chat_dir = workspace / 'chat data sample'
        json_paths = [
            chat_dir / 'data-2025-12-15-17-59-21-batch-0000' / 'conversations.json',
            chat_dir / 'deepseek_data-2025-12-16' / 'conversations.json',
        ]
        
        # Also process memories if exists
        memories_path = chat_dir / 'data-2025-12-15-17-59-21-batch-0000' / 'memories.json'
        if memories_path.exists():
            process_memories_file(memories_path, output_dir)
        
    elif args.paths:
        json_paths = [Path(p) for p in args.paths]
    else:
        # Default to batch export
        chat_dir = workspace / 'chat data sample'
        json_paths = [
            chat_dir / 'data-2025-12-15-17-59-21-batch-0000' / 'conversations.json',
        ]
    
    # Filter to existing files
    json_paths = [p for p in json_paths if p.exists()]
    
    if not json_paths:
        print("❌ No JSON files found to process")
        return
    
    print("="*60)
    print("BIZRA JSON Conversation Processor")
    print("="*60)
    print(f"Files to process: {len(json_paths)}")
    for p in json_paths:
        print(f"  - {p.name}")
    
    process_json_conversations(
        json_paths,
        output_dir,
        use_streaming=not args.no_stream,
        limit=args.limit
    )


if __name__ == '__main__':
    main()
