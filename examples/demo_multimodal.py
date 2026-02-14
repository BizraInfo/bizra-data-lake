# BIZRA Multi-Modal Pipeline Test
# Tests: Vision (CLIP + qwen3-vl) + Image Search + LM Studio

import asyncio
import base64
from pathlib import Path
from bizra_orchestrator import BIZRAOrchestrator, BIZRAQuery, QueryComplexity

async def test_multimodal_pipeline():
    print('🎨 BIZRA Multi-Modal Pipeline Test')
    print('=' * 60)
    
    # Initialize with multimodal enabled
    print('\n📡 Initializing BIZRA with Multi-Modal + Vision...')
    orchestrator = BIZRAOrchestrator(
        enable_pat=True,
        enable_multimodal=True,
        ollama_model="qwen/qwen3-vl-8b"  # Vision-Language model
    )
    
    success = await orchestrator.initialize()
    print(f'   Orchestrator: {"✅ Ready" if success else "❌ Failed"}')
    print(f'   Multi-Modal: {"✅ Enabled" if orchestrator.multimodal_enabled else "⚠️ Disabled"}')
    print(f'   PAT Engine: {"✅ Active" if orchestrator.pat_enabled else "⚠️ Inactive"}')
    
    # Check Dual Agentic (LM Studio) 
    if orchestrator.dual_agentic_enabled and orchestrator.dual_agentic:
        da_available = await orchestrator.dual_agentic.check_availability()
        if da_available:
            print(f'   LM Studio: ✅ Connected ({len(orchestrator.dual_agentic._models)} models)')
            # Check for vision model
            vision_models = [m for m in orchestrator.dual_agentic._models if 'vl' in m.lower() or 'vision' in m.lower()]
            if vision_models:
                print(f'   Vision Models: {vision_models}')
    
    if not success:
        print('\n❌ Initialization failed')
        return
    
    # Test 1: Text-only query with multimodal context
    print('\n' + '=' * 60)
    print('📝 Test 1: Text Query with Multi-Modal Context')
    print('=' * 60)
    
    query1 = BIZRAQuery(
        text='What visual patterns are commonly found in BIZRA architecture diagrams?',
        complexity=QueryComplexity.MODERATE,
        require_sources=True,
        enable_kep=True,
        cross_modal_search=True
    )
    
    response1 = await orchestrator.query(query1)
    print(f'\n📊 Results:')
    print(f'   SNR Score: {response1.snr_score:.2f}')
    print(f'   Sources: {len(response1.sources)}')
    print(f'   Similar Images: {len(response1.similar_images)}')
    print(f'   Modalities Used: {response1.modality_used}')
    print(f'   Execution Time: {response1.execution_time}s')
    
    if response1.discipline_coverage:
        print(f'   Discipline Coverage: {response1.discipline_coverage:.1%}')
    
    # Test 2: Check if any images exist in the data lake
    print('\n' + '=' * 60)
    print('🖼️ Test 2: Checking Image Index')
    print('=' * 60)
    
    from bizra_config import IMAGE_EMBEDDINGS_PATH, PROCESSED_PATH
    
    # Check for image embeddings
    if IMAGE_EMBEDDINGS_PATH.exists():
        image_files = list(IMAGE_EMBEDDINGS_PATH.glob("*.npy"))
        print(f'   Image embeddings found: {len(image_files)}')
    else:
        print(f'   Image embeddings path not found')
    
    # Check for images in processed
    images_path = PROCESSED_PATH / "images"
    if images_path.exists():
        image_count = len(list(images_path.glob("*.*")))
        print(f'   Images in processed: {image_count}')
    else:
        print(f'   No images directory in processed')
    
    # Test 3: Test with sample image if available
    print('\n' + '=' * 60)
    print('🔬 Test 3: Vision Analysis Capability Check')
    print('=' * 60)
    
    if orchestrator.multimodal and orchestrator.multimodal_enabled:
        print('   Multi-Modal Engine: ✅ Available')
        
        # Check CLIP availability
        try:
            from multimodal_engine import CLIP_AVAILABLE, WHISPER_AVAILABLE, OCR_AVAILABLE
            print(f'   CLIP (Image Embeddings): {"✅" if CLIP_AVAILABLE else "❌"}')
            print(f'   Whisper (Audio): {"✅" if WHISPER_AVAILABLE else "❌"}')
            print(f'   OCR (Tesseract): {"✅" if OCR_AVAILABLE else "❌"}')
        except ImportError as e:
            print(f'   Could not check component status: {e}')
    else:
        print('   Multi-Modal Engine: ❌ Not available')
    
    # Summary
    print('\n' + '=' * 60)
    print('📋 MULTI-MODAL STATUS SUMMARY')
    print('=' * 60)
    
    status = {
        'Orchestrator': success,
        'Multi-Modal Engine': orchestrator.multimodal_enabled,
        'PAT Engine': orchestrator.pat_enabled,
        'KEP Bridge': orchestrator.kep_enabled,
        'Discipline Engine': orchestrator.discipline_enabled,
        'Dual Agentic (LM Studio)': orchestrator.dual_agentic_enabled,
    }
    
    for component, enabled in status.items():
        icon = "✅" if enabled else "❌"
        print(f'   {icon} {component}')
    
    print('\n✅ Multi-Modal Pipeline Test Complete!')

if __name__ == "__main__":
    asyncio.run(test_multimodal_pipeline())
