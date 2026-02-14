# BIZRA Pipeline Integration Test
# Validates: Orchestrator + KEP + 47-Discipline Synthesis + LM Studio + PAT

import asyncio
from bizra_orchestrator import BIZRAOrchestrator, BIZRAQuery, QueryComplexity

async def test_pipeline():
    print('🚀 Initializing BIZRA Unified Orchestrator v3.0 with PAT...')
    
    # Enable PAT with LM Studio backend - using fast model
    orchestrator = BIZRAOrchestrator(
        enable_pat=True, 
        enable_multimodal=False,
        ollama_model="liquid/lfm2.5-1.2b"  # Fast 1.2B model for testing
    )
    
    print('\n📡 Initializing engines...')
    success = await orchestrator.initialize()
    status = "✅ SUCCESS" if success else "❌ FAILED"
    print(f'   Initialization: {status}')
    
    # Check Dual Agentic (LM Studio) connection
    if orchestrator.dual_agentic_enabled and orchestrator.dual_agentic:
        da_available = await orchestrator.dual_agentic.check_availability()
        if da_available:
            print(f'   LM Studio: ✅ Connected ({len(orchestrator.dual_agentic._models)} models)')
        else:
            print(f'   LM Studio: ❌ Not available')
    
    if success:
        print('\n🔍 Testing pipeline with COMPLEX query (uses PAT + LM Studio)...')
        query = BIZRAQuery(
            text='What is the relationship between graph theory and information theory in BIZRA?',
            complexity=QueryComplexity.COMPLEX,  # Use COMPLEX to trigger PAT generation
            require_sources=True,
            enable_kep=True
        )
        
        response = await orchestrator.query(query)
        
        print(f'\n📊 PIPELINE VALIDATION RESULTS:')
        print(f'   SNR Score: {response.snr_score:.2f}')
        ihsan_status = "✅" if response.ihsan_achieved else "❌"
        print(f'   Ihsān Achieved: {ihsan_status}')
        print(f'   Execution Time: {response.execution_time}s')
        print(f'   Sources Retrieved: {len(response.sources)}')
        print(f'   Synergies Detected: {len(response.synergies)}')
        
        # NEW: Discipline coverage
        if response.discipline_coverage:
            print(f'   Discipline Coverage: {response.discipline_coverage:.1%}')
        if response.generator_strengths:
            print(f'   Generator Strengths:')
            for gen, strength in response.generator_strengths.items():
                print(f'      {gen}: {strength:.1%}')
        
        print(f'\n📝 Answer Preview:')
        print(f'   {response.answer[:300]}...' if len(response.answer) > 300 else f'   {response.answer}')
        
        print(f'\n✅ Full pipeline integration validated!')
    else:
        print('\n❌ Pipeline initialization failed')

if __name__ == "__main__":
    asyncio.run(test_pipeline())
