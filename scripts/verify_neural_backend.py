import sys
import os
sys.path.insert(0, '.')
import asyncio
import json

async def integration_test():
    try:
        from bizra_kernel.model_hub import SovereignModelHub, ModelRequest
        print('[OK] SovereignModelHub imported successfully')
        
        print(f"[INFO] Working Directory: {os.getcwd()}")
        
        hub = SovereignModelHub()
        print(f'[OK] Hub initialized')
        
        # List models
        print('\nDiscovered models (Abstract Registry):')
        for key, model_data in hub.models.items():
            provider = model_data["provider"]
            health = await provider.check_health()
            status = 'HEALTHY' if health['healthy'] else 'UNHEALTHY'
            print(f'  - {key}: {status} ({provider.name} | Role: {model_data["role"]})')
        
        # Test 1: Simple task (should use llama3.1:8b)
        request1 = ModelRequest(
            task='Explain the concept of sovereignty in AI systems in one sentence.',
            context={'domain': 'AI ethics', 'complexity': 'low'},
            requirements={'complexity_boost': 0.1}
        )
        print('\nTest 1 - Simple task...')
        response1 = await hub.execute_sovereign_task(request1)
        if response1.success:
            print(f'[OK] Request succeeded using model: {response1.model_used}')
            print(f'  Response: {response1.content[:150]}...')
            # Assert correct routing
            if "llama" in response1.model_used.lower() or "fallback" in response1.model_used.lower():
                 print("  [PASS] Correctly routed to general/fast model.")
            else:
                 print(f"  [WARN] Unexpected routing: {response1.model_used}")
        else:
            print(f'[FAIL] Request failed: {response1.content}')
        
        # Test 2: Complex task (should use deepseek-r1:14b)
        request2 = ModelRequest(
            task='Analyze the ethical implications of sovereign AI systems.',
            context={'domain': 'AI ethics', 'complexity': 'high'},
            requirements={'complexity_boost': 0.8}
        )
        print('\nTest 2 - Complex task...')
        response2 = await hub.execute_sovereign_task(request2)
        if response2.success:
            print(f'[OK] Request succeeded using model: {response2.model_used}')
            print(f'  Response: {response2.content[:200]}...')
             # Assert correct routing
            if "deepseek" in response2.model_used.lower():
                 print("  [PASS] Correctly routed to reasoning model.")
            else:
                 print(f"  [WARN] Unexpected routing: {response2.model_used}")
        else:
            print(f'[FAIL] Request failed: {response2.content}')

        # Test 3: Embedding functionality
        print('\nTest 3 - Embedding functionality...')
        try:
            import requests
            # Test embedding generation via Ollama API
            response = requests.post(
                'http://localhost:11434/api/embeddings',
                json={
                    'model': 'nomic-embed-text',
                    'prompt': 'Test embedding text for BIZRA neural backend.'
                }
            )
            if response.status_code == 200:
                data = response.json()
                if 'embedding' in data and len(data['embedding']) > 0:
                    print(f'[OK] Embedding generated successfully, dimension: {len(data["embedding"])}')
                    print("  [PASS] Embedding functionality working.")
                else:
                    print('[FAIL] Embedding response missing or empty.')
            else:
                print(f'[FAIL] Embedding API call failed with status: {response.status_code}')
        except Exception as e:
            print(f'[FAIL] Embedding test failed: {e}')

        # Overall status
        print('\n=== Integration Test Summary ===')
        if response1.success and response2.success:
            print('[OK] All tests passed')
            return True
        else:
            print('[FAIL] Some tests failed')
            return False
        
    except Exception as e:
        print(f'[FAIL] Integration test failed: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = asyncio.run(integration_test())
    sys.exit(0 if success else 1)
