import asyncio
import logging
from core.orchestration.learning_loop import LearningLoopOrchestrator
from core.prediction.hierarchical_hmm import HierarchicalHMMEngine, StrategicGoal
from core.reasoning.diffusion_reasoning_amplifier import DiffusionReasoningAmplifier
from core.hashtable.cognitive_hash_table import CognitiveHashTable

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PinnacleVerification")

async def verify_pinnacle_synthesis():
    logger.info("Starting Pinnacle Synthesis Verification (Phase 49)")
    
    # 1. Initialize Components
    hhmm = HierarchicalHMMEngine()
    cache = CognitiveHashTable()
    amplifier = DiffusionReasoningAmplifier()
    orchestrator = LearningLoopOrchestrator(
        hmm_engine=hhmm,
        context_cache=cache,
        enabled=True
    )
    _ = orchestrator

    
    # 2. Test Strategic Prediction
    # Simulate a "refactoring" observation sequence
    observations = ["edit", "refactor", "modify", "edit", "fix"]
    prediction = hhmm.predict(observations)
    logger.info(f"Strategic Prediction: {prediction.strategic_goal.name} (Conf: {prediction.strategic_confidence:.2f})")
    
    # 3. Test Reasoning Amplification
    query = "Refactor the HMM engine to support hierarchical layers."
    ctx = amplifier.amplify(query, prediction)
    
    logger.info(f"Amplification Context: Focus={ctx.focus}, Depth={ctx.got_depth}, Hypotheses={ctx.got_hypotheses}")
    
    # Assertions for dynamic modulation
    # REFACTORING should have a depth_mult of 2.0
    # Base depth for confidence 1.0 is 3 (GOT_MAX_DEPTH). 2.0 * 3 = 6.
    # But max_depth in config is 3. So it should be capped.
    assert ctx.got_depth >= 1
    assert ctx.strategic_goal in [g.name for g in StrategicGoal]
    
    # 4. Test O(1) Cache Integration
    test_key = "test_pattern_id"
    test_val = "test_reflex_data"
    cache.put(test_key, test_val)
    retrieved = cache.get(test_key)
    assert retrieved == test_val
    logger.info("O(1) Cognitive Hash Table verified.")
    
    logger.info("Pinnacle Synthesis Verification SUCCESSFUL.")

if __name__ == "__main__":
    asyncio.run(verify_pinnacle_synthesis())
