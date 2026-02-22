# Equalizer Agent (SNR Homeostasis)

## Purpose
Add a cognitive-debt equalizer layer that detects ihsan/backlog drift and emits control commands before silent degradation.

## Implemented
1. `core/sovereign/equalizer_agent.py`
   - `EqualizerState(layer, deficit, backlog, presence)`
   - 5 modes: `accumulation`, `saturation`, `flow`, `recovery`, `steady`
   - Commands: `escalate`, `accelerate`, `halt`, `resume`
   - Saturation fail-closed invariant: `deficit >= 0.10` and `presence == 0` -> `halt(ihsan_critical)`
2. Tests in `tests/core/sovereign/test_equalizer_agent.py`

## Integration Points (next wiring step)
1. Telemetry source -> `observe(...)`
2. Command bus sink <- `next_command()`
3. UI widget: mode + deficit + backlog trend + presence
4. Optional threshold tuning from User Zero data

## Notes
- This module is additive and does not alter existing runtime execution paths.
- It is intended to be wired into runtime after release packaging baseline is stabilized.
