# BIZRA Genesis — Verify It Yourself

**Don't trust us. Verify.**

This is the fastest path from `git clone` to constitutional proof.

## Prerequisites

- Rust 1.91+ (`rustup update stable`)
- Python 3.12+ with pip
- Ollama running locally (`ollama serve`) with at least one model

## Step 1: Clone and test the constitutional core (2 min)

```bash
git clone https://github.com/BizraInfo/bizra-data-lake.git
cd bizra-data-lake/bizra-omega
cargo test -p bizra-protocol -p bizra-sippar -p bizra-hooks
```

Expected: **130 tests passed, 0 failed.**

These three crates are the constitutional skeleton:
- `bizra-hooks` (76 tests) — the nervous system, zero external dependencies
- `bizra-protocol` (31 tests) — the 5-phase proof flow (Mint → Execute → Cross → Validate → Propagate)
- `bizra-sippar` (21 tests) — Babylonian exact arithmetic for zero-drift economics

## Step 2: Compile the Python↔Rust bridge (5 min)

```bash
cd bizra-python
pip install maturin
maturin develop --release
cd ../..
```

Or manually: copy `bizra-omega/target/release/bizra.dll` to `bizra.pyd` in the repo root.

Verify:
```python
python -c "import bizra; print(f'bizra v{bizra.__version__}')"
# Expected: bizra v2.0.0
```

## Step 3: Run the synapse proof (instant)

```bash
python first_breath.py
```

Expected output:
```
FIRST BREATH: SUCCESS

5 Python cognitive events crossed to Rust
3 topic translations applied
0 failures, chain intact, bridge healthy

The language boundary IS the trust boundary.
PAT (Python) served the user.
SAT (Rust) validated independently.
The organism breathes.
```

## Step 4: Run a constitutional mission (requires Ollama)

```bash
ollama pull llama3.1:8b   # or any model
python genesis_mission.py
```

This sends a real query to your local LLM through the full constitutional pipeline:
- Python EventBus emits `action.intent`
- Ollama processes the query
- Rust SNR engine measures response quality
- Ihsan gate evaluates against 0.95 threshold
- BLAKE3 receipt is saved to `evidence/`

If the LLM response scores below Ihsan 0.95, the gate **correctly rejects** it.
That rejection is the proof that constitutional governance works.

## Step 5: Run the full test suite

```bash
# Rust (26 crates)
cd bizra-omega
cargo test --workspace --exclude fate-binding --exclude iceoryx-bridge --exclude bizra-python

# Python (synapse)
cd ..
python tests/test_rust_bridge.py
python tests/test_rust_bridge_v2.py
```

## What you're verifying

| Claim | Evidence | Command |
|-------|---------|---------|
| Constitutional proof chain | BLAKE3 + Ed25519 | `cargo test -p bizra-protocol` |
| Zero-drift economics | Babylonian exact arithmetic | `cargo test -p bizra-sippar` |
| Independent verification | PAT/SAT dual pipeline | `python first_breath.py` |
| Real LLM governance | Ihsan gate on live output | `python genesis_mission.py` |
| Nervous system integrity | 76 tests, zero deps | `cargo test -p bizra-hooks` |

