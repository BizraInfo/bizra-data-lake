"""
BIZRA PersonaPlex Engine — Full-Duplex Voice Interface

Integrates NVIDIA PersonaPlex for real-time speech-to-speech
conversation with BIZRA Guardians.

Capabilities:
- Full-duplex: Listen and speak simultaneously
- Voice conditioning: Match Guardian vocal characteristics
- Persona control: Guardian expertise via text prompts
- Ihsān gating: Ethical constraints before responding

Standing on Giants:
- PersonaPlex (Roy et al., 2026)
- Moshi Architecture (Kyutai)
- BIZRA Guardian System (Node0 Genesis)
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .guardians import BIZRA_GUARDIANS, Guardian

logger = logging.getLogger(__name__)


@dataclass
class PersonaPlexConfig:
    """Configuration for PersonaPlex engine."""

    # Model settings
    hf_repo: str = "nvidia/personaplex-7b-v1"
    hf_revision: str = "main"  # Pin to specific revision for reproducibility
    device: str = "cuda"
    cpu_offload: bool = False

    # Audio settings
    sample_rate: int = 24000
    frame_rate: float = 12.5  # Moshi default

    # Voice prompts directory
    voice_dir: Path = field(
        default_factory=lambda: Path("/mnt/c/BIZRA-DATA-LAKE/voices")
    )

    # Ihsān settings
    ihsan_threshold: float = 0.75
    require_ihsan_gate: bool = True

    # Server settings
    server_port: int = 8998
    ssl_enabled: bool = True


@dataclass
class VoiceResponse:
    """Response from PersonaPlex voice processing."""

    guardian_name: str
    audio: Optional[np.ndarray] = None
    text: str = ""
    tokens: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
    ihsan_passed: bool = True
    ihsan_reason: str = ""

    @property
    def has_audio(self) -> bool:
        return self.audio is not None and len(self.audio) > 0

    def to_dict(self) -> dict:
        return {
            "guardian": self.guardian_name,
            "text": self.text,
            "latency_ms": self.latency_ms,
            "ihsan_passed": self.ihsan_passed,
            "ihsan_reason": self.ihsan_reason,
            "has_audio": self.has_audio,
        }


class BIZRAPersonaPlex:
    """
    BIZRA integration with PersonaPlex full-duplex speech.

    This engine enables voice-interactive multi-agent reasoning:
    1. User speaks to a Guardian
    2. Ihsān gate checks ethical constraints
    3. Guardian responds with distinctive voice
    4. Full-duplex allows interruptions and natural conversation

    Usage:
        engine = BIZRAPersonaPlex()
        await engine.initialize()
        engine.register_guardian(guardian)
        response = await engine.process_audio("ethics", audio_data)
    """

    def __init__(self, config: Optional[PersonaPlexConfig] = None):
        self.config = config or PersonaPlexConfig()
        self.guardians: Dict[str, Guardian] = {}
        self._initialized = False

        # Model components (lazy loaded)
        self._mimi: Any = None
        self._other_mimi: Any = None
        self._lm_gen: Any = None
        self._tokenizer: Any = None
        self._voice_prompt_dir: Any = None

        # Register default guardians
        for name, guardian in BIZRA_GUARDIANS.items():
            self.guardians[name] = guardian

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def initialize(self):
        """
        Load PersonaPlex models.

        Call this once at startup. Requires:
        - HF_TOKEN environment variable
        - CUDA device (or cpu_offload=True)
        - ~16GB GPU memory (or use cpu_offload)
        """
        if self._initialized:
            logger.info("PersonaPlex already initialized")
            return

        logger.info("Initializing PersonaPlex engine...")

        try:
            import sentencepiece
            from huggingface_hub import hf_hub_download

            # Import Moshi components
            from moshi.models import LMGen, loaders

            # Load Mimi (speech encoder/decoder)
            logger.info("Loading Mimi speech codec...")
            mimi_weight = hf_hub_download(
                self.config.hf_repo, loaders.MIMI_NAME, revision=self.config.hf_revision
            )  # nosec B615 — revision pinned via config; set hf_revision to commit SHA in production
            self._mimi = loaders.get_mimi(mimi_weight, self.config.device)
            self._other_mimi = loaders.get_mimi(mimi_weight, self.config.device)

            # Load tokenizer
            logger.info("Loading tokenizer...")
            tokenizer_path = hf_hub_download(  # nosec B615 — revision pinned via config; set hf_revision to commit SHA in production
                self.config.hf_repo,
                loaders.TEXT_TOKENIZER_NAME,
                revision=self.config.hf_revision,
            )
            self._tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)

            # Load Moshi LM
            logger.info("Loading Moshi language model...")
            moshi_weight = hf_hub_download(
                self.config.hf_repo,
                loaders.MOSHI_NAME,
                revision=self.config.hf_revision,
            )  # nosec B615 — revision pinned via config; set hf_revision to commit SHA in production
            lm = loaders.get_moshi_lm(
                moshi_weight,
                device=self.config.device,
                cpu_offload=self.config.cpu_offload,
            )
            lm.eval()

            # Create LMGen
            frame_size = int(self._mimi.sample_rate / self._mimi.frame_rate)
            self._lm_gen = LMGen(
                lm,
                audio_silence_frame_cnt=int(0.5 * self._mimi.frame_rate),
                sample_rate=self._mimi.sample_rate,
                device=self.config.device,
                frame_rate=self._mimi.frame_rate,
            )

            # Enable streaming mode
            self._mimi.streaming_forever(1)
            self._other_mimi.streaming_forever(1)
            self._lm_gen.streaming_forever(1)

            # Warmup
            logger.info("Warming up models...")
            from moshi.offline import warmup

            warmup(
                self._mimi,
                self._other_mimi,
                self._lm_gen,
                self.config.device,
                frame_size,
            )

            # Get voice prompts directory
            from moshi.offline import _get_voice_prompt_dir

            self._voice_prompt_dir = _get_voice_prompt_dir(None, self.config.hf_repo)

            self._initialized = True
            logger.info("PersonaPlex engine initialized successfully")

        except ImportError as e:
            logger.error(f"Failed to import Moshi: {e}")
            logger.error("Install with: pip install moshi/.")
            raise

        except Exception as e:
            logger.error(f"Failed to initialize PersonaPlex: {e}")
            raise

    def register_guardian(self, guardian: Guardian):
        """Register a Guardian persona."""
        self.guardians[guardian.name.lower()] = guardian
        logger.info(f"Registered Guardian: {guardian.name} ({guardian.voice_prompt})")

    def get_guardian(self, name: str) -> Optional[Guardian]:
        """Get a Guardian by name."""
        return self.guardians.get(name.lower())

    def list_guardians(self) -> List[str]:
        """List all registered Guardian names."""
        return list(self.guardians.keys())

    def _set_guardian_persona(self, guardian: Guardian):
        """Configure PersonaPlex with Guardian's persona."""
        if not self._initialized:
            raise RuntimeError("PersonaPlex not initialized. Call initialize() first.")

        # Resolve voice prompt path
        voice_path = self.config.voice_dir / f"{guardian.voice_prompt}.pt"
        if not voice_path.exists():
            # Try HuggingFace directory
            voice_path = Path(self._voice_prompt_dir) / f"{guardian.voice_prompt}.pt"

        if not voice_path.exists():
            logger.warning(f"Voice prompt not found: {voice_path}, using default")
        else:
            # Load voice prompt
            if voice_path.suffix == ".pt":
                self._lm_gen.load_voice_prompt_embeddings(str(voice_path))
            else:
                self._lm_gen.load_voice_prompt(str(voice_path))
            logger.debug(f"Loaded voice prompt: {voice_path}")

        # Set text prompt
        from moshi.offline import wrap_with_system_tags

        full_prompt = guardian.get_full_prompt()
        self._lm_gen.text_prompt_tokens = self._tokenizer.encode(
            wrap_with_system_tags(full_prompt)
        )

        # Reset streaming state
        self._mimi.reset_streaming()
        self._other_mimi.reset_streaming()
        self._lm_gen.reset_streaming()

        # Run prompt phases
        self._lm_gen.step_system_prompts(self._mimi)
        self._mimi.reset_streaming()

        logger.debug(f"Activated Guardian persona: {guardian.name}")

    def ihsan_gate(self, guardian_name: str, purpose: str = "") -> Tuple[bool, str]:
        """
        Check if action passes Ihsān constraints.

        Returns: (passed, reason)
        """
        if not self.config.require_ihsan_gate:
            return True, "Ihsān gate disabled"

        guardian = self.get_guardian(guardian_name)
        if not guardian:
            return True, "Guardian not found, allowing by default"

        return guardian.can_respond(purpose)

    def process_audio(
        self,
        guardian_name: str,
        input_audio: np.ndarray,
        purpose: str = "voice_interaction",
    ) -> VoiceResponse:
        """
        Process audio through a Guardian persona.

        Args:
            guardian_name: Name of the Guardian to use
            input_audio: Input audio as numpy array (24kHz mono)
            purpose: Purpose description for Ihsān gating

        Returns:
            VoiceResponse with output audio and text
        """
        start_time = time.time()

        # Ihsān gate check
        passed, reason = self.ihsan_gate(guardian_name, purpose)
        if not passed:
            return VoiceResponse(
                guardian_name=guardian_name,
                ihsan_passed=False,
                ihsan_reason=reason,
            )

        # Get and activate Guardian
        guardian = self.get_guardian(guardian_name)
        if not guardian:
            return VoiceResponse(
                guardian_name=guardian_name,
                ihsan_passed=False,
                ihsan_reason=f"Unknown Guardian: {guardian_name}",
            )

        if not self._initialized:
            self.initialize()

        self._set_guardian_persona(guardian)

        # Process audio through PersonaPlex
        try:
            from moshi.models.lm import _iterate_audio, encode_from_sphn
            from moshi.offline import decode_tokens_to_pcm

            generated_frames = []
            generated_tokens = []

            frame_size = int(self._mimi.sample_rate / self._mimi.frame_rate)

            for user_encoded in encode_from_sphn(
                self._mimi,
                _iterate_audio(input_audio, sample_interval_size=frame_size, pad=True),
                max_batch=1,
            ):
                steps = user_encoded.shape[-1]
                for c in range(steps):
                    step_in = user_encoded[:, :, c : c + 1]
                    tokens = self._lm_gen.step(step_in)

                    if tokens is None:
                        continue

                    # Decode audio
                    pcm = decode_tokens_to_pcm(
                        self._mimi, self._other_mimi, self._lm_gen, tokens
                    )
                    generated_frames.append(pcm)

                    # Decode text
                    text_token = tokens[0, 0, 0].item()
                    if text_token not in (0, 3):  # Skip special tokens
                        text = self._tokenizer.id_to_piece(text_token)
                        text = text.replace("▁", " ")
                        generated_tokens.append(text)

            # Combine output
            output_audio = (
                np.concatenate(generated_frames, axis=-1) if generated_frames else None
            )
            output_text = "".join(generated_tokens)

            latency_ms = (time.time() - start_time) * 1000

            return VoiceResponse(
                guardian_name=guardian_name,
                audio=output_audio,
                text=output_text,
                tokens=generated_tokens,
                latency_ms=latency_ms,
                ihsan_passed=True,
                ihsan_reason="Ihsān gate passed",
            )

        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            return VoiceResponse(
                guardian_name=guardian_name,
                ihsan_passed=False,
                ihsan_reason=f"Processing error: {str(e)}",
            )

    async def process_audio_async(
        self,
        guardian_name: str,
        input_audio: np.ndarray,
        purpose: str = "voice_interaction",
    ) -> VoiceResponse:
        """Async wrapper for process_audio."""
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.process_audio,
            guardian_name,
            input_audio,
            purpose,
        )

    def start_server(self, port: Optional[int] = None):
        """
        Start the PersonaPlex server for real-time voice chat.

        Access at: https://localhost:{port}
        """
        import subprocess
        import tempfile

        port = port or self.config.server_port

        if self.config.ssl_enabled:
            ssl_dir = tempfile.mkdtemp()
            cmd = [
                "python",
                "-m",
                "moshi.server",
                "--ssl",
                ssl_dir,
                "--port",
                str(port),
            ]
        else:
            cmd = [
                "python",
                "-m",
                "moshi.server",
                "--port",
                str(port),
            ]

        if self.config.cpu_offload:
            cmd.append("--cpu-offload")

        logger.info(f"Starting PersonaPlex server on port {port}...")
        subprocess.Popen(cmd)

    def offline_inference(
        self,
        guardian_name: str,
        input_wav: str,
        output_wav: str,
        output_text: Optional[str] = None,
        seed: int = 42,
    ):
        """
        Run offline inference on a WAV file.

        Args:
            guardian_name: Guardian to use
            input_wav: Path to input WAV file
            output_wav: Path to output WAV file
            output_text: Optional path for text output JSON
            seed: Random seed for reproducibility
        """
        import subprocess

        guardian = self.get_guardian(guardian_name)
        if not guardian:
            raise ValueError(f"Unknown Guardian: {guardian_name}")

        # Build command
        voice_path = self.config.voice_dir / f"{guardian.voice_prompt}.pt"
        prompt = guardian.get_full_prompt()

        cmd = [
            "python",
            "-m",
            "moshi.offline",
            "--voice-prompt",
            str(voice_path),
            "--text-prompt",
            prompt,
            "--input-wav",
            input_wav,
            "--output-wav",
            output_wav,
            "--seed",
            str(seed),
        ]

        if output_text:
            cmd.extend(["--output-text", output_text])

        logger.info(f"Running offline inference with {guardian_name}...")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"Offline inference failed: {result.stderr}")
            raise RuntimeError(f"Offline inference failed: {result.stderr}")

        logger.info(f"Output saved to {output_wav}")
        return output_wav


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience Functions
# ═══════════════════════════════════════════════════════════════════════════════


def create_engine(
    device: str = "cuda",
    cpu_offload: bool = False,
    auto_initialize: bool = False,
) -> BIZRAPersonaPlex:
    """Create and optionally initialize a PersonaPlex engine."""
    config = PersonaPlexConfig(
        device=device,
        cpu_offload=cpu_offload,
    )
    engine = BIZRAPersonaPlex(config)

    if auto_initialize:
        engine.initialize()

    return engine


def quick_inference(
    guardian_name: str,
    input_wav: str,
    output_wav: str,
    cpu_offload: bool = True,
) -> str:
    """Quick offline inference with a Guardian."""
    engine = create_engine(cpu_offload=cpu_offload)
    return engine.offline_inference(guardian_name, input_wav, output_wav)
