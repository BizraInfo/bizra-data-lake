// src/voice.rs - Offline Voice I/O (STT/TTS)
//
// Supports local, offline pipelines using external binaries:
// - STT: whisper.cpp (BIZRA_WHISPER_BIN + BIZRA_WHISPER_MODEL)
// - TTS: piper (BIZRA_PIPER_BIN + BIZRA_PIPER_MODEL)
//
// These are optional and fail-closed if not configured.

use anyhow::{anyhow, Context, Result};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct VoiceConfig {
    pub whisper_bin: Option<String>,
    pub whisper_model: Option<String>,
    pub piper_bin: Option<String>,
    pub piper_model: Option<String>,
}

impl VoiceConfig {
    pub fn from_env() -> Self {
        Self {
            whisper_bin: std::env::var("BIZRA_WHISPER_BIN").ok(),
            whisper_model: std::env::var("BIZRA_WHISPER_MODEL").ok(),
            piper_bin: std::env::var("BIZRA_PIPER_BIN").ok(),
            piper_model: std::env::var("BIZRA_PIPER_MODEL").ok(),
        }
    }
}

fn temp_path(prefix: &str, ext: &str) -> PathBuf {
    let filename = format!("{}_{}.{}", prefix, Uuid::new_v4(), ext);
    std::env::temp_dir().join(filename)
}

fn ensure_exists(path: &Path, label: &str) -> Result<()> {
    if path.exists() {
        Ok(())
    } else {
        Err(anyhow!("{label} not found: {}", path.display()))
    }
}

pub fn transcribe_sync(audio_bytes: &[u8]) -> Result<String> {
    let config = VoiceConfig::from_env();
    let whisper_bin = config
        .whisper_bin
        .ok_or_else(|| anyhow!("BIZRA_WHISPER_BIN not set"))?;
    let whisper_model = config
        .whisper_model
        .ok_or_else(|| anyhow!("BIZRA_WHISPER_MODEL not set"))?;

    let input_path = temp_path("bizra_audio", "wav");
    let output_path = temp_path("bizra_transcript", "txt");
    let output_base = output_path.with_extension("");

    fs::write(&input_path, audio_bytes)
        .with_context(|| format!("Failed to write audio to {}", input_path.display()))?;

    ensure_exists(Path::new(&whisper_bin), "whisper.cpp binary")?;
    ensure_exists(Path::new(&whisper_model), "whisper.cpp model")?;

    let status = Command::new(&whisper_bin)
        .args([
            "-m",
            &whisper_model,
            "-f",
            input_path
                .to_str()
                .ok_or_else(|| anyhow!("Invalid input path"))?,
            "-of",
            output_base
                .to_str()
                .ok_or_else(|| anyhow!("Invalid output path"))?,
            "-otxt",
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .context("Failed to run whisper.cpp")?;

    if !status.success() {
        return Err(anyhow!("whisper.cpp failed with status: {status}"));
    }

    let transcript_path = output_base.with_extension("txt");
    let transcript = fs::read_to_string(&transcript_path).with_context(|| {
        format!(
            "Failed to read transcript from {}",
            transcript_path.display()
        )
    })?;

    let _ = fs::remove_file(input_path);
    let _ = fs::remove_file(transcript_path);

    Ok(transcript.trim().to_string())
}

pub fn synthesize_sync(text: &str) -> Result<Vec<u8>> {
    let config = VoiceConfig::from_env();
    let piper_bin = config
        .piper_bin
        .ok_or_else(|| anyhow!("BIZRA_PIPER_BIN not set"))?;
    let piper_model = config
        .piper_model
        .ok_or_else(|| anyhow!("BIZRA_PIPER_MODEL not set"))?;

    ensure_exists(Path::new(&piper_bin), "piper binary")?;
    ensure_exists(Path::new(&piper_model), "piper model")?;

    let output_path = temp_path("bizra_tts", "wav");

    let mut child = Command::new(&piper_bin)
        .args([
            "-m",
            &piper_model,
            "-f",
            output_path
                .to_str()
                .ok_or_else(|| anyhow!("Invalid output path"))?,
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .context("Failed to start piper")?;

    if let Some(mut stdin) = child.stdin.take() {
        use std::io::Write;
        stdin
            .write_all(text.as_bytes())
            .context("Failed to write text to piper stdin")?;
    }

    let status = child.wait().context("Failed to wait for piper")?;
    if !status.success() {
        return Err(anyhow!("piper failed with status: {status}"));
    }

    let audio = fs::read(&output_path)
        .with_context(|| format!("Failed to read audio from {}", output_path.display()))?;
    let _ = fs::remove_file(output_path);

    Ok(audio)
}
