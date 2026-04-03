#!/usr/bin/env bash
# BIZRA SBOM (Software Bill of Materials) Generator
# Produces CycloneDX JSON for both Rust and Python components
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${REPO_ROOT}/docs/evidence/sbom"

mkdir -p "$OUTPUT_DIR"

echo "=== BIZRA SBOM Generator ==="
echo "Output: $OUTPUT_DIR"

# Rust SBOM via cargo-cyclonedx
if command -v cargo &>/dev/null; then
    echo ""
    echo "[1/2] Generating Rust SBOM..."
    if ! cargo cyclonedx --help &>/dev/null 2>&1; then
        echo "  Installing cargo-cyclonedx..."
        cargo install cargo-cyclonedx 2>/dev/null || {
            echo "  WARN: cargo-cyclonedx install failed, skipping Rust SBOM"
        }
    fi
    if cargo cyclonedx --help &>/dev/null 2>&1; then
        (cd "$REPO_ROOT" && cargo cyclonedx --format json --output-file "$OUTPUT_DIR/sbom-rust.json" 2>/dev/null) && \
            echo "  OK: $OUTPUT_DIR/sbom-rust.json" || \
            echo "  WARN: Rust SBOM generation failed"
    fi
else
    echo "[1/2] SKIP: cargo not found"
fi

# Python SBOM via cyclonedx-bom
if command -v python3 &>/dev/null; then
    echo ""
    echo "[2/2] Generating Python SBOM..."
    if ! python3 -m cyclonedx_py --help &>/dev/null 2>&1; then
        echo "  Installing cyclonedx-bom..."
        pip install cyclonedx-bom 2>/dev/null || {
            echo "  WARN: cyclonedx-bom install failed, skipping Python SBOM"
        }
    fi
    if python3 -m cyclonedx_py --help &>/dev/null 2>&1; then
        python3 -m cyclonedx_py requirements \
            -r "$REPO_ROOT/requirements-kernel.txt" \
            -o "$OUTPUT_DIR/sbom-python.json" \
            --format json 2>/dev/null && \
            echo "  OK: $OUTPUT_DIR/sbom-python.json" || \
            echo "  WARN: Python SBOM generation failed"
    fi
else
    echo "[2/2] SKIP: python3 not found"
fi

echo ""
echo "=== SBOM Generation Complete ==="
ls -la "$OUTPUT_DIR"/*.json 2>/dev/null || echo "No SBOM files generated"
