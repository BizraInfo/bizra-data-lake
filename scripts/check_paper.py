"""Convert the CMN v4 markdown to PDF using Desktop Commander's write_pdf via subprocess."""

from pathlib import Path

src = Path(
    r"B:\BIZRA-SOVEREIGN\10_BIZRA-LAB\publications\cmn_preprint\CMN_v4_DEFINITIVE.md"
)
content = src.read_text(encoding="utf-8")
print(f"Source: {len(content)} chars, {len(content.splitlines())} lines")
print("Content loaded successfully. Ready for PDF generation.")
