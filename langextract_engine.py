# BIZRA Extraction Engine (Layer 4: Deterministic Extraction)
# Uses Google's LangExtract for high-precision source grounding
# Engineering Excellence: Verbatim extraction + Provenance mapping

import langextract as lx
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
from bizra_config import CORPUS_TABLE_PATH, EXTRACTION_MODEL, EXTRACTION_PROMPT

class ExtractionEngine:
    def __init__(self):
        print("🔍 Initializing BIZRA Extraction Engine (LangExtract)")
        self.model_id = EXTRACTION_MODEL
        self.prompt = EXTRACTION_PROMPT
        
        # High-quality example to guide the model (As per LangExtract best practices)
        self.examples = [
            lx.data.ExampleData(
                text="The BIZRA system was initiated on Oct 2025. It uses an RTX 4090 for GPU acceleration.",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="milestone",
                        extraction_text="initiated on Oct 2025",
                        attributes={"event": "Inception"}
                    ),
                    lx.data.Extraction(
                        extraction_class="hardware",
                        extraction_text="RTX 4090",
                        attributes={"type": "GPU"}
                    )
                ]
            )
        ]

    def extract_from_corpus(self, limit=5):
        """Processes documents from the corpus and extracts structured facts."""
        if not CORPUS_TABLE_PATH.exists():
            print("❌ Corpus table missing.")
            return

        df = pd.read_parquet(CORPUS_TABLE_PATH)
        # Limit to first few for demo/cost control
        targets = df[df['text_quality'] != 'no_text'].head(limit)
        
        results = []
        for _, row in tqdm(targets.iterrows(), total=len(targets), desc="LangExtract Processing"):
            try:
                result = lx.extract(
                    text_or_documents=row['text'],
                    prompt_description=self.prompt,
                    examples=self.examples,
                    model_id=self.model_id
                )
                
                # result is an lx.data.AnnotatedDocument
                # We save it for Layer 3 (Hypergraph) consumption
                results.append({
                    "doc_id": row['doc_id'],
                    "extractions": [
                        {
                            "class": e.extraction_class,
                            "text": e.extraction_text,
                            "attributes": e.attributes
                        } for e in result.extractions
                    ]
                })
            except Exception as e:
                print(f"⚠️ Extraction failed for {row['title']}: {e}")

        # Save to a dedicated index
        output_path = CORPUS_TABLE_PATH.parent / "assertions.jsonl"
        with open(output_path, 'w') as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        
        print(f"✅ Extraction complete. Saved to: {output_path}")

if __name__ == "__main__":
    engine = ExtractionEngine()
    engine.extract_from_corpus()
