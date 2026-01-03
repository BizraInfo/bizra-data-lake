from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

class MetaPromptRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The primary query or topic")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    preferences: Dict[str, Any] = Field(default_factory=dict, description="User preferences")

    @field_validator("query")
    @classmethod
    def _non_blank_query(cls, v: str) -> str:
        v2 = (v or "").strip()
        if not v2:
            raise ValueError("query must be a non-empty string")
        return v2

class MetaPromptResult(BaseModel):
    content: Union[str, Dict[str, Any], List[Any]]
    source_agent: str = Field(..., min_length=1)
    timestamp: datetime
    success: bool = True
    error: Optional[str] = None

    @field_validator("source_agent")
    @classmethod
    def _non_blank_agent(cls, v: str) -> str:
        v2 = (v or "").strip()
        if not v2:
            raise ValueError("source_agent must be non-empty")
        return v2

    @field_validator("content", mode="before")
    @classmethod
    def _normalize_content(cls, v: Any) -> Union[str, Dict[str, Any], List[Any]]:
        if isinstance(v, (str, dict, list)):
            return v
        # Normalize unknown types to string to keep output serializable.
        return str(v)

class MetaPromptResponse(BaseModel):
    results: List[MetaPromptResult]
    explanation: str = Field(..., min_length=10)
    confidence: float = Field(..., ge=0.0, le=1.0)
    workflow_id: UUID
