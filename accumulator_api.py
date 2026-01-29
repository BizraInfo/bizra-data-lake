"""
BIZRA ACCUMULATOR API
═══════════════════════════════════════════════════════════════════════════════

FastAPI endpoints for the BIZRA Accumulator.

Integrates with:
- Flywheel (autopoietic inference loop)
- PoI Ledger (proof-of-impact chain)
- Zakat Distribution (charitable outflow)
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

from accumulator import (
    get_accumulator,
    BizraAccumulator,
    ImpactCategory,
    Seed,
    Bloom,
    Fruit,
    ProofOfImpact,
    ACCUMULATOR_VERSION,
)

# ═══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ═══════════════════════════════════════════════════════════════════════════════

router = APIRouter(prefix="/accumulator", tags=["accumulator"])


# ═══════════════════════════════════════════════════════════════════════════════
# REQUEST/RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class SeedRequest(BaseModel):
    """Request to plant a seed."""
    contributor: str = Field(..., description="Contributor identifier")
    category: str = Field(..., description="Impact category")
    amount: float = Field(..., gt=0, description="Seed amount")
    metadata: Optional[Dict[str, Any]] = Field(default=None)


class SeedResponse(BaseModel):
    """Seed response."""
    seed_id: str
    contributor: str
    category: str
    amount: float
    initial_bloom: float
    attestation_hash: str


class ImpactRequest(BaseModel):
    """Request to record impact."""
    contributor: str
    action: str
    category: str
    impact_score: float = Field(..., gt=0)
    resources: Optional[Dict[str, Any]] = None
    benchmarks: Optional[Dict[str, float]] = None


class ComputationImpactRequest(BaseModel):
    """Request to record computation impact."""
    contributor: str
    tokens_processed: int = Field(..., gt=0)
    latency_ms: float = Field(..., gt=0)
    model: str = "unknown"


class KnowledgeImpactRequest(BaseModel):
    """Request to record knowledge impact."""
    contributor: str
    documents_processed: int = Field(..., gt=0)
    synthesis_quality: float = Field(..., ge=0, le=1)
    sources: Optional[List[str]] = None


class CodeImpactRequest(BaseModel):
    """Request to record code impact."""
    contributor: str
    lines_changed: int = Field(..., ge=0)
    test_coverage: float = Field(..., ge=0, le=1)
    bugs_fixed: int = Field(default=0, ge=0)
    commit_hash: Optional[str] = None


class EthicsImpactRequest(BaseModel):
    """Request to record ethics impact."""
    contributor: str
    reviews_completed: int = Field(..., gt=0)
    violations_caught: int = Field(default=0, ge=0)


class CommunityImpactRequest(BaseModel):
    """Request to record community impact."""
    contributor: str
    users_helped: int = Field(..., gt=0)
    satisfaction_score: float = Field(..., ge=0, le=1)


class BloomResponse(BaseModel):
    """Bloom status response."""
    contributor: str
    total_bloom: float
    category_bloom: Dict[str, float]
    streak_days: int
    multiplier: float
    harvestable: bool


class FruitResponse(BaseModel):
    """Fruit harvest response."""
    fruit_id: str
    contributor: str
    bloom_source: float
    zakat_portion: float
    net_fruit: float


class ZakatDistributionRequest(BaseModel):
    """Request to distribute zakat."""
    recipients: List[str]
    amounts: Optional[List[float]] = None


class ZakatDistributionResponse(BaseModel):
    """Zakat distribution response."""
    distributions: Dict[str, float]
    remaining_pool: float


class StatusResponse(BaseModel):
    """Accumulator status response."""
    version: str
    state: str
    total_seeds: int
    total_contributors: int
    total_bloom: float
    total_fruits: int
    zakat_pool: float
    poi_attestations: int


class LeaderboardEntry(BaseModel):
    """Leaderboard entry."""
    rank: int
    contributor: str
    total_bloom: float
    categories: Dict[str, float]
    streak_days: int
    multiplier: float
    harvestable: bool


# ═══════════════════════════════════════════════════════════════════════════════
# DEPENDENCY
# ═══════════════════════════════════════════════════════════════════════════════

def get_acc() -> BizraAccumulator:
    """Get accumulator instance."""
    return get_accumulator()


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/status", response_model=StatusResponse)
async def get_status(acc: BizraAccumulator = Depends(get_acc)):
    """Get accumulator status."""
    status = acc.status()
    return StatusResponse(
        version=status["version"],
        state=status["state"],
        total_seeds=status["total_seeds"],
        total_contributors=status["total_contributors"],
        total_bloom=status["total_bloom"],
        total_fruits=status["total_fruits"],
        zakat_pool=status["zakat_pool"],
        poi_attestations=status["poi_attestations"],
    )


@router.get("/leaderboard", response_model=List[LeaderboardEntry])
async def get_leaderboard(
    limit: int = 10,
    acc: BizraAccumulator = Depends(get_acc),
):
    """Get contributor leaderboard."""
    leaders = acc.leaderboard(limit=limit)
    return [LeaderboardEntry(**entry) for entry in leaders]


@router.post("/seed", response_model=SeedResponse)
async def plant_seed(
    request: SeedRequest,
    acc: BizraAccumulator = Depends(get_acc),
):
    """Plant a seed to start accumulation."""
    try:
        category = ImpactCategory(request.category)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid category. Must be one of: {[c.value for c in ImpactCategory]}",
        )
    
    seed = acc.plant_seed(
        contributor=request.contributor,
        category=category,
        amount=request.amount,
        metadata=request.metadata,
    )
    
    bloom = acc.get_bloom(request.contributor)
    
    return SeedResponse(
        seed_id=seed.seed_id,
        contributor=seed.contributor,
        category=seed.category.value,
        amount=seed.amount,
        initial_bloom=bloom.total_bloom if bloom else 0.0,
        attestation_hash=seed.attestation_hash,
    )


@router.post("/impact/generic")
async def record_generic_impact(
    request: ImpactRequest,
    acc: BizraAccumulator = Depends(get_acc),
):
    """Record generic impact."""
    try:
        category = ImpactCategory(request.category)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid category. Must be one of: {[c.value for c in ImpactCategory]}",
        )
    
    bloom_added = acc.record_impact(
        contributor=request.contributor,
        action=request.action,
        category=category,
        impact_score=request.impact_score,
        resources=request.resources,
        benchmarks=request.benchmarks,
    )
    
    return {"bloom_added": bloom_added}


@router.post("/impact/computation")
async def record_computation_impact(
    request: ComputationImpactRequest,
    acc: BizraAccumulator = Depends(get_acc),
):
    """Record computation impact (LLM inference, processing)."""
    bloom_added = acc.record_computation(
        contributor=request.contributor,
        tokens_processed=request.tokens_processed,
        latency_ms=request.latency_ms,
        model=request.model,
    )
    return {"bloom_added": bloom_added, "category": "computation"}


@router.post("/impact/knowledge")
async def record_knowledge_impact(
    request: KnowledgeImpactRequest,
    acc: BizraAccumulator = Depends(get_acc),
):
    """Record knowledge synthesis impact."""
    bloom_added = acc.record_knowledge(
        contributor=request.contributor,
        documents_processed=request.documents_processed,
        synthesis_quality=request.synthesis_quality,
        sources=request.sources,
    )
    return {"bloom_added": bloom_added, "category": "knowledge"}


@router.post("/impact/code")
async def record_code_impact(
    request: CodeImpactRequest,
    acc: BizraAccumulator = Depends(get_acc),
):
    """Record code contribution impact."""
    bloom_added = acc.record_code(
        contributor=request.contributor,
        lines_changed=request.lines_changed,
        test_coverage=request.test_coverage,
        bugs_fixed=request.bugs_fixed,
        commit_hash=request.commit_hash,
    )
    return {"bloom_added": bloom_added, "category": "code"}


@router.post("/impact/ethics")
async def record_ethics_impact(
    request: EthicsImpactRequest,
    acc: BizraAccumulator = Depends(get_acc),
):
    """Record ethics/compliance impact."""
    bloom_added = acc.record_ethics(
        contributor=request.contributor,
        reviews_completed=request.reviews_completed,
        violations_caught=request.violations_caught,
    )
    return {"bloom_added": bloom_added, "category": "ethics"}


@router.post("/impact/community")
async def record_community_impact(
    request: CommunityImpactRequest,
    acc: BizraAccumulator = Depends(get_acc),
):
    """Record community contribution impact."""
    bloom_added = acc.record_community(
        contributor=request.contributor,
        users_helped=request.users_helped,
        satisfaction_score=request.satisfaction_score,
    )
    return {"bloom_added": bloom_added, "category": "community"}


@router.get("/bloom/{contributor}", response_model=BloomResponse)
async def get_contributor_bloom(
    contributor: str,
    acc: BizraAccumulator = Depends(get_acc),
):
    """Get bloom status for a contributor."""
    bloom = acc.get_bloom(contributor)
    if not bloom:
        raise HTTPException(status_code=404, detail="Contributor not found")
    
    return BloomResponse(
        contributor=bloom.contributor,
        total_bloom=bloom.total_bloom,
        category_bloom=bloom.category_bloom,
        streak_days=bloom.streak_days,
        multiplier=bloom.multiplier,
        harvestable=acc.check_fruitability(contributor),
    )


@router.post("/harvest/{contributor}", response_model=FruitResponse)
async def harvest_fruit(
    contributor: str,
    acc: BizraAccumulator = Depends(get_acc),
):
    """Harvest fruit from accumulated bloom."""
    fruit = acc.harvest_fruit(contributor)
    if not fruit:
        raise HTTPException(
            status_code=400,
            detail="Cannot harvest: insufficient bloom or contributor not found",
        )
    
    return FruitResponse(
        fruit_id=fruit.fruit_id,
        contributor=fruit.contributor,
        bloom_source=fruit.bloom_source,
        zakat_portion=fruit.zakat_portion,
        net_fruit=fruit.net_fruit,
    )


@router.get("/zakat/pool")
async def get_zakat_pool(acc: BizraAccumulator = Depends(get_acc)):
    """Get current zakat pool balance."""
    return {"zakat_pool": acc.zakat_pool}


@router.post("/zakat/distribute", response_model=ZakatDistributionResponse)
async def distribute_zakat(
    request: ZakatDistributionRequest,
    acc: BizraAccumulator = Depends(get_acc),
):
    """Distribute zakat to recipients."""
    if not request.recipients:
        raise HTTPException(status_code=400, detail="At least one recipient required")
    
    distributions = acc.distribute_zakat(
        recipients=request.recipients,
        amounts=request.amounts,
    )
    
    return ZakatDistributionResponse(
        distributions=distributions,
        remaining_pool=acc.zakat_pool,
    )


@router.get("/poi/count")
async def get_poi_count(acc: BizraAccumulator = Depends(get_acc)):
    """Get count of PoI attestations."""
    return {"count": len(acc.poi_ledger)}


@router.get("/poi/recent")
async def get_recent_poi(
    limit: int = 10,
    acc: BizraAccumulator = Depends(get_acc),
):
    """Get recent PoI attestations."""
    return {"attestations": acc.poi_ledger[-limit:]}
