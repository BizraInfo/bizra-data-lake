from typing import Dict, List, Any


class GoTOrchestrator:
    """
    Graph of Thoughts orchestrator with a security-first lens.
    """

    SECURITY_KEYWORDS = (
        "security",
        "secure",
        "zero trust",
        "defense in depth",
        "least privilege",
        "audit trail",
        "hardening",
        "safe",
        "safety",
    )
    ETHICAL_KEYWORDS = (
        "ethic",
        "ethics",
        "ihsan",
        "governance",
        "dignity",
        "fairness",
        "compliance",
    )

    def analyze(self, prompt: str, got_links: List[Any]) -> Dict[str, Any]:
        text = (prompt or "").lower()
        lenses: List[str] = ["Technical"]

        if any(keyword in text for keyword in self.ETHICAL_KEYWORDS):
            lenses.append("Ethical")
        if any(keyword in text for keyword in self.SECURITY_KEYWORDS):
            lenses.append("Security")

        base = 0.6
        lens_boost = 0.1 * len(lenses)
        security_bonus = 0.1 if "Security" in lenses else 0.0
        link_bonus = 0.05 if got_links else 0.0
        cluster_snr = min(1.0, base + lens_boost + security_bonus + link_bonus)

        return {
            "lenses": lenses,
            "cluster_snr": round(cluster_snr, 3),
            "link_count": len(got_links),
        }
