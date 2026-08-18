from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ExplanationResult:
    technique_used: str
    prediction: Any

    confidence_score: Optional[float] = None

    feature_ranking: list[dict[str, Any]] = field(default_factory=list)

    raw_output: Any = None

    metadata: dict[str, Any] = field(default_factory=dict)

    success: bool = True
    error: Optional[str] = None