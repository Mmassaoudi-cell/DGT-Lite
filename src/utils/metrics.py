"""Primary/secondary metric convention for this study (adapting the master protocol's
"macro-F1 as primary metric" to a scheduling task, since there is no classification label
here). Deadline compliance, load-balance fairness (Jain's index) and energy are the three
metrics the source paper itself foregrounds (abstract's two headline claims + its own
dedicated energy section). We additionally define one composite Scheduling Quality Score
(SQS) used ONLY for validation-only candidate screening / model selection ranking; every
final reported comparison always shows the three components individually as well."""
from __future__ import annotations

W_COMPLIANCE = 0.50
W_FAIRNESS = 0.35
W_ENERGY = 0.15


def sqs(deadline_compliance: float, load_balance_fairness: float, energy: float,
        energy_ref: float) -> float:
    energy_eff = 1.0 - min(1.0, max(0.0, energy / max(1e-9, energy_ref)))
    return (W_COMPLIANCE * deadline_compliance +
            W_FAIRNESS * load_balance_fairness +
            W_ENERGY * energy_eff)
