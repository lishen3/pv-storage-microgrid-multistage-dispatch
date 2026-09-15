from __future__ import annotations

def terminal_soc_equivalent_cost(soc_end_kwh: float, *, reference_soc_kwh: float = 6000.0, settlement_price_yuan_per_kwh: float, eta_c: float = 0.9, eta_d: float = 0.9) -> float:
    """Finite-horizon terminal-SOC normalization used only for fair strategy comparison.

    Q2/Q3 do not explicitly require Dec.31 SOC to return to 6000 kWh.
    If end SOC is below the reference, charge the equivalent replenishment cost;
    if it is above the reference, credit the usable discharge energy.
    This does not alter actual dispatch or result files.
    """
    s=float(soc_end_kwh); ref=float(reference_soc_kwh); p=float(settlement_price_yuan_per_kwh)
    if s < ref:
        return (ref-s)/float(eta_c)*p
    return -(s-ref)*float(eta_d)*p
