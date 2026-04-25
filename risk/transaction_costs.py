"""
risk/transaction_costs.py
Full Zerodha intraday cost breakdown:
  brokerage + STT + exchange + SEBI + GST + stamp
Accurate to NSE equity intraday rules (as of 2024).
"""


class TransactionCosts:
    """
    Computes full round-trip cost for NSE intraday equity.
    All figures per side unless stated.
    """

    # Zerodha intraday flat fee (capped at ₹20 per order)
    BROKERAGE_FLAT  = 20.0       # ₹ per order, intraday
    BROKERAGE_PCT   = 0.0003     # 0.03% — whichever is lower applies

    # NSE charges
    STT_SELL        = 0.00025    # 0.025% on sell side only (intraday)
    EXCHANGE_TXN    = 0.0000335  # 0.00335% both sides
    SEBI_TURNOVER   = 0.000001   # ₹10 per crore
    GST_RATE        = 0.18       # 18% on (brokerage + exchange charges)
    STAMP_BUY       = 0.00003    # 0.003% on buy side only

    def total(self, price: float, shares: int, direction: str = "BUY") -> float:
        """
        Returns total transaction cost (₹) for ONE SIDE of trade.
        Call twice (entry + exit) for round-trip.
        """
        turnover = price * shares

        brokerage = min(self.BROKERAGE_FLAT,
                        turnover * self.BROKERAGE_PCT)

        stt   = turnover * self.STT_SELL    if direction == "SELL" else 0.0
        stamp = turnover * self.STAMP_BUY   if direction == "BUY"  else 0.0
        exch  = turnover * self.EXCHANGE_TXN
        sebi  = turnover * self.SEBI_TURNOVER
        gst   = (brokerage + exch) * self.GST_RATE

        return brokerage + stt + stamp + exch + sebi + gst

    def round_trip(self, entry: float, exit_: float, shares: int) -> float:
        """Total cost for entry (BUY) + exit (SELL)."""
        return self.total(entry, shares, "BUY") + self.total(exit_, shares, "SELL")

    def bps(self, entry: float, exit_: float, shares: int) -> float:
        """Round-trip cost in basis points of entry notional."""
        notional = entry * shares
        if notional == 0:
            return 0.0
        return self.round_trip(entry, exit_, shares) / notional * 10_000
