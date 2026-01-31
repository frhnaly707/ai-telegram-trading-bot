import os
import ccxt

def normalize_symbol_to_ccxt(sym: str) -> str:
    s = (sym or "").upper().replace("-", "").replace("_", "").replace(" ", "")
    if s.endswith("USDT") and "/" not in s:
        return f"{s[:-4]}/USDT"
    return sym

class BitgetExecutor:
    def __init__(self):
        k = os.getenv("BITGET_API_KEY", "")
        s = os.getenv("BITGET_API_SECRET", "")
        p = os.getenv("BITGET_API_PASSPHRASE", "")
        if not (k and s and p):
            raise RuntimeError("Missing BITGET_API_KEY / BITGET_API_SECRET / BITGET_API_PASSPHRASE")

        default_type = os.getenv("BITGET_DEFAULT_TYPE", "swap")  # swap|spot
        self.ex = ccxt.bitget({
            "apiKey": k,
            "secret": s,
            "password": p,  # passphrase
            "enableRateLimit": True,
            "options": {"defaultType": default_type},
        })

        if os.getenv("BITGET_SANDBOX", "false").lower() == "true":
            self.ex.set_sandbox_mode(True)

    def market_order(self, symbol: str, direction: str, amount: float, leverage: int | None = None):
        ccxt_symbol = normalize_symbol_to_ccxt(symbol)
        side = "buy" if (direction or "").upper() in ("LONG", "BUY") else "sell"

        params = {}
        if os.getenv("BITGET_DEFAULT_TYPE", "swap") == "swap":
            params["marginMode"] = os.getenv("BITGET_MARGIN_MODE", "isolated")
            if leverage:
                self.ex.set_leverage(int(leverage), ccxt_symbol)

        return self.ex.create_order(ccxt_symbol, "market", side, float(amount), None, params)
