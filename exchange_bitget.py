import os
import ccxt


class BitgetExecutor:
    """
    Bitget executor via CCXT.
    - Supports spot/swap depending on BITGET_DEFAULT_TYPE env.
    - Tries to resolve symbols like BTCUSDT, BTC/USDT, BTC/USDT:USDT automatically.
    """

    def __init__(self):
        api_key = os.getenv("BITGET_API_KEY", "")
        api_secret = os.getenv("BITGET_API_SECRET", "")
        api_passphrase = os.getenv("BITGET_API_PASSPHRASE", "")

        if not (api_key and api_secret and api_passphrase):
            raise RuntimeError("Missing BITGET_API_KEY / BITGET_API_SECRET / BITGET_API_PASSPHRASE")

        default_type = os.getenv("BITGET_DEFAULT_TYPE", "swap").lower()  # swap|spot
        self.default_type = default_type

        self.ex = ccxt.bitget({
            "apiKey": api_key,
            "secret": api_secret,
            "password": api_passphrase,  # CCXT uses 'password' for passphrase
            "enableRateLimit": True,
            "options": {"defaultType": default_type},
        })

        sandbox = os.getenv("BITGET_SANDBOX", "false").lower() == "true"
        if sandbox:
            self.ex.set_sandbox_mode(True)

        # Load markets once
        self.markets = self.ex.load_markets()

    @staticmethod
    def _normalize_symbol_input(sym: str) -> str:
        s = (sym or "").strip().upper().replace(" ", "")
        s = s.replace("-", "/").replace("_", "/")
        # BTCUSDT -> BTC/USDT
        if "/" not in s and s.endswith("USDT"):
            s = f"{s[:-4]}/USDT"
        return s

    def resolve_symbol(self, symbol: str) -> str:
        """
        Try to find the best matching CCXT symbol for Bitget.
        """
        s = self._normalize_symbol_input(symbol)
        base = None
        quote = None

        if "/" in s:
            parts = s.split("/")
            if len(parts) >= 2:
                base, quote = parts[0], parts[1]
        else:
            # fallback
            return symbol

        candidates = []

        # For swap, Bitget often uses BTC/USDT:USDT style in CCXT
        if quote == "USDT":
            candidates.append(f"{base}/USDT:USDT")
        candidates.append(f"{base}/{quote}")
        # Some signals might come as BTC/USDTUSDT etc
        candidates.append(s)

        # Pick first that exists in markets
        for c in candidates:
            if c in self.markets:
                # If default_type is swap, prefer swap market if available
                if self.default_type == "swap":
                    if self.markets[c].get("swap") is True:
                        return c
                return c

        # If not found, try brute search by base/quote
        for mk, info in self.markets.items():
            if info.get("base") == base and info.get("quote") == quote:
                if self.default_type == "swap" and info.get("swap") is True:
                    return mk
                if self.default_type == "spot" and info.get("spot") is True:
                    return mk

        # Give up, return normalized
        return s

    def fetch_usdt_balance(self) -> float:
        bal = self.ex.fetch_balance()
        # Most CCXT balances put free USDT in bal['free']['USDT']
        free = bal.get("free", {}).get("USDT")
        if free is None:
            # fallback
            total = bal.get("total", {}).get("USDT")
            return float(total or 0)
        return float(free or 0)

    def fetch_last_price(self, symbol: str) -> float | None:
        ccxt_symbol = self.resolve_symbol(symbol)
        t = self.ex.fetch_ticker(ccxt_symbol)
        last = t.get("last")
        return float(last) if last else None

    def set_leverage(self, symbol: str, leverage: int) -> None:
        ccxt_symbol = self.resolve_symbol(symbol)
        try:
            self.ex.set_leverage(int(leverage), ccxt_symbol)
        except Exception:
            # Not all accounts / markets allow setting leverage via CCXT; ignore safely.
            pass

    def market_order(self, symbol: str, direction: str, amount: float, leverage: int | None = None):
        ccxt_symbol = self.resolve_symbol(symbol)
        side = "buy" if (direction or "").upper() in ("LONG", "BUY") else "sell"

        if leverage:
            self.set_leverage(ccxt_symbol, int(leverage))

        params = {}
        if self.default_type == "swap":
            params["marginMode"] = os.getenv("BITGET_MARGIN_MODE", "isolated").lower()

        return self.ex.create_order(ccxt_symbol, "market", side, float(amount), None, params)
