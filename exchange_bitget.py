# exchange_bitget.py
import os
import ccxt

class BitgetExecutor:
    def __init__(self, api_key: str, api_secret: str, api_passphrase: str):
        default_type = os.getenv("BITGET_DEFAULT_TYPE", "swap")  # swap|spot
        self.ex = ccxt.bitget({
            "apiKey": api_key,
            "secret": api_secret,
            "password": api_passphrase,  # Bitget passphrase via CCXT
            "options": {"defaultType": default_type},
            "enableRateLimit": True,
        })

        # Sandbox (kalau kamu benar-benar punya demo env yang cocok)
        sandbox = os.getenv("BITGET_SANDBOX", "false").lower() == "true"
        if sandbox:
            self.ex.set_sandbox_mode(True)

    def execute_signal(self, symbol: str, direction: str, amount: float, leverage: int | None = None,
                       tp: float | None = None, sl: float | None = None):
        """
        symbol: contoh 'BTC/USDT' (CCXT unified)
        direction: 'LONG' / 'SHORT'
        amount: ukuran kontrak/qty (sesuaikan dengan market type)
        leverage: optional
        tp/sl: optional (kalau bot kamu pasang TP/SL setelah entry, itu perlu logic tambahan)
        """
        side = "buy" if direction.upper() == "LONG" else "sell"

        params = {}
        # Untuk swap (futures) Bitget, sering perlu tradeSide=open untuk hedge-mode.
        # Kita set default aman:
        if os.getenv("BITGET_DEFAULT_TYPE", "swap") == "swap":
            params["tradeSide"] = "open"
            params["marginMode"] = os.getenv("BITGET_MARGIN_MODE", "isolated")

            if leverage:
                # CCXT punya setLeverage di banyak exchange; Bitget support untuk swap.
                # Kalau error di environment kamu, tinggal comment baris ini.
                self.ex.set_leverage(int(leverage), symbol)

        order = self.ex.create_order(symbol, "market", side, amount, None, params)
        return order
