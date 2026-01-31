import os
import re
import json
import yaml
import time
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

# --- Telegram Bot (admin + alert sender) ---
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

# --- Telethon (monitor channel sources) ---
from telethon import TelegramClient, events
from telethon.sessions import StringSession

# --- Optional exchange libs ---
EXCHANGE = os.getenv("EXCHANGE", "bitget").lower().strip()

# Bybit via pybit
if EXCHANGE == "bybit":
    from pybit.unified_trading import HTTP as BybitHTTP

# Bitget via ccxt
if EXCHANGE == "bitget":
    import ccxt.async_support as ccxt

# --- Optional OpenAI (AI signal parsing) ---
# Support BOTH old openai (0.28.x) and new openai (1.x)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
ENABLE_AI = os.getenv("ENABLE_AI", "1").strip().lower() in ("1", "true", "yes", "on")

openai_client = None
try:
    import openai  # either old or new
    # New SDK style (openai>=1.0.0)
    if hasattr(openai, "AsyncOpenAI"):
        openai_client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
    else:
        # Old SDK style (openai==0.28.1)
        openai.api_key = OPENAI_API_KEY
        openai_client = openai
except Exception:
    openai_client = None
    ENABLE_AI = False


# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("trading_bot")


# ---------------- Helpers ----------------
def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def parse_csv_list(v: str) -> List[str]:
    if not v:
        return []
    parts = [x.strip() for x in v.split(",")]
    return [p for p in parts if p]


def normalize_channel(s: str) -> str:
    s = s.strip()
    if not s:
        return s
    # allow @username or numeric id
    if s.startswith("@"):
        return s
    if s.lstrip("-").isdigit():
        return s
    return "@" + s


# ---------------- Data ----------------
@dataclass
class TradingSignal:
    direction: str                       # LONG/SHORT
    symbol: str                          # BTCUSDT
    order_type: str = "MARKET"           # MARKET/LIMIT
    entry_price: Optional[float] = None
    take_profit: List[float] = field(default_factory=list)
    stop_loss: Optional[float] = None
    leverage: int = 10
    confidence: float = 0.0
    source_channel: str = ""
    original_message: str = ""
    created_at: str = field(default_factory=now_str)

    def is_valid(self) -> bool:
        if self.direction not in ("LONG", "SHORT"):
            return False
        if not self.symbol or "USDT" not in self.symbol:
            return False
        if self.leverage < 1 or self.leverage > 125:
            return False
        return True


# ---------------- Config Manager ----------------
class ConfigManager:
    """
    Config disimpan di config.yaml (persist di Railway kalau kamu pakai volume / atau commit file)
    Kalau tidak ada volume, config akan reset saat redeploy. (Masih bisa pakai ENV juga)
    """
    def __init__(self, path: str = "config.yaml"):
        self.path = path
        self.config: Dict[str, Any] = self.load()

    def load(self) -> Dict[str, Any]:
        default = {
            "channels": {},  # key: "@channel" -> metadata
            "users": {},     # key: "user_id" -> settings
            "ai": {
                "confidence_threshold": float(os.getenv("AI_CONFIDENCE_THRESHOLD", "70")),
            },
            "risk": {
                "default_risk": float(os.getenv("DEFAULT_RISK", "1.0")),
                "max_risk": float(os.getenv("MAX_RISK", "5.0")),
                "default_leverage": int(os.getenv("DEFAULT_LEVERAGE", "10")),
                "max_leverage": int(os.getenv("MAX_LEVERAGE", "25")),
            },
        }

        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                # merge shallow
                for k, v in data.items():
                    default[k] = v
            except Exception as e:
                logger.warning(f"Failed reading {self.path}: {e}")
        return default

    def save(self) -> None:
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                yaml.safe_dump(self.config, f, sort_keys=False, allow_unicode=True)
        except Exception as e:
            logger.error(f"Failed saving config: {e}")

    # channels
    def add_channel(self, ch: str, title: str = "") -> None:
        ch = normalize_channel(ch)
        self.config["channels"][ch] = {
            "title": title,
            "enabled": True,
            "added_at": now_str(),
            "signal_count": int(self.config["channels"].get(ch, {}).get("signal_count", 0)),
        }
        self.save()

    def remove_channel(self, ch: str) -> bool:
        ch = normalize_channel(ch)
        if ch in self.config["channels"]:
            del self.config["channels"][ch]
            self.save()
            return True
        return False

    def list_channels(self) -> List[str]:
        return [k for k, v in self.config["channels"].items() if v.get("enabled", True)]

    # users
    def add_user(self, user_id: int, api_key: str, api_secret: str) -> None:
        user_id_s = str(user_id)
        self.config["users"][user_id_s] = {
            "api_key": api_key,
            "api_secret": api_secret,
            "auto_trade": False,
            "risk": float(self.config["risk"]["default_risk"]),
            "leverage": int(self.config["risk"]["default_leverage"]),
            "created_at": now_str(),
        }
        self.save()

    def remove_user(self, user_id: int) -> bool:
        user_id_s = str(user_id)
        if user_id_s in self.config["users"]:
            del self.config["users"][user_id_s]
            self.save()
            return True
        return False

    def get_user(self, user_id: int) -> Optional[Dict[str, Any]]:
        return self.config["users"].get(str(user_id))

    def set_user_value(self, user_id: int, key: str, val: Any) -> bool:
        u = self.get_user(user_id)
        if not u:
            return False
        u[key] = val
        self.save()
        return True

    def count_auto_users(self) -> int:
        c = 0
        for u in self.config["users"].values():
            if u.get("auto_trade"):
                c += 1
        return c


# ---------------- Signal Analyzer ----------------
class SignalAnalyzer:
    """
    1) parse cepat pakai regex (gratis)
    2) kalau ENABLE_AI=1 dan ada OpenAI key, AI dipakai untuk validasi + confidence
    """
    def __init__(self, confidence_threshold: float = 70.0):
        self.conf_threshold = confidence_threshold

    def _regex_extract(self, text: str) -> Optional[TradingSignal]:
        raw = text.strip()
        t = raw.upper()

        # direction
        dir_match = re.search(r"\b(LONG|SHORT|BUY|SELL)\b", t)
        if not dir_match:
            return None
        direction = dir_match.group(1)
        direction = "LONG" if direction in ("LONG", "BUY") else "SHORT"

        # symbol: BTCUSDT or BTC/USDT
        sym_match = re.search(r"\b([A-Z0-9]{2,15})\s*/?\s*USDT\b", t)
        if not sym_match:
            # sometimes "BTCUSDT"
            sym_match = re.search(r"\b([A-Z0-9]{2,15}USDT)\b", t)
        if not sym_match:
            return None
        symbol = sym_match.group(1).replace("/", "")
        if not symbol.endswith("USDT"):
            symbol += "USDT"

        # entry price (optional)
        entry_price = None
        m_entry = re.search(r"\b(ENTRY|ENTR|ENTRY PRICE)\b[:\s]*([0-9]+(?:\.[0-9]+)?)", t)
        if m_entry:
            entry_price = float(m_entry.group(2))

        # TP (can be multiple)
        tp = []
        # formats: "TP 43500 44000" OR "TP1: 43500" etc
        for m in re.finditer(r"\bTP\d*\b[:\s]*([0-9]+(?:\.[0-9]+)?)", t):
            tp.append(float(m.group(1)))
        if not tp:
            # "TP 43500 44000"
            m_tp_block = re.search(r"\bTP\b[:\s]*([0-9\.\s,]+)", t)
            if m_tp_block:
                nums = re.findall(r"([0-9]+(?:\.[0-9]+)?)", m_tp_block.group(1))
                tp = [float(x) for x in nums[:5]]

        # SL
        sl = None
        m_sl = re.search(r"\bSL\b[:\s]*([0-9]+(?:\.[0-9]+)?)", t)
        if m_sl:
            sl = float(m_sl.group(1))

        # leverage
        lev = None
        m_lev = re.search(r"\b(LEV|LEVERAGE)\b[:\s]*([0-9]{1,3})", t)
        if m_lev:
            lev = int(m_lev.group(2))

        sig = TradingSignal(
            direction=direction,
            symbol=symbol,
            order_type="LIMIT" if entry_price else "MARKET",
            entry_price=entry_price,
            take_profit=tp,
            stop_loss=sl,
            leverage=lev or int(os.getenv("DEFAULT_LEVERAGE", "10")),
            confidence=60.0,  # default from regex-only
            original_message=raw,
        )
        return sig if sig.is_valid() else None

    async def _ai_validate(self, text: str, channel_title: str = "") -> Optional[Dict[str, Any]]:
        if not ENABLE_AI or not OPENAI_API_KEY or not openai_client:
            return None

        prompt = f"""
Analyze this message and decide if it is a clear trading signal.
Return STRICT JSON only.

Channel: {channel_title}
Message:
{text}

JSON schema:
{{
  "is_signal": true/false,
  "confidence": 0-100,
  "direction": "LONG|SHORT",
  "symbol": "BTCUSDT",
  "order_type": "MARKET|LIMIT",
  "entry_price": number|null,
  "take_profit": [number],
  "stop_loss": number|null,
  "leverage": number|null,
  "notes": "short reason"
}}
"""
        try:
            # new sdk
            if hasattr(openai_client, "chat"):
                resp = await openai_client.chat.completions.create(
                    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    messages=[
                        {"role": "system", "content": "You are a strict trading signal extractor."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    max_tokens=450,
                )
                content = resp.choices[0].message.content.strip()
            else:
                # old sdk
                resp = await openai_client.ChatCompletion.acreate(
                    model=os.getenv("OPENAI_MODEL", "gpt-4"),
                    messages=[
                        {"role": "system", "content": "You are a strict trading signal extractor."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    max_tokens=450,
                )
                content = resp.choices[0].message.content.strip()

            # extract json
            j0 = content.find("{")
            j1 = content.rfind("}")
            if j0 == -1 or j1 == -1:
                return None
            data = json.loads(content[j0:j1 + 1])
            return data
        except Exception as e:
            logger.warning(f"AI validate failed: {e}")
            return None

    async def analyze(self, text: str, channel_title: str = "") -> Optional[TradingSignal]:
        # 1) regex extract first
        sig = self._regex_extract(text)
        if not sig:
            return None

        # 2) AI validate (optional)
        ai = await self._ai_validate(text, channel_title)
        if not ai:
            # if no AI, accept regex-only BUT mark confidence 60
            return sig

        if not ai.get("is_signal", False):
            return None

        confidence = float(ai.get("confidence", 0))
        if confidence < self.conf_threshold:
            return None

        # Use AI fields when available, fallback to regex
        sig.direction = (ai.get("direction") or sig.direction).upper()
        sig.symbol = (ai.get("symbol") or sig.symbol).upper().replace("/", "")
        sig.order_type = (ai.get("order_type") or sig.order_type).upper()
        sig.entry_price = ai.get("entry_price") if ai.get("entry_price") is not None else sig.entry_price
        sig.take_profit = ai.get("take_profit") or sig.take_profit
        sig.stop_loss = ai.get("stop_loss") if ai.get("stop_loss") is not None else sig.stop_loss
        sig.leverage = int(ai.get("leverage") or sig.leverage)
        sig.confidence = confidence

        return sig if sig.is_valid() else None


# ---------------- Exchange Executors ----------------
class BaseExecutor:
    async def close(self):
        pass

    async def get_usdt_balance(self) -> float:
        return 0.0

    async def place_trade(self, signal: TradingSignal, user_cfg: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class BitgetExecutor(BaseExecutor):
    def __init__(self, api_key: str, api_secret: str, password: str = ""):
        self.api_key = api_key
        self.api_secret = api_secret
        self.password = password
        self.ex = ccxt.bitget({
            "apiKey": api_key,
            "secret": api_secret,
            "password": password or os.getenv("BITGET_PASSPHRASE", ""),  # optional
            "enableRateLimit": True,
            "options": {"defaultType": os.getenv("BITGET_DEFAULT_TYPE", "swap")},  # swap = futures
        })

    async def close(self):
        try:
            await self.ex.close()
        except Exception:
            pass

    async def get_usdt_balance(self) -> float:
        try:
            bal = await self.ex.fetch_balance()
            # depends on account type; this is "best-effort"
            usdt = bal.get("USDT", {})
            free = usdt.get("free", 0) or 0
            total = usdt.get("total", 0) or 0
            return float(free or total or 0)
        except Exception as e:
            logger.warning(f"Bitget balance error: {e}")
            return 0.0

    async def _set_leverage(self, symbol: str, leverage: int):
        try:
            if hasattr(self.ex, "set_leverage"):
                await self.ex.set_leverage(leverage, symbol)
        except Exception as e:
            logger.warning(f"Bitget set_leverage failed: {e}")

    async def place_trade(self, signal: TradingSignal, user_cfg: Dict[str, Any]) -> Dict[str, Any]:
        safe_mode = os.getenv("SAFE_MODE", "ON").strip().lower() in ("1", "true", "yes", "on")
        if safe_mode:
            return {"success": True, "mode": "paper", "order_id": "PAPER", "note": "SAFE_MODE=ON"}

        side = "buy" if signal.direction == "LONG" else "sell"
        lev = int(user_cfg.get("leverage", signal.leverage))
        await self._set_leverage(signal.symbol, lev)

        # qty calculation: very basic (risk% of usdt balance, then *leverage / price)
        risk_pct = float(user_cfg.get("risk", 1.0))
        bal = await self.get_usdt_balance()
        if bal <= 0:
            return {"success": False, "error": "No USDT balance"}

        price = None
        try:
            ticker = await self.ex.fetch_ticker(signal.symbol)
            price = float(ticker.get("last") or ticker.get("close") or 0)
        except Exception as e:
            return {"success": False, "error": f"fetch_ticker failed: {e}"}

        if not price or price <= 0:
            return {"success": False, "error": "Invalid price"}

        risk_usdt = bal * (risk_pct / 100.0)
        position_value = risk_usdt * lev
        qty = position_value / price
        qty = float(f"{qty:.4f}")  # rounding

        try:
            # Market order
            order = await self.ex.create_order(symbol=signal.symbol, type="market", side=side, amount=qty)
            return {"success": True, "order_id": order.get("id"), "qty": qty, "price": price}
        except Exception as e:
            return {"success": False, "error": str(e)}


class BybitExecutor(BaseExecutor):
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.session = BybitHTTP(api_key=api_key, api_secret=api_secret, testnet=testnet)

    async def get_usdt_balance(self) -> float:
        try:
            result = self.session.get_wallet_balance(accountType="UNIFIED")
            if result.get("retCode") != 0:
                return 0.0
            coins = result["result"]["list"][0]["coin"]
            for c in coins:
                if c["coin"] == "USDT":
                    return float(c.get("walletBalance") or 0)
            return 0.0
        except Exception:
            return 0.0

    async def place_trade(self, signal: TradingSignal, user_cfg: Dict[str, Any]) -> Dict[str, Any]:
        safe_mode = os.getenv("SAFE_MODE", "ON").strip().lower() in ("1", "true", "yes", "on")
        if safe_mode:
            return {"success": True, "mode": "paper", "order_id": "PAPER", "note": "SAFE_MODE=ON"}

        side = "Buy" if signal.direction == "LONG" else "Sell"
        lev = int(user_cfg.get("leverage", signal.leverage))

        # set leverage
        try:
            self.session.set_leverage(
                category="linear",
                symbol=signal.symbol,
                buyLeverage=str(lev),
                sellLeverage=str(lev),
            )
        except Exception as e:
            logger.warning(f"Bybit set_leverage failed: {e}")

        # qty (basic)
        risk_pct = float(user_cfg.get("risk", 1.0))
        bal = await self.get_usdt_balance()
        if bal <= 0:
            return {"success": False, "error": "No USDT balance"}

        try:
            ticker = self.session.get_tickers(category="linear", symbol=signal.symbol)
            if ticker.get("retCode") != 0:
                return {"success": False, "error": ticker.get("retMsg", "ticker error")}
            price = float(ticker["result"]["list"][0]["lastPrice"])
        except Exception as e:
            return {"success": False, "error": str(e)}

        risk_usdt = bal * (risk_pct / 100.0)
        position_value = risk_usdt * lev
        qty = position_value / price
        qty = float(f"{qty:.4f}")

        try:
            params = {
                "category": "linear",
                "symbol": signal.symbol,
                "side": side,
                "orderType": "Market",
                "qty": str(qty),
                "timeInForce": "GTC",
            }
            res = self.session.place_order(**params)
            if res.get("retCode") == 0:
                return {"success": True, "order_id": res["result"]["orderId"], "qty": qty, "price": price}
            return {"success": False, "error": res.get("retMsg", "order failed")}
        except Exception as e:
            return {"success": False, "error": str(e)}


# ---------------- Telethon Channel Monitor ----------------
class ChannelMonitor:
    def __init__(self, api_id: int, api_hash: str, session_str: str):
        self.api_id = api_id
        self.api_hash = api_hash
        self.session_str = session_str
        self.client: Optional[TelegramClient] = None
        self.on_signal = None

    def set_callback(self, cb):
        self.on_signal = cb

    async def start(self, channels: List[str]):
        if not self.session_str:
            raise RuntimeError("TELETHON_SESSION is empty. Generate StringSession locally first.")

        self.client = TelegramClient(StringSession(self.session_str), self.api_id, self.api_hash)
        await self.client.connect()

        if not await self.client.is_user_authorized():
            raise RuntimeError("Telethon session is NOT authorized. Re-generate TELETHON_SESSION.")

        # resolve entities once
        resolved = []
        for ch in channels:
            ch = normalize_channel(ch)
            try:
                ent = await self.client.get_entity(ch)
                resolved.append(ent)
                logger.info(f"Telethon monitoring added: {ch}")
            except Exception as e:
                logger.warning(f"Cannot access channel {ch}: {e}")

        @self.client.on(events.NewMessage(chats=resolved))
        async def handler(event):
            try:
                msg = event.message.message or ""
                if not msg.strip():
                    return
                chat = await event.get_chat()
                title = getattr(chat, "title", "") or getattr(chat, "username", "") or "unknown"

                # callback to bot
                if self.on_signal:
                    await self.on_signal(msg, title)
            except Exception as e:
                logger.error(f"Telethon handler error: {e}")

        logger.info("Telethon monitoring started.")
        await self.client.run_until_disconnected()


# ---------------- Main Bot ----------------
class AutoTradingBot:
    def __init__(self):
        # env required
        self.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()

        self.telegram_api_id = int(os.getenv("TELEGRAM_API_ID", "0"))
        self.telegram_api_hash = os.getenv("TELEGRAM_API_HASH", "").strip()
        self.telethon_session = os.getenv("TELETHON_SESSION", "").strip()

        # admin access
        self.admin_ids = set()
        for x in parse_csv_list(os.getenv("ADMIN_USER_IDS", "")):
            if x.isdigit():
                self.admin_ids.add(int(x))

        # alerts
        self.alert_channels = [normalize_channel(x) for x in parse_csv_list(os.getenv("ALERT_CHANNELS", ""))]

        # runtime
        self.cfg = ConfigManager()
        self.analyzer = SignalAnalyzer(confidence_threshold=float(self.cfg.config["ai"]["confidence_threshold"]))
        self.app = Application.builder().token(self.telegram_token).build()

        # active executors per user_id (cache)
        self.executors: Dict[str, BaseExecutor] = {}

    def _is_admin(self, user_id: int) -> bool:
        # if ADMIN_USER_IDS not set, allow everyone (not recommended)
        if not self.admin_ids:
            return True
        return user_id in self.admin_ids

    async def _get_executor(self, user_id_s: str, user_cfg: Dict[str, Any]) -> BaseExecutor:
        if user_id_s in self.executors:
            return self.executors[user_id_s]

        api_key = user_cfg["api_key"]
        api_secret = user_cfg["api_secret"]

        if EXCHANGE == "bybit":
            testnet = os.getenv("BYBIT_TESTNET", "0").strip().lower() in ("1", "true", "yes", "on")
            ex = BybitExecutor(api_key, api_secret, testnet=testnet)
        else:
            # Bitget may require passphrase
            passphrase = os.getenv("BITGET_PASSPHRASE", "").strip()
            ex = BitgetExecutor(api_key, api_secret, password=passphrase)

        self.executors[user_id_s] = ex
        return ex

    async def send_alert(self, text: str):
        if not self.alert_channels:
            return
        for ch in self.alert_channels:
            try:
                await self.app.bot.send_message(chat_id=ch, text=text, parse_mode="Markdown")
            except Exception as e:
                logger.warning(f"Failed to send alert to {ch}: {e}")

    async def on_new_message_from_channel(self, msg: str, channel_title: str):
        # analyze
        sig = await self.analyzer.analyze(msg, channel_title=channel_title)
        if not sig:
            return

        sig.source_channel = channel_title
        logger.info(f"Signal detected: {sig.direction} {sig.symbol} conf={sig.confidence}")

        # alert format
        alert = (
            "🚨 **SIGNAL DETECTED**\n\n"
            f"Source: **{sig.source_channel}**\n"
            f"Direction: **{sig.direction}**\n"
            f"Symbol: **{sig.symbol}**\n"
            f"Entry: **{sig.entry_price if sig.entry_price else 'MARKET'}**\n"
            f"TP: **{', '.join(map(str, sig.take_profit)) if sig.take_profit else 'N/A'}**\n"
            f"SL: **{sig.stop_loss if sig.stop_loss else 'N/A'}**\n"
            f"Lev: **{sig.leverage}**\n"
            f"Confidence: **{sig.confidence}%**\n"
            f"Time: `{now_str()}`"
        )
        await self.send_alert(alert)

        # execute for users with auto_trade=True
        executed = 0
        for user_id_s, user_cfg in self.cfg.config["users"].items():
            if not user_cfg.get("auto_trade", False):
                continue

            ex = await self._get_executor(user_id_s, user_cfg)
            res = await ex.place_trade(sig, user_cfg)

            if res.get("success"):
                executed += 1
                await self._dm_user(int(user_id_s), sig, res)
            else:
                await self._dm_user(int(user_id_s), sig, res)

        logger.info(f"Signal processed. executed={executed}")

    async def _dm_user(self, user_id: int, sig: TradingSignal, res: Dict[str, Any]):
        try:
            ok = res.get("success")
            status = "✅ SUCCESS" if ok else "❌ FAILED"
            mode = res.get("mode", "live")
            msg = (
                f"🤖 **AUTO-TRADE RESULT**\n\n"
                f"Status: **{status}**\n"
                f"Mode: **{mode}**\n"
                f"Symbol: **{sig.symbol}**\n"
                f"Direction: **{sig.direction}**\n"
                f"Order ID: `{res.get('order_id', 'N/A')}`\n"
                f"Info: `{res.get('note') or res.get('error') or ''}`\n"
                f"Time: `{now_str()}`"
            )
            await self.app.bot.send_message(chat_id=user_id, text=msg, parse_mode="Markdown")
        except Exception as e:
            logger.warning(f"DM user failed: {e}")

    # ---------- Commands ----------
    def register_handlers(self):
        self.app.add_handler(CommandHandler("start", self.cmd_start))
        self.app.add_handler(CommandHandler("status", self.cmd_status))
        self.app.add_handler(CommandHandler("add_channel", self.cmd_add_channel))
        self.app.add_handler(CommandHandler("remove_channel", self.cmd_remove_channel))
        self.app.add_handler(CommandHandler("add_user", self.cmd_add_user))
        self.app.add_handler(CommandHandler("remove_user", self.cmd_remove_user))
        self.app.add_handler(CommandHandler("toggle_auto_trading", self.cmd_toggle_auto))
        self.app.add_handler(CommandHandler("set_risk", self.cmd_set_risk))
        self.app.add_handler(CommandHandler("set_leverage", self.cmd_set_leverage))

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = update.effective_user.id
        if not self._is_admin(uid):
            await update.message.reply_text("❌ Not allowed.")
            return

        safe_mode = os.getenv("SAFE_MODE", "ON").strip()
        await update.message.reply_text(
            "🤖 Auto-Trading Bot\n"
            f"Exchange: {EXCHANGE.upper()}\n"
            f"SAFE_MODE: {safe_mode}\n\n"
            "Commands:\n"
            "/status\n"
            "/add_channel @channel\n"
            "/remove_channel @channel\n"
            "/add_user USER_ID API_KEY API_SECRET\n"
            "/remove_user USER_ID\n"
            "/toggle_auto_trading\n"
            "/set_risk PERCENT(0-5)\n"
            "/set_leverage NUMBER\n"
        )

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = update.effective_user.id
        if not self._is_admin(uid):
            await update.message.reply_text("❌ Not allowed.")
            return

        channels = self.cfg.list_channels()
        users_total = len(self.cfg.config["users"])
        auto_users = self.cfg.count_auto_users()

        my_cfg = self.cfg.get_user(uid)
        my_auto = my_cfg.get("auto_trade") if my_cfg else False
        my_risk = my_cfg.get("risk") if my_cfg else "N/A"
        my_lev = my_cfg.get("leverage") if my_cfg else "N/A"

        # best-effort balance (only if you are registered)
        bal = "N/A"
        try:
            if my_cfg:
                ex = await self._get_executor(str(uid), my_cfg)
                b = await ex.get_usdt_balance()
                bal = f"{b:.4f}"
        except Exception:
            pass

        await update.message.reply_text(
            "📊 Status\n\n"
            f"Monitored channels: {len(channels)}\n"
            f"Registered users: {users_total}\n"
            f"Auto-users: {auto_users}\n"
            f"USDT balance ({EXCHANGE}): {bal}\n"
            f"Your ID: {uid}\n"
            f"Auto-trade: {my_auto}\n"
            f"Risk: {my_risk}\n"
            f"Leverage: {my_lev}\n"
        )

    async def cmd_add_channel(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = update.effective_user.id
        if not self._is_admin(uid):
            await update.message.reply_text("❌ Not allowed.")
            return
        if not context.args:
            await update.message.reply_text("Usage: /add_channel @channel_username")
            return
        ch = normalize_channel(context.args[0])
        self.cfg.add_channel(ch)
        await update.message.reply_text(f"✅ Added channel {ch}")

    async def cmd_remove_channel(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = update.effective_user.id
        if not self._is_admin(uid):
            await update.message.reply_text("❌ Not allowed.")
            return
        if not context.args:
            await update.message.reply_text("Usage: /remove_channel @channel_username")
            return
        ch = normalize_channel(context.args[0])
        ok = self.cfg.remove_channel(ch)
        await update.message.reply_text("✅ Removed" if ok else "❌ Not found")

    async def cmd_add_user(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = update.effective_user.id
        if not self._is_admin(uid):
            await update.message.reply_text("❌ Not allowed.")
            return

        if len(context.args) != 3:
            await update.message.reply_text("Usage: /add_user USER_ID API_KEY API_SECRET")
            return
        user_id = int(context.args[0])
        api_key = context.args[1].strip()
        api_secret = context.args[2].strip()

        self.cfg.add_user(user_id, api_key, api_secret)
        await update.message.reply_text(f"✅ User {user_id} added (auto_trade default OFF)")

    async def cmd_remove_user(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = update.effective_user.id
        if not self._is_admin(uid):
            await update.message.reply_text("❌ Not allowed.")
            return

        if len(context.args) != 1:
            await update.message.reply_text("Usage: /remove_user USER_ID")
            return
        user_id = int(context.args[0])
        ok = self.cfg.remove_user(user_id)
        await update.message.reply_text("✅ Removed" if ok else "❌ Not found")

    async def cmd_toggle_auto(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = update.effective_user.id
        if not self._is_admin(uid):
            await update.message.reply_text("❌ Not allowed.")
            return

        u = self.cfg.get_user(uid)
        if not u:
            await update.message.reply_text("❌ You are not registered. Use /add_user YOUR_ID KEY SECRET")
            return

        new_status = not bool(u.get("auto_trade", False))
        self.cfg.set_user_value(uid, "auto_trade", new_status)
        await update.message.reply_text(f"✅ Auto-trading {'ENABLED' if new_status else 'DISABLED'}")

    async def cmd_set_risk(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = update.effective_user.id
        if not self._is_admin(uid):
            await update.message.reply_text("❌ Not allowed.")
            return

        if len(context.args) != 1:
            await update.message.reply_text("Usage: /set_risk 1.0 (0-5)")
            return
        try:
            v = float(context.args[0])
        except ValueError:
            await update.message.reply_text("❌ Risk must be a number.")
            return

        max_risk = float(self.cfg.config["risk"]["max_risk"])
        if v < 0 or v > max_risk:
            await update.message.reply_text(f"❌ Risk must be 0-{max_risk}")
            return

        ok = self.cfg.set_user_value(uid, "risk", v)
        await update.message.reply_text("✅ Risk updated" if ok else "❌ Not registered")

    async def cmd_set_leverage(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = update.effective_user.id
        if not self._is_admin(uid):
            await update.message.reply_text("❌ Not allowed.")
            return

        if len(context.args) != 1:
            await update.message.reply_text("Usage: /set_leverage 10")
            return
        try:
            v = int(context.args[0])
        except ValueError:
            await update.message.reply_text("❌ Leverage must be an integer.")
            return

        max_lev = int(self.cfg.config["risk"]["max_leverage"])
        if v < 1 or v > max_lev:
            await update.message.reply_text(f"❌ Leverage must be 1-{max_lev}")
            return

        ok = self.cfg.set_user_value(uid, "leverage", v)
        await update.message.reply_text("✅ Leverage updated" if ok else "❌ Not registered")

    async def run(self):
        # sanity checks
        if not self.telegram_token:
            raise RuntimeError("Missing TELEGRAM_BOT_TOKEN")
        if self.telegram_api_id <= 0 or not self.telegram_api_hash:
            raise RuntimeError("Missing TELEGRAM_API_ID / TELEGRAM_API_HASH")
        if not self.telethon_session:
            raise RuntimeError("Missing TELETHON_SESSION (StringSession). Required for monitoring without OTP.")

        # load channels from ENV (optional bootstrap)
        env_channels = [normalize_channel(x) for x in parse_csv_list(os.getenv("MONITOR_CHANNELS", ""))]
        for ch in env_channels:
            self.cfg.add_channel(ch)

        self.register_handlers()

        # start both: telegram polling + telethon monitoring
        monitor = ChannelMonitor(self.telegram_api_id, self.telegram_api_hash, self.telethon_session)
        monitor.set_callback(self.on_new_message_from_channel)

        channels_to_watch = self.cfg.list_channels()
        logger.info(f"Monitoring channels: {channels_to_watch}")
        logger.info(f"Alert channels: {self.alert_channels}")
        logger.info(f"Exchange: {EXCHANGE}, SAFE_MODE={os.getenv('SAFE_MODE','ON')}")

        async with self.app:
            await self.app.initialize()
            await self.app.start()
            await self.app.updater.start_polling(drop_pending_updates=True)

            # start telethon monitoring concurrently
            monitor_task = asyncio.create_task(monitor.start(channels_to_watch))

            # keep alive
            try:
                await monitor_task
            finally:
                await self.app.updater.stop()
                await self.app.stop()
                await self.app.shutdown()

                # close executors
                for ex in self.executors.values():
                    try:
                        await ex.close()
                    except Exception:
                        pass


# ---------------- Entry ----------------
async def main():
    bot = AutoTradingBot()
    await bot.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Stopped.")
