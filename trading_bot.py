import asyncio
import logging
import os
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, List

from dotenv import load_dotenv
load_dotenv()

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

import yaml
from telethon import TelegramClient, events
from telethon.sessions import StringSession
from telethon.tl.types import Channel

from exchange_bitget import BitgetExecutor

# =========================
# LOGGING
# =========================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=getattr(logging, LOG_LEVEL, logging.INFO),
)
logger = logging.getLogger("trading_bot")

# =========================
# SAFETY FLAGS
# =========================
# Default SAFE MODE = true => BOT WILL NOT EXECUTE REAL TRADES
SAFE_MODE = os.getenv("SAFE_MODE", "true").lower() == "true"

# If you want real execution:
# SAFE_MODE=false
# Also ensure your Bitget API key has trading permissions.
# =========================================


@dataclass
class TradingSignal:
    direction: str
    symbol: str
    order_type: str
    entry_price: Optional[float] = None
    take_profit: List[float] = None
    stop_loss: float = 0.0
    leverage: int = 10
    risk_percentage: float = 3.0
    confidence_score: float = 0.0
    original_message: str = ""
    timestamp: datetime = None

    def __post_init__(self):
        if self.take_profit is None:
            self.take_profit = []
        if self.timestamp is None:
            self.timestamp = datetime.now()


class AISignalAnalyzer:
    """
    AI extractor for signals.
    Works with OpenAI SDK v1.x (recommended).
    """

    def __init__(self, openai_api_key: str):
        self.api_key = openai_api_key
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.conf_threshold = float(os.getenv("AI_CONFIDENCE_THRESHOLD", "80"))

        # Lazy init client
        self._client = None

    def _client_init(self):
        if self._client is not None:
            return

        # Try new SDK
        try:
            from openai import AsyncOpenAI  # type: ignore
            self._client = AsyncOpenAI(api_key=self.api_key)
            return
        except Exception:
            pass

        # Fallback old style (best effort)
        import openai  # type: ignore
        openai.api_key = self.api_key
        self._client = openai

    def _contains_keywords(self, text: str) -> bool:
        t = (text or "").upper()
        keywords = ["LONG", "SHORT", "BUY", "SELL", "USDT", "TP", "SL", "ENTRY", "LEVERAGE", "SIGNAL", "MARKET", "LIMIT"]
        return any(k in t for k in keywords)

    def _prompt(self, message: str, channel: str) -> str:
        return f"""
Analyze this message and decide if it contains a CLEAR trading signal.
Channel: {channel}
Message: "{message}"

Extract these fields if possible:
- direction: LONG/SHORT
- symbol: e.g. BTCUSDT
- order_type: MARKET/LIMIT
- entry_price: number or null
- take_profit: list of numbers
- stop_loss: number or null
- leverage: number (default 10)
- confidence: 0-100

Return STRICT JSON only:
{{
  "is_signal": true/false,
  "confidence": 0-100,
  "direction": "LONG/SHORT",
  "symbol": "BTCUSDT",
  "order_type": "MARKET/LIMIT",
  "entry_price": null,
  "take_profit": [0],
  "stop_loss": null,
  "leverage": 10,
  "reasoning": "short reason"
}}
"""

    @staticmethod
    def _extract_json(s: str) -> Optional[dict]:
        if not s:
            return None
        start = s.find("{")
        end = s.rfind("}") + 1
        if start == -1 or end <= 0:
            return None
        try:
            return json.loads(s[start:end])
        except Exception:
            return None

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        s = (symbol or "").upper().replace("/", "").strip()
        if not s:
            return ""
        if not s.endswith("USDT"):
            s = s + "USDT"
        return s

    async def analyze_message(self, message_text: str, channel_name: str = "") -> Optional[TradingSignal]:
        try:
            if not self._contains_keywords(message_text):
                return None

            self._client_init()
            prompt = self._prompt(message_text, channel_name)

            # New SDK path
            if hasattr(self._client, "chat"):
                resp = await self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You extract crypto trading signals. Output STRICT JSON only."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    max_tokens=400,
                )
                content = resp.choices[0].message.content or ""
            else:
                # Old SDK fallback
                resp = await self._client.ChatCompletion.acreate(  # type: ignore
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You extract crypto trading signals. Output STRICT JSON only."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    max_tokens=400,
                )
                content = resp.choices[0].message["content"]  # type: ignore

            data = self._extract_json(content)
            if not data or not data.get("is_signal"):
                return None

            conf = float(data.get("confidence") or 0)
            if conf < self.conf_threshold:
                return None

            direction = (data.get("direction") or "").upper()
            if direction not in ("LONG", "SHORT", "BUY", "SELL"):
                return None

            symbol = self._normalize_symbol(data.get("symbol") or "")
            if not symbol or "USDT" not in symbol:
                return None

            signal = TradingSignal(
                direction=direction,
                symbol=symbol,
                order_type=(data.get("order_type") or "MARKET").upper(),
                entry_price=data.get("entry_price"),
                take_profit=data.get("take_profit") or [],
                stop_loss=float(data.get("stop_loss") or 0.0),
                leverage=int(data.get("leverage") or 10),
                confidence_score=conf,
                original_message=message_text,
            )
            return signal

        except Exception as e:
            logger.error(f"AI analyze error: {e}")
            return None


class ChannelMonitor:
    """
    Telethon channel monitor.
    IMPORTANT: On Railway you MUST use TELEGRAM_SESSION_STRING (no stdin).
    """

    def __init__(self, api_id: int, api_hash: str, phone: str, analyzer: AISignalAnalyzer):
        session_str = os.getenv("TELEGRAM_SESSION_STRING", "").strip()
        if not session_str:
            raise RuntimeError(
                "Missing TELEGRAM_SESSION_STRING. "
                "Generate it locally once using Telethon, then paste to Railway Variables."
            )

        self.client = TelegramClient(StringSession(session_str), api_id, api_hash)
        self.phone = phone
        self.analyzer = analyzer
        self.monitored_channels: List[dict] = []
        self.signal_callback = None

    async def initialize(self) -> bool:
        try:
            await self.client.connect()
            if not await self.client.is_user_authorized():
                raise RuntimeError("Telethon session is not authorized. Re-generate TELEGRAM_SESSION_STRING.")
            logger.info("✅ Telethon connected & authorized")
            return True
        except Exception as e:
            logger.error(f"Telethon init failed: {e}")
            return False

    def set_signal_callback(self, cb):
        self.signal_callback = cb

    async def add_channel(self, channel_username: str) -> bool:
        try:
            if not channel_username.startswith("@"):
                channel_username = "@" + channel_username
            ent = await self.client.get_entity(channel_username)
            if not isinstance(ent, Channel):
                return False
            self.monitored_channels.append({"entity": ent, "username": channel_username, "title": ent.title})
            return True
        except Exception as e:
            logger.error(f"Add channel error {channel_username}: {e}")
            return False

    async def remove_channel(self, channel_username: str) -> bool:
        if not channel_username.startswith("@"):
            channel_username = "@" + channel_username
        for ch in self.monitored_channels[:]:
            if ch["username"] == channel_username:
                self.monitored_channels.remove(ch)
                return True
        return False

    async def start_monitoring(self):
        @self.client.on(events.NewMessage())
        async def handler(event):
            try:
                channel_info = None
                for ch in self.monitored_channels:
                    if event.chat_id == ch["entity"].id:
                        channel_info = ch
                        break
                if not channel_info:
                    return

                text = event.message.message
                if not text:
                    return

                logger.info(f"📩 {channel_info['title']}: {text[:120]}")

                signal = await self.analyzer.analyze_message(text, channel_info["title"])
                if signal and self.signal_callback:
                    await self.signal_callback(signal, channel_info)

            except Exception as e:
                logger.error(f"Monitor handler error: {e}")

        logger.info("✅ Monitoring started")
        await self.client.run_until_disconnected()


class BitgetTrader:
    """
    Single-account trader (Bitget) using ENV credentials.
    Per-user config only controls risk/auto_trade, not API keys.
    """

    def __init__(self):
        self.ex = BitgetExecutor()
        self.trade_history = []

    async def get_usdt_balance(self) -> float:
        try:
            return float(self.ex.fetch_usdt_balance())
        except Exception:
            return 0.0

    async def get_price(self, symbol: str) -> Optional[float]:
        try:
            return self.ex.fetch_last_price(symbol)
        except Exception:
            return None

    async def calculate_position_size(self, signal: TradingSignal, user_cfg: Dict) -> float:
        bal = await self.get_usdt_balance()
        if bal <= 0:
            return 0.0

        risk = float(user_cfg.get("risk", 3.0))
        leverage = int(user_cfg.get("leverage", signal.leverage))
        price = await self.get_price(signal.symbol)
        if not price:
            return 0.0

        # risk $ = balance * risk%
        risk_amount = bal * (risk / 100.0)
        position_value = risk_amount * leverage
        qty = position_value / price

        # Simple rounding
        if qty < 0.001:
            return round(qty, 6)
        if qty < 1:
            return round(qty, 4)
        return round(qty, 2)

    async def execute_signal(self, signal: TradingSignal, user_cfg: Dict) -> Dict:
        """
        SAFE_MODE=true => no real orders, only simulation.
        SAFE_MODE=false => send market order.
        """
        try:
            qty = await self.calculate_position_size(signal, user_cfg)
            if qty <= 0:
                return {"success": False, "error": "Invalid position size (qty <= 0)"}

            leverage = int(user_cfg.get("leverage", signal.leverage))

            if SAFE_MODE:
                return {
                    "success": True,
                    "paper": True,
                    "order_id": "PAPER",
                    "position_size": qty,
                    "message": f"[SAFE_MODE] Paper trade: {signal.direction} {signal.symbol} qty={qty} lev={leverage}",
                }

            order = self.ex.market_order(signal.symbol, signal.direction, qty, leverage=leverage)
            order_id = order.get("id") or order.get("orderId") or "UNKNOWN"
            return {
                "success": True,
                "paper": False,
                "order_id": str(order_id),
                "position_size": qty,
                "message": f"Order executed: {signal.direction} {signal.symbol} qty={qty}",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


class ConfigManager:
    def __init__(self, config_file="config.yaml"):
        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self) -> Dict:
        default = {
            "users": {},
            "channels": {},
            "risk_management": {"default_leverage": 10, "default_risk": 3.0, "max_risk": 5.0},
        }
        try:
            with open(self.config_file, "r") as f:
                data = yaml.safe_load(f) or {}
                default.update(data)
        except FileNotFoundError:
            pass
        return default

    def save_config(self):
        with open(self.config_file, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)

    def add_user(self, user_id: int):
        self.config["users"][str(user_id)] = {
            "auto_trade": False,
            "leverage": self.config["risk_management"]["default_leverage"],
            "risk": self.config["risk_management"]["default_risk"],
            "created_at": datetime.now().isoformat(),
        }
        self.save_config()

    def remove_user(self, user_id: int):
        if str(user_id) in self.config["users"]:
            del self.config["users"][str(user_id)]
            self.save_config()

    def get_user(self, user_id: int) -> Optional[Dict]:
        return self.config["users"].get(str(user_id))

    def set_user(self, user_id: int, key: str, value):
        u = self.get_user(user_id)
        if not u:
            return False
        u[key] = value
        self.save_config()
        return True

    def add_channel(self, username: str, title: str = ""):
        self.config["channels"][username] = {"title": title, "added_at": datetime.now().isoformat()}
        self.save_config()

    def remove_channel(self, username: str):
        if username in self.config["channels"]:
            del self.config["channels"][username]
            self.save_config()


class AutoTradingBot:
    def __init__(self, telegram_token: str, openai_api_key: str, telegram_api_id: int, telegram_api_hash: str, phone: str):
        self.admin_user_id = int(os.getenv("ADMIN_USER_ID", "0") or "0")
        self.config = ConfigManager()
        self.analyzer = AISignalAnalyzer(openai_api_key)
        self.monitor = ChannelMonitor(telegram_api_id, telegram_api_hash, phone, self.analyzer)
        self.monitor.set_signal_callback(self.handle_signal)

        self.trader = BitgetTrader()

        self.app = Application.builder().token(telegram_token).build()
        self._register_handlers()

    def _is_admin(self, update: Update) -> bool:
        if self.admin_user_id <= 0:
            return True  # if not set, allow
        return int(update.effective_user.id) == self.admin_user_id

    def _register_handlers(self):
        async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
            msg = (
                "🤖 Auto-Trading Bot (Bitget)\n"
                f"SAFE_MODE={'ON (paper only)' if SAFE_MODE else 'OFF (real execution)'}\n\n"
                "Commands:\n"
                "/status\n"
                "/add_channel @channel\n"
                "/remove_channel @channel\n"
                "/add_user USER_ID\n"
                "/remove_user USER_ID\n"
                "/toggle_auto_trading\n"
                "/set_risk PERCENT (0-5)\n"
                "/set_leverage NUMBER\n"
            )
            await update.message.reply_text(msg)

        async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
            u = self.config.get_user(update.effective_user.id)
            channels_count = len(self.monitor.monitored_channels)
            users_count = len(self.config.config.get("users", {}))
            bal = await self.trader.get_usdt_balance()
            msg = (
                "📊 Status\n\n"
                f"Monitored channels: {channels_count}\n"
                f"Registered users: {users_count}\n"
                f"USDT balance (Bitget): {bal}\n"
                f"Your ID: {update.effective_user.id}\n"
                f"Auto-trade: {u.get('auto_trade') if u else False}\n"
                f"Risk: {u.get('risk') if u else 'N/A'}%\n"
                f"Leverage: {u.get('leverage') if u else 'N/A'}\n"
            )
            await update.message.reply_text(msg)

        async def add_channel(update: Update, context: ContextTypes.DEFAULT_TYPE):
            if not self._is_admin(update):
                return await update.message.reply_text("❌ Admin only.")
            if len(context.args) != 1:
                return await update.message.reply_text("Usage: /add_channel @channel")
            ch = context.args[0]
            ok = await self.monitor.add_channel(ch)
            if ok:
                if not ch.startswith("@"):
                    ch = "@" + ch
                self.config.add_channel(ch)
                await update.message.reply_text(f"✅ Added channel {ch}")
            else:
                await update.message.reply_text("❌ Failed to add channel. Make sure it’s public & you can access it.")

        async def remove_channel(update: Update, context: ContextTypes.DEFAULT_TYPE):
            if not self._is_admin(update):
                return await update.message.reply_text("❌ Admin only.")
            if len(context.args) != 1:
                return await update.message.reply_text("Usage: /remove_channel @channel")
            ch = context.args[0]
            ok = await self.monitor.remove_channel(ch)
            if ok:
                if not ch.startswith("@"):
                    ch = "@" + ch
                self.config.remove_channel(ch)
                await update.message.reply_text(f"✅ Removed channel {ch}")
            else:
                await update.message.reply_text("❌ Channel not found in monitoring list.")

        async def add_user(update: Update, context: ContextTypes.DEFAULT_TYPE):
            if not self._is_admin(update):
                return await update.message.reply_text("❌ Admin only.")
            if len(context.args) != 1:
                return await update.message.reply_text("Usage: /add_user USER_ID")
            uid = int(context.args[0])
            self.config.add_user(uid)
            await update.message.reply_text(f"✅ User {uid} registered. (Auto-trade OFF by default)")

        async def remove_user(update: Update, context: ContextTypes.DEFAULT_TYPE):
            if not self._is_admin(update):
                return await update.message.reply_text("❌ Admin only.")
            if len(context.args) != 1:
                return await update.message.reply_text("Usage: /remove_user USER_ID")
            uid = int(context.args[0])
            self.config.remove_user(uid)
            await update.message.reply_text(f"✅ User {uid} removed.")

        async def toggle_auto(update: Update, context: ContextTypes.DEFAULT_TYPE):
            uid = int(update.effective_user.id)
            u = self.config.get_user(uid)
            if not u:
                return await update.message.reply_text("❌ You are not registered. Ask admin to /add_user your_id")
            new_val = not bool(u.get("auto_trade", False))
            self.config.set_user(uid, "auto_trade", new_val)
            await update.message.reply_text(f"✅ Auto-trading {'ENABLED' if new_val else 'DISABLED'}")

        async def set_risk(update: Update, context: ContextTypes.DEFAULT_TYPE):
            uid = int(update.effective_user.id)
            u = self.config.get_user(uid)
            if not u:
                return await update.message.reply_text("❌ You are not registered.")
            if len(context.args) != 1:
                return await update.message.reply_text("Usage: /set_risk 0-5")
            val = float(context.args[0])
            if val < 0 or val > float(self.config.config["risk_management"]["max_risk"]):
                return await update.message.reply_text("❌ Risk must be 0..5")
            self.config.set_user(uid, "risk", val)
            await update.message.reply_text(f"✅ Risk set to {val}%")

        async def set_leverage(update: Update, context: ContextTypes.DEFAULT_TYPE):
            uid = int(update.effective_user.id)
            u = self.config.get_user(uid)
            if not u:
                return await update.message.reply_text("❌ You are not registered.")
            if len(context.args) != 1:
                return await update.message.reply_text("Usage: /set_leverage NUMBER")
            lev = int(context.args[0])
            if lev < 1 or lev > 125:
                return await update.message.reply_text("❌ Leverage must be 1..125")
            self.config.set_user(uid, "leverage", lev)
            await update.message.reply_text(f"✅ Leverage set to {lev}")

        self.app.add_handler(CommandHandler("start", start))
        self.app.add_handler(CommandHandler("status", status))
        self.app.add_handler(CommandHandler("add_channel", add_channel))
        self.app.add_handler(CommandHandler("remove_channel", remove_channel))
        self.app.add_handler(CommandHandler("add_user", add_user))
        self.app.add_handler(CommandHandler("remove_user", remove_user))
        self.app.add_handler(CommandHandler("toggle_auto_trading", toggle_auto))
        self.app.add_handler(CommandHandler("set_risk", set_risk))
        self.app.add_handler(CommandHandler("set_leverage", set_leverage))

    async def handle_signal(self, signal: TradingSignal, channel_info: Dict):
        logger.info(f"🎯 Signal: {signal.direction} {signal.symbol} conf={signal.confidence_score}")

        executed = 0
        for uid_str, ucfg in self.config.config.get("users", {}).items():
            if not ucfg.get("auto_trade", False):
                continue

            uid = int(uid_str)
            result = await self.trader.execute_signal(signal, ucfg)

            if result.get("success"):
                executed += 1
                await self.notify(uid, signal, result)
            else:
                logger.error(f"Trade failed for {uid}: {result.get('error')}")

        logger.info(f"✅ Done. executed={executed}")

    async def notify(self, user_id: int, signal: TradingSignal, result: Dict):
        txt = (
            "🤖 AUTO TRADE\n\n"
            f"Direction: {signal.direction}\n"
            f"Symbol: {signal.symbol}\n"
            f"Confidence: {signal.confidence_score}%\n\n"
            f"Order ID: {result.get('order_id')}\n"
            f"Position size: {result.get('position_size')}\n"
            f"Mode: {'PAPER (SAFE_MODE)' if result.get('paper') else 'REAL'}\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        await self.app.bot.send_message(chat_id=user_id, text=txt)

    async def run(self):
        if not await self.monitor.initialize():
            raise RuntimeError("Telethon monitor init failed")

        # Start admin bot
        async with self.app:
            await self.app.initialize()
            await self.app.start()
            await self.app.updater.start_polling()

            logger.info("✅ Admin bot ready")
            logger.info("✅ Starting Telethon monitoring")
            monitor_task = asyncio.create_task(self.monitor.start_monitoring())
            await monitor_task


async def main():
    # Required env
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    TELEGRAM_API_ID = int(os.getenv("TELEGRAM_API_ID", "0") or "0")
    TELEGRAM_API_HASH = os.getenv("TELEGRAM_API_HASH", "")
    PHONE_NUMBER = os.getenv("PHONE_NUMBER", "")

    missing = []
    if not TELEGRAM_BOT_TOKEN: missing.append("TELEGRAM_BOT_TOKEN")
    if not OPENAI_API_KEY: missing.append("OPENAI_API_KEY")
    if TELEGRAM_API_ID <= 0: missing.append("TELEGRAM_API_ID")
    if not TELEGRAM_API_HASH: missing.append("TELEGRAM_API_HASH")
    if not PHONE_NUMBER: missing.append("PHONE_NUMBER")
    if not os.getenv("TELEGRAM_SESSION_STRING", "").strip(): missing.append("TELEGRAM_SESSION_STRING")

    if missing:
        logger.error(f"❌ Missing env vars: {', '.join(missing)}")
        return

    bot = AutoTradingBot(
        telegram_token=TELEGRAM_BOT_TOKEN,
        openai_api_key=OPENAI_API_KEY,
        telegram_api_id=TELEGRAM_API_ID,
        telegram_api_hash=TELEGRAM_API_HASH,
        phone=PHONE_NUMBER,
    )
    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())
