import os
import re
import json
import time
import yaml
import logging
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any, Tuple

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ConversationHandler,
    ContextTypes,
    filters,
)

# =========================
# Logging
# =========================
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("trading_bot")

# =========================
# ENV
# =========================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()

EXCHANGE = os.getenv("EXCHANGE", "bitget").strip().lower()  # bitget | bybit (bitget implemented)
SAFE_MODE = os.getenv("SAFE_MODE", "ON").strip().upper()    # ON = paper only, OFF = real trade
DEFAULT_LEVERAGE = int(os.getenv("DEFAULT_LEVERAGE", "10"))
DEFAULT_RISK_PCT = float(os.getenv("DEFAULT_RISK_PCT", "1.0"))  # 0-5 recommended
ALERT_CHANNEL_ID = os.getenv("ALERT_CHANNEL_ID", "").strip()  # optional: channel id for alerts (e.g. -1001234567890)
DELETE_WEBHOOK = os.getenv("DELETE_WEBHOOK", "true").strip().lower() in ("1", "true", "yes")

CONFIG_FILE = os.getenv("CONFIG_FILE", "config.yaml")

# Optional: OpenAI (only used if USE_AI=1)
USE_AI = os.getenv("USE_AI", "0").strip() in ("1", "true", "yes")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

# =========================
# Conversation states
# =========================
(
    ST_MENU,
    ST_SET_API_KEY,
    ST_SET_API_SECRET,
    ST_SET_PASSPHRASE,
    ST_SET_ALERT_CHANNEL,
    ST_SET_RISK,
    ST_SET_LEVERAGE,
) = range(7)


# =========================
# Data model
# =========================
@dataclass
class TradingSignal:
    direction: str               # LONG / SHORT
    symbol: str                  # BTC/USDT or BTCUSDT normalized
    entry: Optional[float]       # entry price (optional)
    tps: List[float]             # take profit levels
    sl: Optional[float]          # stop loss
    leverage: int
    raw: str
    confidence: float = 0.0      # 0-100


# =========================
# Config storage
# =========================
class ConfigStore:
    def __init__(self, path: str):
        self.path = path
        self.data = self._load()

    def _default(self) -> Dict[str, Any]:
        return {
            "users": {
                # "123": {"api_key":"", "api_secret":"", "passphrase":"", "auto_trade": False, "risk": 1.0, "leverage": 10}
            },
            "monitored_channels": {
                # "-100123...": {"title":"...", "enabled": True}
            },
            "alert_channel_id": None,  # overrides env if set
        }

    def _load(self) -> Dict[str, Any]:
        if not os.path.exists(self.path):
            return self._default()
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            base = self._default()
            # shallow merge
            base.update(loaded)
            base["users"] = loaded.get("users", base["users"])
            base["monitored_channels"] = loaded.get("monitored_channels", base["monitored_channels"])
            return base
        except Exception as e:
            logger.error("Failed to load config: %s", e)
            return self._default()

    def save(self) -> None:
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                yaml.safe_dump(self.data, f, sort_keys=False, allow_unicode=True)
        except Exception as e:
            logger.error("Failed to save config: %s", e)

    def user_get(self, user_id: int) -> Optional[Dict[str, Any]]:
        return self.data.get("users", {}).get(str(user_id))

    def user_ensure(self, user_id: int) -> Dict[str, Any]:
        users = self.data.setdefault("users", {})
        users.setdefault(str(user_id), {
            "api_key": "",
            "api_secret": "",
            "passphrase": "",
            "auto_trade": False,
            "risk": DEFAULT_RISK_PCT,
            "leverage": DEFAULT_LEVERAGE,
        })
        return users[str(user_id)]

    def user_set(self, user_id: int, key: str, value: Any) -> None:
        u = self.user_ensure(user_id)
        u[key] = value
        self.save()

    def add_monitored_channel(self, chat_id: int, title: str) -> None:
        ch = self.data.setdefault("monitored_channels", {})
        ch[str(chat_id)] = {"title": title, "enabled": True}
        self.save()

    def remove_monitored_channel(self, chat_id: int) -> None:
        ch = self.data.setdefault("monitored_channels", {})
        ch.pop(str(chat_id), None)
        self.save()

    def list_channels(self) -> Dict[str, Any]:
        return self.data.get("monitored_channels", {})

    def is_monitored(self, chat_id: int) -> bool:
        ch = self.data.get("monitored_channels", {})
        item = ch.get(str(chat_id))
        return bool(item and item.get("enabled", True))

    def get_alert_channel_id(self) -> Optional[int]:
        # priority: config > env
        if self.data.get("alert_channel_id"):
            try:
                return int(self.data["alert_channel_id"])
            except:
                return None
        if ALERT_CHANNEL_ID:
            try:
                return int(ALERT_CHANNEL_ID)
            except:
                return None
        return None

    def set_alert_channel_id(self, chat_id: int) -> None:
        self.data["alert_channel_id"] = int(chat_id)
        self.save()


# =========================
# Signal parsing (regex)
# =========================
class SignalParser:
    # Accept formats:
    # LONG BTCUSDT
    # ENTRY 43000
    # TP 43500 44000
    # SL 42500
    # LEV 10
    DIRECTION_RE = re.compile(r"\b(LONG|SHORT|BUY|SELL)\b", re.IGNORECASE)
    SYMBOL_RE = re.compile(r"\b([A-Z0-9]{2,15})(?:/)?(USDT)\b", re.IGNORECASE)
    ENTRY_RE = re.compile(r"\bENTRY\b[:\s]*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
    SL_RE = re.compile(r"\bSL\b[:\s]*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
    LEV_RE = re.compile(r"\b(LEV|LEVERAGE)\b[:\s]*([0-9]{1,3})", re.IGNORECASE)
    # TP can be: "TP 43500 44000" or multiple lines "TP1 43500" etc.
    TP_ALL_RE = re.compile(r"\bTP\d*\b[:\s]*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)

    def parse(self, text: str) -> Optional[TradingSignal]:
        if not text:
            return None
        t = text.strip()

        m_dir = self.DIRECTION_RE.search(t)
        if not m_dir:
            return None
        direction = m_dir.group(1).upper()
        if direction == "BUY":
            direction = "LONG"
        if direction == "SELL":
            direction = "SHORT"

        m_sym = self.SYMBOL_RE.search(t)
        if not m_sym:
            return None
        base = m_sym.group(1).upper()
        symbol = f"{base}/USDT"

        entry = None
        m_entry = self.ENTRY_RE.search(t)
        if m_entry:
            entry = float(m_entry.group(1))

        sl = None
        m_sl = self.SL_RE.search(t)
        if m_sl:
            sl = float(m_sl.group(1))

        leverage = DEFAULT_LEVERAGE
        m_lev = self.LEV_RE.search(t)
        if m_lev:
            leverage = int(m_lev.group(2))

        tps = [float(x) for x in self.TP_ALL_RE.findall(t)]
        # Also support: "TP 43500 44000" in one line
        if not tps:
            # try a simpler split after "TP"
            lines = t.splitlines()
            for line in lines:
                if re.search(r"\bTP\b", line, re.IGNORECASE):
                    nums = re.findall(r"([0-9]+(?:\.[0-9]+)?)", line)
                    if nums:
                        tps.extend([float(n) for n in nums])
        # dedup keep order
        seen = set()
        tps2 = []
        for x in tps:
            if x not in seen:
                seen.add(x)
                tps2.append(x)
        tps = tps2

        # confidence heuristic
        conf = 60.0
        if entry is not None:
            conf += 10
        if sl is not None:
            conf += 10
        if tps:
            conf += 10
        if leverage:
            conf += 5
        conf = min(conf, 95.0)

        return TradingSignal(
            direction=direction,
            symbol=symbol,
            entry=entry,
            tps=tps,
            sl=sl,
            leverage=leverage,
            raw=t,
            confidence=conf,
        )


# =========================
# Exchange executor (Bitget via ccxt)
# =========================
class BitgetExecutor:
    """
    Real trading uses CCXT.
    For Bitget futures/swap you usually need:
    - apiKey
    - secret
    - password (passphrase)
    """
    def __init__(self, api_key: str, api_secret: str, passphrase: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self._ex = None

    def _init(self):
        if self._ex:
            return
        import ccxt  # pip install ccxt
        self._ex = ccxt.bitget({
            "apiKey": self.api_key,
            "secret": self.api_secret,
            "password": self.passphrase,
            "enableRateLimit": True,
            # If you need futures, bitget uses "swap" markets in ccxt:
            "options": {"defaultType": "swap"},
        })

    def place_market(self, symbol: str, side: str, amount: float, params: Optional[dict] = None) -> Dict[str, Any]:
        self._init()
        params = params or {}
        # side: buy/sell
        return self._ex.create_order(symbol, "market", side, amount, None, params)

    def set_leverage(self, symbol: str, leverage: int) -> None:
        self._init()
        # ccxt unified: setLeverage on some exchanges
        if hasattr(self._ex, "set_leverage"):
            self._ex.set_leverage(leverage, symbol)
        else:
            # best effort
            pass

    def fetch_ticker_last(self, symbol: str) -> float:
        self._init()
        t = self._ex.fetch_ticker(symbol)
        return float(t["last"])

    def market_amount_from_risk(
        self,
        symbol: str,
        risk_pct: float,
        leverage: int,
        usdt_balance: float,
        entry_price: Optional[float],
    ) -> float:
        """
        Simple sizing:
        position_value = balance * (risk_pct/100) * leverage
        qty = position_value / price
        """
        price = entry_price if entry_price else self.fetch_ticker_last(symbol)
        position_value = usdt_balance * (risk_pct / 100.0) * leverage
        qty = position_value / price
        # round a bit
        return float(f"{qty:.4f}")

    def fetch_usdt_balance(self) -> float:
        self._init()
        bal = self._ex.fetch_balance()
        # in swap accounts, USDT might show in total/free
        usdt = 0.0
        if "USDT" in bal.get("free", {}):
            usdt = float(bal["free"]["USDT"])
        elif "USDT" in bal.get("total", {}):
            usdt = float(bal["total"]["USDT"])
        return usdt


# =========================
# Telegram Bot App
# =========================
class TradingBotApp:
    def __init__(self, store: ConfigStore):
        self.store = store
        self.parser = SignalParser()

    # ---------- UI helpers ----------
    def _menu_kb(self) -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup([
            [InlineKeyboardButton("✅ Setup API (Bitget)", callback_data="menu_setup_api")],
            [InlineKeyboardButton("📌 Set Alert Channel", callback_data="menu_set_alert")],
            [InlineKeyboardButton("📺 Link Channel (monitor)", callback_data="menu_link_channel")],
            [InlineKeyboardButton("⚡ Toggle Auto-Trade", callback_data="menu_toggle_autotrade")],
            [InlineKeyboardButton("💰 Set Risk %", callback_data="menu_set_risk")],
            [InlineKeyboardButton("📈 Set Leverage", callback_data="menu_set_lev")],
            [InlineKeyboardButton("📊 Status", callback_data="menu_status")],
        ])

    async def _send_menu(self, update: Update, text: str) -> None:
        if update.message:
            await update.message.reply_text(text, reply_markup=self._menu_kb())
        elif update.callback_query:
            await update.callback_query.message.reply_text(text, reply_markup=self._menu_kb())

    # ---------- Commands ----------
    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = update.effective_user.id
        self.store.user_ensure(uid)
        mode = SAFE_MODE
        ex = EXCHANGE.upper()
        msg = (
            f"🤖 *Auto-Trading Bot*\n"
            f"Exchange: *{ex}*\n"
            f"SAFE_MODE: *{mode}*  (ON = paper only)\n\n"
            f"Gunakan menu tombol di bawah untuk setup step-by-step.\n"
            f"User ID kamu: `{uid}`"
        )
        await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN, reply_markup=self._menu_kb())

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await self._status(update)

    async def cmd_toggle(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = update.effective_user.id
        u = self.store.user_ensure(uid)
        u["auto_trade"] = not bool(u.get("auto_trade", False))
        self.store.save()
        await update.message.reply_text(f"✅ Auto-trade = {u['auto_trade']}")

    async def cmd_set_risk(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = update.effective_user.id
        if not context.args:
            await update.message.reply_text("Usage: /set_risk 1.0  (0-5)")
            return
        try:
            risk = float(context.args[0])
            if risk < 0 or risk > 5:
                raise ValueError("range")
            self.store.user_set(uid, "risk", risk)
            await update.message.reply_text(f"✅ Risk set to {risk}%")
        except:
            await update.message.reply_text("❌ Risk invalid. Pakai 0-5, contoh: /set_risk 1.0")

    async def cmd_set_leverage(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = update.effective_user.id
        if not context.args:
            await update.message.reply_text("Usage: /set_leverage 10")
            return
        try:
            lev = int(context.args[0])
            if lev < 1 or lev > 125:
                raise ValueError("range")
            self.store.user_set(uid, "leverage", lev)
            await update.message.reply_text(f"✅ Leverage set to {lev}")
        except:
            await update.message.reply_text("❌ Leverage invalid. 1-125, contoh: /set_leverage 10")

    async def cmd_link_here(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Jalankan command ini DI DALAM CHANNEL (sebagai post di channel),
        bot akan menyimpan chat_id channel tsb sebagai monitored.
        """
        if update.channel_post:
            chat = update.channel_post.chat
            self.store.add_monitored_channel(chat.id, chat.title or str(chat.id))
            await update.channel_post.reply_text("✅ Channel ini sekarang DIMONITOR oleh bot.")
        else:
            await update.message.reply_text("Kirim /link_here di DALAM CHANNEL (bot harus jadi admin).")

    # ---------- Callback menu ----------
    async def on_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        q = update.callback_query
        await q.answer()
        uid = q.from_user.id
        self.store.user_ensure(uid)

        data = q.data

        if data == "menu_status":
            await self._status(update)
            return ConversationHandler.END

        if data == "menu_toggle_autotrade":
            u = self.store.user_ensure(uid)
            u["auto_trade"] = not bool(u.get("auto_trade", False))
            self.store.save()
            await q.message.reply_text(f"✅ Auto-trade sekarang: *{u['auto_trade']}*", parse_mode=ParseMode.MARKDOWN)
            await self._send_menu(update, "Menu:")
            return ConversationHandler.END

        if data == "menu_setup_api":
            await q.message.reply_text(
                "Masukkan *API KEY Bitget* kamu.\n\n"
                "Catatan: Bitget butuh 3 item: API Key, API Secret, dan *Passphrase*.",
                parse_mode=ParseMode.MARKDOWN,
            )
            return ST_SET_API_KEY

        if data == "menu_set_alert":
            await q.message.reply_text(
                "Kirim *ALERT_CHANNEL_ID*.\n\n"
                "Cara gampang:\n"
                "1) Tambahkan bot sebagai admin di channel alert\n"
                "2) Post pesan apa saja di channel\n"
                "3) Bot akan melihat chat_id di log/status.\n\n"
                "Atau kirim angka chat_id langsung (format: -100xxxxxxxxxx).",
                parse_mode=ParseMode.MARKDOWN,
            )
            return ST_SET_ALERT_CHANNEL

        if data == "menu_set_risk":
            await q.message.reply_text("Kirim angka risk % (0-5). Contoh: `1.0`", parse_mode=ParseMode.MARKDOWN)
            return ST_SET_RISK

        if data == "menu_set_lev":
            await q.message.reply_text("Kirim leverage (1-125). Contoh: `10`", parse_mode=ParseMode.MARKDOWN)
            return ST_SET_LEVERAGE

        if data == "menu_link_channel":
            await q.message.reply_text(
                "Untuk monitor channel:\n"
                "1) Tambahkan bot sebagai *admin* di channel target\n"
                "2) Di channel itu, kirim command: `/link_here`\n\n"
                "Setelah itu, semua post signal di channel akan diproses otomatis.",
                parse_mode=ParseMode.MARKDOWN,
            )
            await self._send_menu(update, "Menu:")
            return ConversationHandler.END

        await self._send_menu(update, "Menu:")
        return ConversationHandler.END

    # ---------- Setup steps ----------
    async def step_set_api_key(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = update.effective_user.id
        api_key = (update.message.text or "").strip()
        if len(api_key) < 10:
            await update.message.reply_text("❌ API key tidak valid. Coba lagi.")
            return ST_SET_API_KEY
        self.store.user_set(uid, "api_key", api_key)
        await update.message.reply_text("Sekarang kirim *API SECRET*.", parse_mode=ParseMode.MARKDOWN)
        return ST_SET_API_SECRET

    async def step_set_api_secret(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = update.effective_user.id
        api_secret = (update.message.text or "").strip()
        if len(api_secret) < 10:
            await update.message.reply_text("❌ API secret tidak valid. Coba lagi.")
            return ST_SET_API_SECRET
        self.store.user_set(uid, "api_secret", api_secret)
        await update.message.reply_text("Sekarang kirim *PASSPHRASE Bitget* (yang kamu set saat buat API).", parse_mode=ParseMode.MARKDOWN)
        return ST_SET_PASSPHRASE

    async def step_set_passphrase(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = update.effective_user.id
        passphrase = (update.message.text or "").strip()
        if len(passphrase) < 4:
            await update.message.reply_text("❌ Passphrase tidak valid. Coba lagi.")
            return ST_SET_PASSPHRASE
        self.store.user_set(uid, "passphrase", passphrase)
        await update.message.reply_text("✅ API Bitget tersimpan.")
        await self._send_menu(update, "Menu:")
        return ConversationHandler.END

    async def step_set_alert_channel(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        txt = (update.message.text or "").strip()
        try:
            cid = int(txt)
            self.store.set_alert_channel_id(cid)
            await update.message.reply_text(f"✅ Alert channel id diset ke: `{cid}`", parse_mode=ParseMode.MARKDOWN)
            await self._send_menu(update, "Menu:")
            return ConversationHandler.END
        except:
            await update.message.reply_text("❌ Harus angka chat_id. Contoh: `-1001234567890`", parse_mode=ParseMode.MARKDOWN)
            return ST_SET_ALERT_CHANNEL

    async def step_set_risk(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = update.effective_user.id
        try:
            risk = float((update.message.text or "").strip())
            if risk < 0 or risk > 5:
                raise ValueError("range")
            self.store.user_set(uid, "risk", risk)
            await update.message.reply_text(f"✅ Risk set: {risk}%")
            await self._send_menu(update, "Menu:")
            return ConversationHandler.END
        except:
            await update.message.reply_text("❌ Risk invalid. 0-5. Contoh: 1.0")
            return ST_SET_RISK

    async def step_set_leverage(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = update.effective_user.id
        try:
            lev = int((update.message.text or "").strip())
            if lev < 1 or lev > 125:
                raise ValueError("range")
            self.store.user_set(uid, "leverage", lev)
            await update.message.reply_text(f"✅ Leverage set: {lev}")
            await self._send_menu(update, "Menu:")
            return ConversationHandler.END
        except:
            await update.message.reply_text("❌ Leverage invalid. 1-125. Contoh: 10")
            return ST_SET_LEVERAGE

    # ---------- Core: handle channel posts ----------
    async def on_channel_post(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        # channel post update
        post = update.channel_post
        if not post or not post.text:
            return

        chat = post.chat
        chat_id = chat.id

        # Process only monitored channels
        if not self.store.is_monitored(chat_id):
            return

        text = post.text
        sig = self.parser.parse(text)
        if not sig:
            return

        logger.info("Signal detected in channel %s (%s): %s %s conf=%.1f",
                    chat.title, chat_id, sig.direction, sig.symbol, sig.confidence)

        # Send alert
        await self._send_alert(context, sig, source_channel=chat)

        # Auto trade for every user who enabled
        # NOTE: This is high-risk. Make sure SAFE_MODE=ON while testing.
        if SAFE_MODE == "ON":
            logger.info("SAFE_MODE=ON -> skip real trade (paper only).")
            return

        # Execute trade for users that enabled auto_trade
        users = self.store.data.get("users", {})
        for uid_str, uconf in users.items():
            try:
                uid = int(uid_str)
            except:
                continue
            if not uconf.get("auto_trade", False):
                continue

            # must have api credentials
            api_key = (uconf.get("api_key") or "").strip()
            api_secret = (uconf.get("api_secret") or "").strip()
            passphrase = (uconf.get("passphrase") or "").strip()
            if not (api_key and api_secret and passphrase):
                logger.warning("User %s missing api credentials -> skip trade", uid)
                continue

            risk = float(uconf.get("risk", DEFAULT_RISK_PCT))
            lev = int(uconf.get("leverage", sig.leverage or DEFAULT_LEVERAGE))

            await self._execute_bitget_trade(context, uid, sig, api_key, api_secret, passphrase, risk, lev)

    async def _send_alert(self, context: ContextTypes.DEFAULT_TYPE, sig: TradingSignal, source_channel):
        alert_to = self.store.get_alert_channel_id()
        # if not set, send to source channel itself
        if not alert_to:
            alert_to = source_channel.id

        msg = (
            f"🚨 *SIGNAL DETECTED*\n"
            f"Channel: *{source_channel.title}*\n\n"
            f"• Direction: *{sig.direction}*\n"
            f"• Symbol: *{sig.symbol}*\n"
            f"• Entry: `{sig.entry}`\n"
            f"• TP: `{sig.tps}`\n"
            f"• SL: `{sig.sl}`\n"
            f"• Lev: `{sig.leverage}`\n"
            f"• Confidence: *{sig.confidence:.1f}%*\n\n"
            f"_SAFE_MODE: {SAFE_MODE}_"
        )
        try:
            await context.bot.send_message(chat_id=alert_to, text=msg, parse_mode=ParseMode.MARKDOWN)
        except Exception as e:
            logger.error("Failed send alert: %s", e)

    async def _execute_bitget_trade(
        self,
        context: ContextTypes.DEFAULT_TYPE,
        user_id: int,
        sig: TradingSignal,
        api_key: str,
        api_secret: str,
        passphrase: str,
        risk: float,
        lev: int,
    ):
        """
        Places market order. TP/SL automation differs per exchange;
        here we only place main market order to prove execution works.
        """
        try:
            ex = BitgetExecutor(api_key, api_secret, passphrase)

            # Set leverage (best-effort)
            try:
                ex.set_leverage(sig.symbol, lev)
            except Exception as e:
                logger.warning("set_leverage failed: %s", e)

            # Fetch balance and size
            bal = ex.fetch_usdt_balance()
            if bal <= 0:
                raise RuntimeError("USDT balance is 0 (or cannot fetch balance).")

            qty = ex.market_amount_from_risk(
                symbol=sig.symbol,
                risk_pct=risk,
                leverage=lev,
                usdt_balance=bal,
                entry_price=sig.entry,
            )

            side = "buy" if sig.direction == "LONG" else "sell"

            # Some Bitget swap markets in ccxt are like "BTC/USDT:USDT"
            # We'll attempt to load markets and map symbol if needed.
            try:
                # re-init markets by calling fetch ticker
                _ = ex.fetch_ticker_last(sig.symbol)
                used_symbol = sig.symbol
            except Exception:
                # fallback common swap symbol format
                used_symbol = f"{sig.symbol}:USDT"

            order = ex.place_market(used_symbol, side, qty)

            logger.info("Trade executed for user %s: %s", user_id, order.get("id", "no-id"))

            # Notify user in DM
            msg = (
                f"✅ *AUTO TRADE EXECUTED*\n\n"
                f"User: `{user_id}`\n"
                f"Side: *{side.upper()}*\n"
                f"Symbol: *{used_symbol}*\n"
                f"Qty: `{qty}`\n"
                f"Lev: `{lev}`\n"
                f"Risk: `{risk}%`\n"
                f"OrderId: `{order.get('id', 'N/A')}`\n"
            )
            await context.bot.send_message(chat_id=user_id, text=msg, parse_mode=ParseMode.MARKDOWN)

        except Exception as e:
            logger.error("Trade failed user %s: %s", user_id, e)
            try:
                await context.bot.send_message(chat_id=user_id, text=f"❌ Trade gagal: {e}")
            except:
                pass

    async def _status(self, update: Update):
        uid = update.effective_user.id
        u = self.store.user_get(uid) or {}
        chans = self.store.list_channels()
        alert_id = self.store.get_alert_channel_id()

        msg = (
            f"📊 *Status*\n\n"
            f"Monitored channels: *{len(chans)}*\n"
            f"Registered users: *{len(self.store.data.get('users', {}))}*\n"
            f"Your ID: `{uid}`\n"
            f"Auto-trade: *{bool(u.get('auto_trade', False))}*\n"
            f"Risk: `{u.get('risk', DEFAULT_RISK_PCT)}%`\n"
            f"Leverage: `{u.get('leverage', DEFAULT_LEVERAGE)}`\n"
            f"SAFE_MODE: *{SAFE_MODE}*\n"
            f"Alert channel id: `{alert_id}`\n\n"
            f"⚠️ Agar bot bisa baca channel: bot harus *admin* di channel tsb.\n"
            f"Link channel: kirim `/link_here` di dalam channel."
        )
        if update.message:
            await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN, reply_markup=self._menu_kb())
        elif update.callback_query:
            await update.callback_query.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN, reply_markup=self._menu_kb())


# =========================
# Build & Run
# =========================
def build_app() -> Application:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN env")

    store = ConfigStore(CONFIG_FILE)
    botapp = TradingBotApp(store)

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # (Optional) prevent "Conflict: terminated by other getUpdates request"
    # and ensure polling can start
    app.post_init = post_init_factory(DELETE_WEBHOOK)

    # Menu conversation
    conv = ConversationHandler(
        entry_points=[CallbackQueryHandler(botapp.on_menu, pattern=r"^menu_")],
        states={
            ST_SET_API_KEY: [MessageHandler(filters.TEXT & ~filters.COMMAND, botapp.step_set_api_key)],
            ST_SET_API_SECRET: [MessageHandler(filters.TEXT & ~filters.COMMAND, botapp.step_set_api_secret)],
            ST_SET_PASSPHRASE: [MessageHandler(filters.TEXT & ~filters.COMMAND, botapp.step_set_passphrase)],
            ST_SET_ALERT_CHANNEL: [MessageHandler(filters.TEXT & ~filters.COMMAND, botapp.step_set_alert_channel)],
            ST_SET_RISK: [MessageHandler(filters.TEXT & ~filters.COMMAND, botapp.step_set_risk)],
            ST_SET_LEVERAGE: [MessageHandler(filters.TEXT & ~filters.COMMAND, botapp.step_set_leverage)],
        },
        fallbacks=[CommandHandler("start", botapp.cmd_start)],
        allow_reentry=True,
    )

    # Commands
    app.add_handler(CommandHandler("start", botapp.cmd_start))
    app.add_handler(CommandHandler("status", botapp.cmd_status))
    app.add_handler(CommandHandler("toggle_auto_trading", botapp.cmd_toggle))
    app.add_handler(CommandHandler("set_risk", botapp.cmd_set_risk))
    app.add_handler(CommandHandler("set_leverage", botapp.cmd_set_leverage))
    app.add_handler(CommandHandler("link_here", botapp.cmd_link_here))

    # Buttons
    app.add_handler(conv)

    # Channel posts handler (THIS is where signals are detected)
    app.add_handler(MessageHandler(filters.UpdateType.CHANNEL_POSTS & filters.TEXT, botapp.on_channel_post))

    return app


def post_init_factory(delete_webhook: bool):
    async def _post_init(app: Application):
        if delete_webhook:
            try:
                await app.bot.delete_webhook(drop_pending_updates=True)
                logger.info("delete_webhook(drop_pending_updates=True) OK")
            except Exception as e:
                logger.warning("delete_webhook failed: %s", e)
    return _post_init


def main():
    app = build_app()
    logger.info("Starting bot... EXCHANGE=%s SAFE_MODE=%s", EXCHANGE, SAFE_MODE)
    # ✅ IMPORTANT: no asyncio.run() here -> avoids event loop error
    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()
