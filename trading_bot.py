import os
import re
import json
import yaml
import time
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv
load_dotenv()

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.ext import (
    Application,
    ContextTypes,
    CallbackQueryHandler,
    MessageHandler,
    filters,
)

# Optional: AI analysis
USE_OPENAI = bool(os.getenv("OPENAI_API_KEY", "").strip())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

# Trading
EXCHANGE_DEFAULT = os.getenv("EXCHANGE", "bitget").lower()
SAFE_MODE_DEFAULT = os.getenv("SAFE_MODE", "ON").upper()  # ON = paper only

# Storage
CONFIG_FILE = os.getenv("CONFIG_FILE", "config.yaml")

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("wizard_bot")


# =========================
# Data model
# =========================
@dataclass
class TradingSignal:
    direction: str            # LONG/SHORT
    symbol: str               # BTCUSDT
    entry: Optional[float]    # None => MARKET
    take_profit: List[float]
    stop_loss: Optional[float]
    leverage: int
    raw_text: str
    confidence: float = 0.0


# =========================
# Simple Config Manager
# =========================
class ConfigManager:
    def __init__(self, path: str):
        self.path = path
        self.data = self._load()

    def _load(self) -> dict:
        default = {
            "users": {},        # user_id -> {exchange, api_key, api_secret, api_passphrase, auto_trade, risk, leverage}
            "signal_sources": {}  # source_chat_id(str) -> {title, enabled}
        }
        if not os.path.exists(self.path):
            return default
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            # merge
            default.update(loaded)
            default["users"] = default.get("users", {}) or {}
            default["signal_sources"] = default.get("signal_sources", {}) or {}
            return default
        except Exception as e:
            log.error(f"Failed to load config: {e}")
            return default

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            yaml.dump(self.data, f, sort_keys=False)

    def get_user(self, user_id: int) -> dict:
        return self.data["users"].get(str(user_id), {})

    def set_user(self, user_id: int, patch: dict):
        cur = self.data["users"].get(str(user_id), {})
        cur.update(patch)
        self.data["users"][str(user_id)] = cur
        self.save()

    def register_source(self, chat_id: int, title: str):
        self.data["signal_sources"][str(chat_id)] = {"title": title, "enabled": True}
        self.save()

    def remove_source(self, chat_id: int):
        if str(chat_id) in self.data["signal_sources"]:
            del self.data["signal_sources"][str(chat_id)]
            self.save()

    def is_source_allowed(self, chat_id: int) -> bool:
        src = self.data["signal_sources"].get(str(chat_id))
        return bool(src and src.get("enabled", True))

    def list_sources(self) -> List[dict]:
        out = []
        for k, v in self.data["signal_sources"].items():
            out.append({"chat_id": int(k), "title": v.get("title", ""), "enabled": v.get("enabled", True)})
        return out


cfg = ConfigManager(CONFIG_FILE)


# =========================
# UI helpers
# =========================
def main_menu_kb(user_id: int) -> InlineKeyboardMarkup:
    user = cfg.get_user(user_id)
    is_registered = bool(user.get("api_key"))
    auto = bool(user.get("auto_trade", False))
    safe = user.get("safe_mode", SAFE_MODE_DEFAULT)

    btn_setup = "✅ Update API" if is_registered else "🧩 Setup (Wizard)"
    btn_auto = "⛔ Auto-Trade OFF" if auto else "⚡ Auto-Trade ON"
    btn_safe = f"🧪 SAFE_MODE: {safe}"

    keyboard = [
        [InlineKeyboardButton(btn_setup, callback_data="MENU_SETUP")],
        [InlineKeyboardButton("📡 Register Signal Channel", callback_data="MENU_REG_SOURCE")],
        [InlineKeyboardButton("📋 Sources List", callback_data="MENU_LIST_SOURCES")],
        [InlineKeyboardButton(btn_safe, callback_data="MENU_TOGGLE_SAFE"),
         InlineKeyboardButton(btn_auto, callback_data="MENU_TOGGLE_AUTO")],
        [InlineKeyboardButton("📈 Analisa Coin", callback_data="MENU_ANALYZE")],
        [InlineKeyboardButton("📊 Status", callback_data="MENU_STATUS")],
    ]
    return InlineKeyboardMarkup(keyboard)


async def send_or_edit(update: Update, text: str, kb: Optional[InlineKeyboardMarkup] = None):
    if update.callback_query:
        await update.callback_query.answer()
        await update.callback_query.edit_message_text(text=text, reply_markup=kb, disable_web_page_preview=True)
    else:
        await update.message.reply_text(text=text, reply_markup=kb, disable_web_page_preview=True)


# =========================
# Signal Parsing (basic)
# =========================
def parse_signal(text: str) -> Optional[TradingSignal]:
    t = text.strip().upper()

    # direction
    direction = None
    if re.search(r"\bLONG\b", t): direction = "LONG"
    if re.search(r"\bSHORT\b", t): direction = "SHORT"
    if not direction:
        return None

    # symbol
    m = re.search(r"\b([A-Z0-9]{3,15})\s*(USDT)\b", t)
    symbol = None
    if m:
        symbol = f"{m.group(1)}USDT"
    else:
        # allow "BTC/USDT"
        m2 = re.search(r"\b([A-Z0-9]{3,15})\s*/\s*USDT\b", t)
        if m2:
            symbol = f"{m2.group(1)}USDT"
    if not symbol:
        return None

    # entry price optional
    entry = None
    m_entry = re.search(r"\bENTRY\b\s*([0-9]+(\.[0-9]+)?)", t)
    if m_entry:
        entry = float(m_entry.group(1))

    # leverage
    lev = 10
    m_lev = re.search(r"\bLEV(ERAGE)?\b\s*([0-9]{1,3})", t)
    if m_lev:
        lev = int(m_lev.group(2))

    # SL
    sl = None
    m_sl = re.search(r"\bSL\b\s*([0-9]+(\.[0-9]+)?)", t)
    if m_sl:
        sl = float(m_sl.group(1))

    # TP (allow 1..n)
    tp = []
    # "TP 43500 44000"
    m_tp = re.search(r"\bTP\b\s*([0-9\.\s,]+)", t)
    if m_tp:
        nums = re.findall(r"([0-9]+(\.[0-9]+)?)", m_tp.group(1))
        tp = [float(x[0]) for x in nums][:5]

    return TradingSignal(
        direction=direction,
        symbol=symbol,
        entry=entry,
        take_profit=tp,
        stop_loss=sl,
        leverage=lev,
        raw_text=text,
        confidence=90.0,  # basic parser confidence
    )


# =========================
# Bitget Executor (ccxt)
# =========================
class BitgetExecutor:
    """
    Real trading uses ccxt.
    Requires: ccxt
    Bitget usually needs apiKey + secret + password(passphrase).
    """
    def __init__(self, api_key: str, api_secret: str, passphrase: str, sandbox: bool = False):
        import ccxt  # type: ignore
        self.ex = ccxt.bitget({
            "apiKey": api_key,
            "secret": api_secret,
            "password": passphrase,  # IMPORTANT
            "enableRateLimit": True,
        })
        if sandbox:
            # NOTE: Bitget sandbox support depends on ccxt; leave false if not available.
            try:
                self.ex.set_sandbox_mode(True)
            except Exception:
                pass

    def place_order(self, signal: TradingSignal, qty: float) -> dict:
        # futures/perp: ccxt market type config can vary; simplest: createMarketOrder/Limit
        side = "buy" if signal.direction == "LONG" else "sell"
        if signal.entry is None:
            order = self.ex.create_market_order(signal.symbol, side, qty)
        else:
            order = self.ex.create_limit_order(signal.symbol, side, qty, signal.entry)
        return order


# =========================
# Bot State Machine (per user)
# =========================
# user_data keys:
# step: "WAIT_API_KEY" | "WAIT_API_SECRET" | "WAIT_API_PASSPHRASE" | "WAIT_ANALYZE_SYMBOL" | "WAIT_FORWARD_SOURCE"
# temp: dict scratch

def set_step(context: ContextTypes.DEFAULT_TYPE, step: Optional[str]):
    if step is None:
        context.user_data.pop("step", None)
        context.user_data.pop("temp", None)
    else:
        context.user_data["step"] = step
        context.user_data.setdefault("temp", {})

def get_step(context: ContextTypes.DEFAULT_TYPE) -> Optional[str]:
    return context.user_data.get("step")


# =========================
# Handlers
# =========================
async def on_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    # ensure user exists
    if not cfg.get_user(user_id):
        cfg.set_user(user_id, {
            "exchange": EXCHANGE_DEFAULT,
            "auto_trade": False,
            "risk": 1.0,
            "leverage": 10,
            "safe_mode": SAFE_MODE_DEFAULT,
        })

    text = (
        "🤖 *Lunero AI Trading Bot*\n\n"
        "Bot ini kerja seperti ini:\n"
        "1) Kamu *register channel sumber signal* (dengan forward 1 pesan dari channel itu ke bot)\n"
        "2) Bot akan *parse signal* → kirim *ALERT* ke DM kamu\n"
        "3) Jika Auto-Trade ON dan SAFE_MODE OFF → bot bisa kirim order ke exchange\n\n"
        "Klik menu di bawah."
    )
    await send_or_edit(update, text, main_menu_kb(user_id))


async def on_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    user_id = q.from_user.id
    user = cfg.get_user(user_id) or {}

    data = q.data

    if data == "MENU_SETUP":
        set_step(context, "WAIT_API_KEY")
        context.user_data["temp"] = {}
        await send_or_edit(update,
            "🧩 *Setup API (Step 1/3)*\n\nKirim *API KEY* kamu (teks saja).",
            None
        )
        return

    if data == "MENU_REG_SOURCE":
        set_step(context, "WAIT_FORWARD_SOURCE")
        await send_or_edit(update,
            "📡 *Register Signal Channel*\n\n"
            "Cara paling gampang:\n"
            "1) Tambahkan bot sebagai *admin* di channel sumber signal\n"
            "2) *Forward 1 pesan* dari channel itu ke bot ini\n\n"
            "Sekarang: *forward 1 pesan dari channel sumber signal* ke sini.",
            None
        )
        return

    if data == "MENU_LIST_SOURCES":
        sources = cfg.list_sources()
        if not sources:
            await send_or_edit(update, "📋 Belum ada source. Klik *Register Signal Channel* dulu.", main_menu_kb(user_id))
            return

        lines = ["📋 *Signal Sources:*"]
        for s in sources:
            lines.append(f"- `{s['chat_id']}` | {s['title']} | {'ON' if s['enabled'] else 'OFF'}")
        await send_or_edit(update, "\n".join(lines), main_menu_kb(user_id))
        return

    if data == "MENU_TOGGLE_SAFE":
        cur = user.get("safe_mode", SAFE_MODE_DEFAULT)
        newv = "OFF" if cur == "ON" else "ON"
        cfg.set_user(user_id, {"safe_mode": newv})
        await send_or_edit(update, f"🧪 SAFE_MODE sekarang: *{newv}*", main_menu_kb(user_id))
        return

    if data == "MENU_TOGGLE_AUTO":
        if not user.get("api_key"):
            await send_or_edit(update, "❌ Kamu belum setup API. Klik *Setup (Wizard)* dulu.", main_menu_kb(user_id))
            return
        newv = not bool(user.get("auto_trade", False))
        cfg.set_user(user_id, {"auto_trade": newv})
        await send_or_edit(update, f"⚡ Auto-Trade sekarang: *{'ON' if newv else 'OFF'}*", main_menu_kb(user_id))
        return

    if data == "MENU_ANALYZE":
        set_step(context, "WAIT_ANALYZE_SYMBOL")
        await send_or_edit(update, "📈 Kirim symbol untuk dianalisa. Contoh: `BTCUSDT`", None)
        return

    if data == "MENU_STATUS":
        sources = cfg.list_sources()
        text = (
            "📊 *Status*\n\n"
            f"Exchange: *{user.get('exchange', EXCHANGE_DEFAULT).upper()}*\n"
            f"SAFE_MODE: *{user.get('safe_mode', SAFE_MODE_DEFAULT)}*\n"
            f"Auto-Trade: *{user.get('auto_trade', False)}*\n"
            f"Risk: *{user.get('risk', 1.0)}%*\n"
            f"Leverage: *{user.get('leverage', 10)}*\n"
            f"Sources: *{len(sources)}*\n\n"
            f"Your ID: `{user_id}`"
        )
        await send_or_edit(update, text, main_menu_kb(user_id))
        return

    # fallback
    await send_or_edit(update, "Menu tidak dikenal.", main_menu_kb(user_id))


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user = cfg.get_user(user_id) or {}
    step = get_step(context)
    text = (update.message.text or "").strip()

    # ----- Wizard: API setup -----
    if step == "WAIT_API_KEY":
        cfg.set_user(user_id, {"api_key": text})
        set_step(context, "WAIT_API_SECRET")
        await update.message.reply_text("🧩 *Setup API (Step 2/3)*\n\nKirim *API SECRET* kamu (teks saja).", parse_mode="Markdown")
        return

    if step == "WAIT_API_SECRET":
        cfg.set_user(user_id, {"api_secret": text})
        set_step(context, "WAIT_API_PASSPHRASE")
        await update.message.reply_text(
            "🧩 *Setup API (Step 3/3)*\n\n"
            "Kirim *API PASSPHRASE* Bitget kamu.\n"
            "⚠️ Ini wajib di Bitget (biasanya).",
            parse_mode="Markdown"
        )
        return

    if step == "WAIT_API_PASSPHRASE":
        cfg.set_user(user_id, {"api_passphrase": text})
        set_step(context, None)
        await update.message.reply_text("✅ API tersimpan. Sekarang register channel sumber signal.", parse_mode="Markdown")
        await update.message.reply_text("Kembali ke menu:", reply_markup=main_menu_kb(user_id))
        return

    # ----- Analyze coin -----
    if step == "WAIT_ANALYZE_SYMBOL":
        symbol = text.upper().replace("/", "")
        if not symbol.endswith("USDT"):
            symbol += "USDT"

        set_step(context, None)

        # Simple analysis (no price fetch here to keep it stable)
        # You can enhance by calling OpenAI or adding ccxt fetchTicker.
        analysis = (
            f"📈 *Analisa (basic)*\n\n"
            f"Symbol: *{symbol}*\n"
            "Saran implementasi:\n"
            "- Ambil harga terakhir (ccxt fetchTicker)\n"
            "- Hitung EMA/RSI (ta-lib/pandas-ta)\n"
            "- Buat summary & bias\n\n"
            "Kalau kamu mau, aku bisa upgrade jadi analisa lengkap (EMA/RSI/MACD) + data real-time."
        )
        await update.message.reply_text(analysis, parse_mode="Markdown", reply_markup=main_menu_kb(user_id))
        return

    # If user types random text in DM, show menu
    await update.message.reply_text("Pilih dari menu:", reply_markup=main_menu_kb(user_id))


async def on_any_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    This handler catches:
    - forwarded messages to register a channel source
    - channel posts (signals) when bot is admin in that channel
    """
    user_id = update.effective_user.id if update.effective_user else None
    step = get_step(context)

    # ----- Register source: user forwards a message from a channel -----
    if step == "WAIT_FORWARD_SOURCE" and update.message:
        fwd = update.message.forward_from_chat
        if not fwd:
            await update.message.reply_text("❌ Forward-nya tidak terbaca. Forward *dari channel*, bukan copy-paste.")
            return

        # register source chat id
        cfg.register_source(fwd.id, fwd.title or fwd.username or str(fwd.id))
        set_step(context, None)

        await update.message.reply_text(
            f"✅ Source registered:\n- Title: {fwd.title}\n- Chat ID: {fwd.id}\n\n"
            "Sekarang kalau ada posting signal di channel itu, bot akan deteksi & kirim alert.",
        )
        if user_id:
            await update.message.reply_text("Kembali ke menu:", reply_markup=main_menu_kb(user_id))
        return

    # ----- Detect signals from channel posts -----
    # channel_post update arrives as update.channel_post (python-telegram-bot),
    # but also can arrive as update.message in some cases. We'll check both.
    msg = update.channel_post or update.message
    if not msg:
        return

    # Only handle if it's from registered source
    chat = msg.chat
    if not chat:
        return

    if not cfg.is_source_allowed(chat.id):
        return

    text = (msg.text or msg.caption or "").strip()
    if not text:
        return

    sig = parse_signal(text)
    if not sig:
        return

    # ALERT to all registered users (simple approach)
    # If you want only the admin/owner: send only to the person who configured.
    all_users = list(cfg.data.get("users", {}).keys())

    alert = (
        "🚨 *SIGNAL DETECTED*\n\n"
        f"Source: *{chat.title or chat.username or chat.id}*\n"
        f"Direction: *{sig.direction}*\n"
        f"Symbol: *{sig.symbol}*\n"
        f"Entry: *{sig.entry if sig.entry else 'MARKET'}*\n"
        f"TP: *{', '.join(map(str, sig.take_profit)) if sig.take_profit else '-'}*\n"
        f"SL: *{sig.stop_loss if sig.stop_loss else '-'}*\n"
        f"Lev: *{sig.leverage}*\n"
        f"Conf: *{sig.confidence}%*\n"
    )

    for uid in all_users:
        try:
            await context.bot.send_message(chat_id=int(uid), text=alert, parse_mode="Markdown")
        except Exception as e:
            log.warning(f"Failed to alert user {uid}: {e}")

    # ----- Auto-trade (Bitget) -----
    for uid in all_users:
        ucfg = cfg.get_user(int(uid))
        if not ucfg.get("auto_trade", False):
            continue

        safe_mode = ucfg.get("safe_mode", SAFE_MODE_DEFAULT)
        if safe_mode == "ON":
            # paper only
            try:
                await context.bot.send_message(
                    chat_id=int(uid),
                    text="🧪 SAFE_MODE=ON → tidak kirim order ke exchange (paper only).",
                )
            except Exception:
                pass
            continue

        # must have api fields
        api_key = ucfg.get("api_key")
        api_secret = ucfg.get("api_secret")
        api_pass = ucfg.get("api_passphrase")
        if not (api_key and api_secret and api_pass):
            try:
                await context.bot.send_message(
                    chat_id=int(uid),
                    text="❌ API Bitget belum lengkap. Butuh API_KEY + API_SECRET + API_PASSPHRASE.",
                )
            except Exception:
                pass
            continue

        # Very basic sizing: fixed risk not implemented (placeholder)
        qty = 0.001  # TODO: ganti dengan sizing berdasarkan balance+risk

        try:
            executor = BitgetExecutor(api_key, api_secret, api_pass, sandbox=False)
            order = executor.place_order(sig, qty)
            await context.bot.send_message(
                chat_id=int(uid),
                text=f"✅ Order sent to Bitget:\n`{json.dumps(order, indent=2)[:3500]}`",
                parse_mode="Markdown"
            )
        except Exception as e:
            await context.bot.send_message(chat_id=int(uid), text=f"❌ Order failed: {e}")


async def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN")

    app = Application.builder().token(token).build()

    # /start
    app.add_handler(MessageHandler(filters.COMMAND & filters.Regex(r"^/start$"), on_start))

    # menu buttons
    app.add_handler(CallbackQueryHandler(on_menu))

    # text input (wizard steps)
    app.add_handler(MessageHandler(filters.TEXT & filters.ChatType.PRIVATE, on_text))

    # catch-all for forwarded message registration + channel posts signal detection
    app.add_handler(MessageHandler(filters.ALL, on_any_message))

    log.info("Bot running...")
    await app.run_polling(close_loop=False)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
