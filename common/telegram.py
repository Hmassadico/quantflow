# common/telegram.py
import os, requests
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass



TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_PARSE_MODE = os.getenv("TELEGRAM_PARSE_MODE", "Markdown")  # or MarkdownV2 / HTML
TELEGRAM_SILENT = os.getenv("TELEGRAM_SILENT", "false").lower() in {"1","true","yes"}

def send_telegram_message(text: str) -> dict:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        raise RuntimeError("Telegram is not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.")

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": TELEGRAM_PARSE_MODE,
        "disable_notification": TELEGRAM_SILENT,
        "disable_web_page_preview": True,
    }
    r = requests.post(url, json=payload, timeout=15)
    r.raise_for_status()
    return r.json()
