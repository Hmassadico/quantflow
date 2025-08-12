# test_telegram.py
import os, requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass
token = os.environ["TELEGRAM_BOT_TOKEN"]
chat_id = os.environ["TELEGRAM_CHAT_ID"]
r = requests.post(
    f"https://api.telegram.org/bot{token}/sendMessage",
    json={"chat_id": chat_id, "text": "Hello from QuantFlow 👋", "disable_web_page_preview": True},
    timeout=15,
)
r.raise_for_status()
print("OK:", r.json())
