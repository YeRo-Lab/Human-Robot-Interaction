# Quick Gemini model check. Run from the server folder:  py -3 check_models.py
# Calls each model twice (plain, then with Yuno's system prompt) and prints status and time.
import os, time
from dotenv import load_dotenv
from google import genai
from google.genai import types, errors
import context

load_dotenv()
client = genai.Client(api_key=os.getenv("API_KEY"), http_options=types.HttpOptions(timeout=15000))
MODELS = [
    "gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-flash-lite-latest",
    "gemini-3-flash-preview", "gemini-3.5-flash", "gemini-2.5-flash",
]
SYSTEM = context.build_system_prompt()

def call(model, system):
    cfg = types.GenerateContentConfig(system_instruction=system) if system else None
    t0 = time.time()
    try:
        text = client.models.generate_content(model=model, contents="Hi! What does the lab work on?", config=cfg).text
        return "%.2fs  ok   %s" % (time.time() - t0, (text or "").replace("\n", " ")[:50])
    except errors.APIError as e:
        return "%.2fs  FAIL %s %s" % (time.time() - t0, e.code, str(e.message)[:70])
    except Exception as e:
        return "%.2fs  FAIL %s" % (time.time() - t0, type(e).__name__)

for m in MODELS:
    print("%-26s plain:  %s" % (m, call(m, None)), flush=True)
    print("%-26s prompt: %s" % ("", call(m, SYSTEM)), flush=True)