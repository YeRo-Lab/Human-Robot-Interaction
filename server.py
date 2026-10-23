# server.py (Python 3)
from flask import Flask, request, jsonify
from google import genai
from google.genai import types, errors
import gpu_libs
gpu_libs.add_nvidia_libs_to_path()  # must run before faster_whisper loads CUDA
from faster_whisper import WhisperModel
import numpy as np
import os
import shutil
from dotenv import load_dotenv
from deepface import DeepFace
import time
import queue
import threading
from collections import OrderedDict
import context
import httpx
import re

load_dotenv()

api_key = os.getenv("API_KEY")

MODEL_NAME = "gemini-3.5-flash-lite"
FALLBACK_MODELS = ["gemini-3.1-flash-lite", "gemini-3.5-flash", "gemini-3-flash-preview", "gemini-flash-lite-latest"]  # tried in order if the one before fails  # tried in order if the one before fails
FALLBACK_ERROR_CODES = (429, 500, 503, 504)  # quota exceeded, server error, overloaded, deadline exceeded
HEDGE_DELAY_SECONDS = 3.0    # if no model has answered after this long, start the next one too (the first answer wins)
GEMINI_BUDGET_SECONDS = 15   # give up after this long; the robot waits 20 s total, and speech-to-text takes about 1 s

GEMINI_TIMEOUT_MS = 10000  # Google rejects anything under 10 s

SYSTEM_PROMPT = context.build_system_prompt()

HISTORY_MAX_TURNS = 8    # exchanges (you + YeRo) remembered per visit
MAX_CONVERSATIONS = 20   # oldest visits are forgotten past this
histories = OrderedDict()  # conversation_id -> list of types.Content
history_lock = threading.Lock()

client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT_MS))

def generate(prompt, system_instruction=None, history=None):
    """Gemini text call with staggered fallback. Starts the main model; if it fails, the next model starts at once,
    and if it is merely slow, the next model starts after HEDGE_DELAY_SECONDS. The first answer wins."""
    config = types.GenerateContentConfig(system_instruction=system_instruction) if system_instruction else None
    contents = list(history or []) + [types.Content(role="user", parts=[types.Part(text=prompt)])]
    models = [MODEL_NAME] + FALLBACK_MODELS
    results = queue.Queue()  # (model, text, error, seconds)

    def attempt(model):
        t0 = time.time()
        try:
            text = client.models.generate_content(model=model, contents=contents, config=config).text
            results.put((model, text, None, time.time() - t0))
        except Exception as e:
            results.put((model, None, e, time.time() - t0))

    start = time.time()
    started = 0            # models launched so far
    running = 0            # launched and not finished yet
    next_launch = start    # earliest time to launch the next model
    last_error = None

    while True:
        now = time.time()
        if started < len(models) and (running == 0 or now >= next_launch):
            threading.Thread(target=attempt, args=(models[started],), daemon=True).start()
            started += 1
            running += 1
            next_launch = now + HEDGE_DELAY_SECONDS
        if running == 0:
            raise last_error  # every model was tried and failed
        budget_left = GEMINI_BUDGET_SECONDS - (now - start)
        if budget_left <= 0:
            raise last_error or TimeoutError("Gemini did not answer in time")
        wait = budget_left
        if started < len(models):
            wait = min(wait, max(next_launch - now, 0.01))
        try:
            model, text, error, took = results.get(timeout=wait)
        except queue.Empty:
            continue

        running -= 1
        if error is None and text is not None:
            print("Gemini call (%s) took %.2fs (%.2fs since the request started)" % (model, took, time.time() - start))
            return text

        if error is None:
            error = RuntimeError("Gemini returned no text (blocked or empty response)")
            code = "empty"
        elif isinstance(error, errors.APIError):
            code = error.code
            if code not in FALLBACK_ERROR_CODES:
                raise error
        elif isinstance(error, httpx.TransportError):
            code = type(error).__name__
        else:
            raise error
        print("Gemini %s failed after %.2fs (%s)" % (model, took, code))
        last_error = error
        next_launch = time.time()  # a failure starts the next model right away

def chat(conversation_id, user_text):
    """Yuno's reply to user_text, remembering this visit's earlier turns. No id means no memory."""
    if conversation_id is None:
        return generate(user_text, SYSTEM_PROMPT)
    with history_lock:
        history = list(histories.get(conversation_id, []))
    answer = generate(user_text, SYSTEM_PROMPT, history)
    with history_lock:
        turns = histories.setdefault(conversation_id, [])
        turns.append(types.Content(role="user", parts=[types.Part(text=user_text)]))
        turns.append(types.Content(role="model", parts=[types.Part(text=answer)]))
        del turns[:-2 * HISTORY_MAX_TURNS]
        histories.move_to_end(conversation_id)
        while len(histories) > MAX_CONVERSATIONS:
            histories.popitem(last=False)
    return answer

app = Flask(__name__)

# Whisper settings. Defaults are the old CPU setup; to use the GPU put WHISPER_DEVICE=cuda in .env
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "float16" if WHISPER_DEVICE == "cuda" else "int8")

whisper_model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
list(whisper_model.transcribe(np.zeros(16000, dtype=np.float32))[0])  # warm-up so the first real request isn't slow
print("Whisper ready:", WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE)

FACE_DB_PATH = "face_database"

def transcribe_audio(filepath):
    t0 = time.time()
    segments, info = whisper_model.transcribe(filepath)
    text = " ".join([segment.text for segment in segments])
    print("Whisper took %.2fs" % (time.time() - t0))
    print("Detected language:", info.language, "probability:", info.language_probability)

    if info.language_probability < 0.60:
        print("Low confidence transcription, discarding:", text)
        return ""

    return text.strip()

@app.route("/respond", methods=["POST"])
def respond():
    data = request.get_json()
    audio_filename = data.get("audio_filename", "")

    transcribed_text = transcribe_audio(audio_filename)
    print("Transcribed:", transcribed_text)

    if not transcribed_text:
        print("Empty transcript, asking the person to repeat.")
        return jsonify({"answer": "Sorry, I didn't quite catch that. Could you say that again?", "heard": ""})

    answer = chat(data.get("conversation_id"), transcribed_text)

    return jsonify({"answer": answer, "heard": transcribed_text})

@app.route("/start_conversation", methods=["POST"])
def start_conversation():
    data = request.get_json()
    person_name = data.get("name", "")

    if person_name:
        prompt = (
            f"You are YeRo, a robot having a spoken conversation with someone in person. "
            f"You just recognized them — their name is {person_name}, and you've talked before. "
            "Greet them warmly by name and ask a short, natural opening question to get them "
            "talking — e.g. what they're up to, how their day is going, or what they want to "
            "chat about. Keep it to one or two short sentences, like natural speech, not written text."
        )
    else:
        prompt = (
            "You are YeRo, a robot having a spoken conversation with someone in person. "
            "Greet them and ask a short, natural opening question to get them talking. "
            "Keep it to one or two short sentences, like natural speech, not written text."
        )

    opener = chat(data.get("conversation_id"), prompt)

    return jsonify({"opener": opener})

@app.route("/extract_name", methods=["POST"])
def extract_name_route():
    data = request.get_json()
    audio_filename = data.get("audio_filename")
    transcribed_text = transcribe_audio(audio_filename)
    print("Transcribed for name extraction:", transcribed_text)
    name = extract_name(transcribed_text)
    return jsonify({"name": name})

def pick_person(results):
    """Majority vote over the per-photo matches. Ties go to the lower average distance. Returns (name, average distance)."""
    votes = {}
    for name, distance in results:
        if name is not None:
            votes.setdefault(name, []).append(distance)
    if not votes:
        return None, None
    winner = min(votes, key=lambda n: (-len(votes[n]), sum(votes[n]) / len(votes[n])))
    return winner, sum(votes[winner]) / len(votes[winner])

@app.route("/capture_recognize", methods=["POST"])
def capture_recognize():
    data = request.get_json()
    image_paths = data.get("image_paths", [])
    valid_paths = [p for p in image_paths if p and os.path.exists(p)]

    if not valid_paths:
        return jsonify({"face_id": None, "error": "No valid images"}), 400

    results = []
    for path in valid_paths:
        name, distance = find_person(path)
        print("Recognition for", path, "->", name, distance)
        results.append((name, distance))

    best_name, best_distance = pick_person(results)
    print("Recognition vote:", best_name, best_distance)

    return jsonify({"face_id": best_name, "distance": best_distance})

@app.route("/learn_face", methods=["POST"])
def learn_face():
    data = request.get_json()
    image_paths = data.get("image_paths", [])
    name = data.get("name")

    valid_paths = [p for p in image_paths if p and os.path.exists(p)]

    if not valid_paths:
        return jsonify({"success": False, "error": "No valid images"}), 400
    if not name:
        return jsonify({"success": False, "error": "No name provided"}), 400

    saved_paths = save_new_person(valid_paths, name)
    print("Learned new face:", name, "saved", len(saved_paths), "photos:", saved_paths)

    return jsonify({"success": True, "saved_paths": saved_paths})

def save_new_person(image_paths, name, db_path="face_database"):
    person_folder = os.path.join(db_path, name)
    os.makedirs(person_folder, exist_ok=True)

    existing = [f for f in os.listdir(person_folder) if f.startswith(name + "_") and f.endswith(".jpg")]
    start_index = len(existing) + 1

    saved_paths = []
    for i, image_path in enumerate(image_paths, start=start_index):
        dest = os.path.join(person_folder, name + "_" + str(i) + ".jpg")
        shutil.copy(image_path, dest)
        saved_paths.append(dest)

    return saved_paths

def find_person(image_path, db_path=FACE_DB_PATH):
    try:
        results = DeepFace.find(img_path=image_path, db_path=db_path, enforce_detection=False)
        if len(results) > 0 and len(results[0]) > 0:
            top = results[0].iloc[0]
            matched_path = top["identity"]
            distance = top["distance"]
            name = matched_path.split(os.sep)[-2]
            return name, distance
    except Exception as e:
        print("Recognition error:", e)
    return None, None

def extract_name(transcribed_text):
    if not transcribed_text.strip():
        return None
    """Uses the LLM to pull just a name out of a full sentence, or None if no name was said."""
    prompt = (
        "Extract only the person's first name from this sentence. "
        "Reply with just the name, nothing else. "
        "If no name is present, reply with exactly: NONE\n\n"
        f"Sentence: {transcribed_text}"
    )
    result = generate(prompt).strip()
    return None if result.upper() == "NONE" else result

YES_WORDS = {"yes", "yeah", "yep", "yup", "yea", "sure", "correct", "absolutely", "definitely", "affirmative"}
NO_WORDS = {"no", "nope", "nah", "not", "wrong", "incorrect", "negative", "never"}
YES_NO_MAX_WORDS = 6

def quick_yes_no(text):
    """True/False when a short reply is clearly yes or clearly no, else None (caller asks the LLM)."""
    words = re.findall(r"[a-z']+", text.lower())
    if not words or len(words) > YES_NO_MAX_WORDS:
        return None
    yes = any(w in YES_WORDS for w in words)
    no = any(w in NO_WORDS for w in words)
    if yes and not no:
        return True
    if no and not yes:
        return False
    return None

def extract_yes_no(transcribed_text):
    if not transcribed_text.strip():
        return None
    quick = quick_yes_no(transcribed_text)
    if quick is not None:
        print("Yes/no decided locally:", quick)
        return quick
    prompt = (
        "Does this sentence mean yes or no? Reply with exactly one word: YES, NO, or UNCLEAR.\n\n"
        f"Sentence: {transcribed_text}"
    )
    result = generate(prompt).strip().upper()
    if result == "YES":
        return True
    elif result == "NO":
        return False
    return None


@app.route("/confirm_yes_no", methods=["POST"])
def confirm_yes_no_route():
    data = request.get_json()
    audio_filename = data.get("audio_filename")

    t0 = time.time()
    transcribed_text = transcribe_audio(audio_filename)
    t1 = time.time()
    print("Transcribed for yes/no (%.2fs):" % (t1 - t0), transcribed_text)

    result = extract_yes_no(transcribed_text)
    print("LLM yes/no call took %.2fs" % (time.time() - t1))

    return jsonify({"confirmed": result})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)