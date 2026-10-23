# Blocking HTTP calls from the Pepper client (Python 2.7) to the Flask server.
import json
import urllib2
import time

from config import SERVER_URL, TIMEOUT_DEFAULT, TIMEOUT_LLM, TIMEOUT_OPENER, OPENER_FALLBACK
from logger import log


def _post_json(path, payload, timeout):
    req = urllib2.Request(SERVER_URL + path, json.dumps(payload),
                          {'Content-Type': 'application/json'})
    response = urllib2.urlopen(req, timeout=timeout)
    return json.loads(response.read())


def _to_str(value):
    # NAOqi's say() needs a Python 2 byte str, json gives unicode.
    if isinstance(value, unicode):
        return value.encode('utf-8')
    return value


# Conversation ids let the server keep a separate chat history per engagement. The start time keeps
# ids unique if this client is restarted while the server keeps running.
_RUN_ID = int(time.time())

def conversation_id(engagement_id):
    if engagement_id is None:
        return None
    return "%d-%d" % (_RUN_ID, engagement_id)

# /respond: audio filename -> LLM answer. Does NOT catch errors; the caller
# (_process_and_respond) handles failures and speaks the fallback line.
def get_llm_response(filename, engagement_id=None):
    payload = {"audio_filename": filename, "conversation_id": conversation_id(engagement_id)}
    result = _post_json("/respond", payload, TIMEOUT_LLM)
    return result["answer"].encode('utf-8')


# /capture_recognize: list of photo paths -> best matching name, or None.
def check_recognition(filepaths):
    if not filepaths:
        return None
    try:
        result = _post_json("/capture_recognize", {"image_paths": filepaths}, TIMEOUT_DEFAULT)
        return _to_str(result.get("face_id"))
    except Exception as e:
        log("Recognition call failed:", e)
        return None


# /extract_name: audio filename -> first name, or None.
def extract_name(filename):
    try:
        result = _post_json("/extract_name", {"audio_filename": filename}, TIMEOUT_LLM)
        return _to_str(result.get("name"))
    except Exception as e:
        log("Name extraction call failed:", e)
        return None


# /learn_face: save photos under a name. Returns True/False.
def learn_face(photo_paths, name):
    try:
        result = _post_json("/learn_face", {"image_paths": photo_paths, "name": name}, TIMEOUT_DEFAULT)
        return result.get("success", False)
    except Exception as e:
        log("Learn face call failed:", e)
        return False


# /confirm_yes_no: audio filename -> True, False or None.
def extract_yes_no(filename):
    try:
        result = _post_json("/confirm_yes_no", {"audio_filename": filename}, TIMEOUT_LLM)
        return result.get("confirmed")  # True, False, or None
    except Exception as e:
        log("Yes/no confirmation call failed:", e)
        return None


# /start_conversation: name -> spoken opener. Falls back to a generic line.
# (The LED start/stop and the "took" timing log stay in the caller.)
def get_conversation_opener(name, engagement_id=None):
    try:
        payload = {"name": name or "", "conversation_id": conversation_id(engagement_id)}
        result = _post_json("/start_conversation", payload, TIMEOUT_OPENER)
        opener = result.get("opener") or OPENER_FALLBACK
        return _to_str(opener)
    except Exception as e:
        log("start_conversation call failed:", e)
        return OPENER_FALLBACK
