# Shared constants for the Yuno / MemoryListener client (Python 2.7).
# Pure data, no imports, no side effects.

ROBOT_IP = "10.201.168.249"
ROBOT_PORT = 9559

SERVER_URL = "http://127.0.0.1:5000"

# Seconds. Same values the inline urlopen calls used before the split.
TIMEOUT_DEFAULT = 30
TIMEOUT_OPENER = 15
TIMEOUT_LLM = 20  # /respond, /extract_name, /confirm_yes_no (Whisper + Gemini)

OPENER_FALLBACK = "Good to see you again! What's going on?"

LLM_FALLBACK_LINE = "Sorry, I lost my train of thought. Could you say that again?"  # spoken when /respond fails or times out

# --- Guest mode (open house) ---
# True: no photos, no recognition, no name questions. Greets with the generic opener and just talks.
GUEST_MODE = False
GUEST_NAME = "guest"  # placeholder id so replies go to the LLM branch

# --- Recognize while greeting ---
# True (with GUEST_MODE = False): greet right away while a background thread recognizes the face. Once the robot
# has answered a turn it asks "are you <name>?" Ignored when GUEST_MODE is True.
RECOGNIZE_WHILE_GREETING = True

# --- Speech ---
SPEECH_CONFIG = {"bodyLanguageMode": "contextual"}
SPEECH_CONFIG_STILL = {"bodyLanguageMode": "disabled"}

# --- Interaction ---
COOLDOWN_SECONDS = 5
COOLDOWN_SECONDS = 5
HUMAN_LOST_GRACE_SECONDS = 1.5  # a lost person must stay lost this long before the robot disengages (rides out tracking flicker)

# --- Awareness / sound localization ---
SOUND_LOCALIZATION_SENSITIVITY = 0.9

# --- Recording ---
RECORDING_DURATION = 60  # TEMP (hard-cap timer, seconds)

# --- VAD (energy based, see _vad_check) ---
VAD_POLL_INTERVAL = 0.25         # seconds between trailing-window checks
VAD_WINDOW_SECONDS = 0.5         # how much trailing audio each check analyzes
VAD_SILENCE_DURATION = 1.0      # trailing silence after speech that ends the turn
VAD_MIN_RECORDING_SECONDS = 1.0  # floor before any early cutoff is honored
VAD_THRESHOLD = 700              # RMS bar on the front mic channel

# --- Audio capture format (ALAudioDevice, 4 mics interleaved) ---
AUDIO_SAMPLE_RATE = 48000
AUDIO_CHANNELS = 4
AUDIO_BYTES_PER_SAMPLE = 2   # 16-bit
AUDIO_FRONT_CHANNEL = 2      # index of the front mic channel used for RMS/VAD

# --- filler line settings ---
FILLER_LINES = ["Hmm, let me think.", "Let me think.", "Hmm, let me see.", "One sec.", "Let me see.", "Hmm."]
FILLER_AFTER_SECONDS = 1.5     # say a filler only if the answer isn't back by then
FILLER_REPEAT_SECONDS = 10.0     # still waiting: next filler after this long
MAX_FILLERS_PER_REPLY = 2