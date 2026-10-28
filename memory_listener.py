from __future__ import print_function
from naoqi import ALProxy, ALBroker, ALModule
import time
import threading
import random

import server_client
from leds import LedsMixin
from camera import CameraMixin
from audio import AudioMixin
from identity import IdentityMixin
import config
from config import ROBOT_IP, ROBOT_PORT
from logger import log

myBroker = ALBroker("myBroker", "0.0.0.0", 0, ROBOT_IP, ROBOT_PORT)

class MemoryListenerModule(LedsMixin, CameraMixin, AudioMixin, IdentityMixin, ALModule):
    # Sets up proxies, default (nobody engaged) state, awareness config and event subscriptions.
    def __init__(self, name):
        ALModule.__init__(self, name)

        # Core Modules:
        self.memory = ALProxy("ALMemory")
        self.motion = ALProxy("ALMotion")
        self.posture = ALProxy("ALRobotPosture")

        # Speech:
        self.animated_speech = ALProxy("ALAnimatedSpeech")
        self.tts = ALProxy("ALTextToSpeech")
        self.speech_config = dict(config.SPEECH_CONFIG)
        self.speech_config_still = dict(config.SPEECH_CONFIG_STILL)

        # Awareness / Engagement:
        self.awareness = ALProxy("ALBasicAwareness")
        self.sound_localization = ALProxy("ALSoundLocalization")
        # self.people_perception = ALProxy("ALPeoplePerception")
        # self.people_perception.setTimeBeforeVisiblePersonDisappears(3)
        self.engagement_session_id = 0
        self.human_engaged = False
        self.lock = threading.Lock()
        self.last_response_time = 0
        self.cooldown_seconds = config.COOLDOWN_SECONDS
        self.human_lost_grace = config.HUMAN_LOST_GRACE_SECONDS
        self._lost_pending_id = None  # engagement id whose HumanLost grace countdown is running, else None

        # Audio Recording:
        self.audio_service = ALProxy("ALAudioDevice")
        self.recorder = ALProxy("ALAudioRecorder")
        self.listening_movement = ALProxy("ALListeningMovement")
        self.audio_frames = []
        self.is_recording = False
        self.recording_timer = None
        self.recording_duration = config.RECORDING_DURATION
        self.recording_session_id = 0

        # VAD:
        self.vad_poll_interval = config.VAD_POLL_INTERVAL
        self.vad_window_seconds = config.VAD_WINDOW_SECONDS
        self.vad_silence_duration = config.VAD_SILENCE_DURATION
        self.vad_min_recording_seconds = config.VAD_MIN_RECORDING_SECONDS
        self.vad_threshold = config.VAD_THRESHOLD
        self._vad_timer = None
        self._vad_has_spoken = False
        self._vad_silence_start = None
        self._recording_start_time = None

        # Face Capture:
        self.video_device = ALProxy("ALVideoDevice")
        self.pending_photo_paths = None

        # Identity / Confirmation:
        self.pending_face_id = None
        self.awaiting_confirmation = None  # None | "name" | "identity"
        self.candidate_name = None
        self.candidate_face_id = None
        self.identity_confirm_attempts = 0

        # LEDs:
        self.leds = ALProxy("ALLeds")
        self.has_explained_leds = False
        self._processing_active = False
        self._processing_thread = None

        # Wake Motors (needs the proxies above):
        self.motion.wakeUp()

        # Basic Awareness configuration:
        self.awareness.setEngagementMode("FullyEngaged")
        self.awareness.setTrackingMode("Head")
        self.awareness.setStimulusDetectionEnabled("Sound", True)
        self._log("Sound stimulus detection enabled:", self.awareness.isStimulusDetectionEnabled("Sound"))

        self.sound_localization.setParameter("Sensitivity", config.SOUND_LOCALIZATION_SENSITIVITY)
        self.sound_localization.setParameter("EnergyComputation", True)
        self.sound_localization.subscribe("MemoryListenerSoundDebug")
        self.awareness.setEnabled(True)

        # Event subscriptions (must stay after awareness is enabled):
        self.memory.subscribeToEvent("ALBasicAwareness/HumanTracked", "MemoryListener", "onHumanTracked")
        self.memory.subscribeToEvent("ALBasicAwareness/HumanLost", "MemoryListener", "onHumanLost")
        self.memory.subscribeToEvent("ALSoundLocalization/SoundLocated", "MemoryListener", "onSoundLocated")
        self.memory.subscribeToEvent("ALBasicAwareness/StimulusDetected", "MemoryListener", "onStimulusDetected")

        # Setup completed: white eyes = idle
        self.leds.fadeRGB("FaceLeds", 1.0, 1.0, 1.0, 0.3)


    # True if the person left or a newer engagement replaced engagement_id (so slow steps should bail).
    def _is_stale(self, engagement_id):
        return not self.human_engaged or engagement_id != self.engagement_session_id

    # Wrapper around logger.log()
    def _log(self, *args):
        log(*args)

    # Unused cooldown gate (True once per cooldown_seconds). Kept in case it's needed again.
    def _cooldown_elapsed(self):
        with self.lock:
            now = time.time()
            if now - self.last_response_time > self.cooldown_seconds:
                self.last_response_time = now
                return True
        return False
    
    
    # INTERACTION LOGIC:

    # HumanTracked callback (new thread per event). Under self.lock, claims the engagement and mints
    # a new engagement id; slow steps compare against it to detect being superseded.
    # Then resets per-person state, captures + recognizes, and greets.
    def onHumanTracked(self, key, value, message):
        if value is None or value < 0:
            self._log("Ignoring invalid HumanTracked event, ID:", value)
            return

        with self.lock:
            if self.human_engaged:
                if self._lost_pending_id is not None:
                    self._log("Human re-tracked within the grace period, staying engaged.")
                    self._lost_pending_id = None
                return
            self.human_engaged = True
            self.engagement_session_id += 1
            engagement_id = self.engagement_session_id

        self._log("Human tracked - engaging. ID:", value, "session:", engagement_id)

        self.awaiting_confirmation = None
        self.candidate_name = None
        self.identity_confirm_attempts = 0
        self.pending_face_id = None

        if config.GUEST_MODE:
            self._greet_guest(engagement_id)
            return

        if config.RECOGNIZE_WHILE_GREETING:
            self._greet_and_recognize(engagement_id)
            return

        self.leds.fadeRGB("FaceLeds", 0.0, 1.0, 0.0, 0.3)
        speech_id = self.animated_speech.post.say("One moment, let me get a good look at you!", self.speech_config_still)

        self._capture_and_recognize(engagement_id, speech_id)

        if self._is_stale(engagement_id):
            self._log("Engagement superseded or ended during capture, aborting.")
            return

        self._respond_to_engagement(engagement_id)

    # HumanLost callback. Doesn't disengage right away: starts a short countdown (human_lost_grace) so a
    # brief tracking flicker doesn't end the conversation. onHumanTracked cancels it if the person returns.
    def onHumanLost(self, key, value, message):
        with self.lock:
            if not self.human_engaged:
                self._log("Human lost (ID:", value, ") but nobody engaged, ignoring.")
                return
            if self._lost_pending_id is not None:
                return
            engagement_id = self.engagement_session_id
            self._lost_pending_id = engagement_id

        self._log("Human lost (ID:", value, ") - disengaging in", self.human_lost_grace, "s unless they come back.")
        timer = threading.Timer(self.human_lost_grace, self._disengage, args=[engagement_id])
        timer.daemon = True
        timer.start()

    # Runs when the grace period ends. Does nothing if the person came back or a newer engagement exists.
    # Otherwise clears engagement and confirmation state so in-flight calls see human_engaged == False,
    # resets LEDs/posture, and force-finishes any active recording.
    def _disengage(self, engagement_id):
        with self.lock:
            if self._lost_pending_id != engagement_id or not self.human_engaged or engagement_id != self.engagement_session_id:
                return
            self._lost_pending_id = None
            self.human_engaged = False

        self._log("Human gone for good - disengaging. session:", engagement_id)

        self.awaiting_confirmation = None
        self.candidate_name = None
        self.candidate_face_id = None
        self.identity_confirm_attempts = 0

        self.leds.fadeRGB("FaceLeds", 1.0, 0.0, 0.0, 0.3)
        self.posture.goToPosture("Stand", 0.5)
        self.leds.fadeRGB("FaceLeds", 1.0, 1.0, 1.0, 0.3)

        self._log("is_recording at disengage time:", self.is_recording)
        if self.is_recording:
            self._log("Attempting early recording cutoff.")
            if self.recording_timer is not None:
                self.recording_timer.cancel()
                self._log("Timer cancel() called.")
            if self._vad_timer is not None:
                self._vad_timer.cancel()
            self._finish_recording(self.recording_session_id, self.engagement_session_id)
            self._log("Forced _finish_recording call completed.")

    # Dispatcher for a finished recording: routes to a pending confirmation handler, else treats it as a
    # spoken name if the person is unidentified, else gets an LLM reply. Every path ends in _continue_or_idle.
    def _process_and_respond(self, filename, engagement_id):
        if self._is_stale(engagement_id):
            self._log("Engagement ended before processing, skipping.")
            return

        if self.awaiting_confirmation == "identity":
            self._handle_identity_confirmation(filename, engagement_id)
            return

        if self.awaiting_confirmation == "name":
            self._handle_name_confirmation(filename, engagement_id)
            return

        if not self.pending_face_id:
            self._start_processing_leds()
            t0 = time.time()
            name = server_client.extract_name(filename)
            self._log("Name extraction took: %.2fs" % (time.time() - t0))
            self._stop_processing_leds()

            if self._is_stale(engagement_id):
                self._log("Human left before name response ready, staying silent.")
                return

            if name:
                self.candidate_name = name
                self.awaiting_confirmation = "name"
                self.animated_speech.say("Did you say " + name + "?", self.speech_config)
            else:
                self.animated_speech.say("Sorry, I didn't catch your name. What's your name?", self.speech_config)
            self._continue_or_idle(engagement_id)
            return

        self._start_processing_leds()
        t0 = time.time()
        try:
            answer = self._get_answer_with_fillers(filename, engagement_id)
        except Exception as e:
            self._stop_processing_leds()
            self._log("LLM/STT call failed:", e)
            if not self._is_stale(engagement_id):
                self.animated_speech.say(config.LLM_FALLBACK_LINE, self.speech_config)
            self._continue_or_idle(engagement_id)
            return
        self._log("LLM response call took: %.2fs" % (time.time() - t0))
        self._stop_processing_leds()

        if self._is_stale(engagement_id):
            self._log("Human left before LLM response ready, staying silent.")
            return

        self._log("LLM response:", answer)
        self.leds.fadeRGB("FaceLeds", 0.0, 1.0, 0.0, 0.3)
        self._speak_reply(answer, engagement_id)
        self._continue_or_idle(engagement_id)

    # Asks the server for the reply on a worker thread. If it isn't back after FILLER_AFTER_SECONDS, says a short
    # varied filler line (never the same one twice in a row), at most MAX_FILLERS_PER_REPLY per reply.
    def _get_answer_with_fillers(self, filename, engagement_id):
        name = self._known_name()
        box = {}

        def work():
            try:
                box["answer"] = server_client.get_llm_response(filename, engagement_id, name)
            except Exception as e:
                box["error"] = e

        worker = threading.Thread(target=work)
        worker.daemon = True
        worker.start()

        unused = list(config.FILLER_LINES)
        random.shuffle(unused)
        last = getattr(self, "_last_filler", None)
        if len(unused) > 1 and unused[-1] == last:
            unused[0], unused[-1] = unused[-1], unused[0]
        fillers_said = 0
        wait = config.FILLER_AFTER_SECONDS
        while True:
            worker.join(wait)
            if not worker.is_alive():
                break
            if self._is_stale(engagement_id) or fillers_said >= config.MAX_FILLERS_PER_REPLY or not unused:
                worker.join()  # nothing more to say, just wait for the reply
                break
            line = unused.pop()
            self.animated_speech.say(line, self.speech_config_still)
            self._last_filler = line
            fillers_said += 1
            wait = config.FILLER_REPEAT_SECONDS
        if "error" in box:
            raise box["error"]
        return box["answer"]

    # Keeps listening (blue eyes + _start_listening) if the same person is still engaged, else idles (white eyes).
    def _continue_or_idle(self, engagement_id):  # NEW: added engagement_id param (was def _continue_or_idle(self):)
        if engagement_id != self.engagement_session_id:
            return
        if self.human_engaged:
            self.leds.fadeRGB("FaceLeds", 0.0, 0.0, 1.0, 0.3)
            self._start_listening(engagement_id)
        else:
            self.leds.fadeRGB("FaceLeds", 1.0, 1.0, 1.0, 0.3)

    # NAOqi audio callback: buffers incoming audio while recording.
    def processRemote(self, nbOfChannels, nbOfSamplesByChannel, timeStamp, inputBuffer):
        if self.is_recording:
            self.audio_frames.append(inputBuffer)


MemoryListener = MemoryListenerModule("MemoryListener")

print("Subscribed and listening...")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("KeyboardInterrupt: Stopping...")
    MemoryListener.sound_localization.unsubscribe("MemoryListenerSoundDebug")  # NEW
    MemoryListener.awareness.setEnabled(False)
    MemoryListener.motion.rest()
    myBroker.shutdown()
    print("Cleaned up. Exiting.")