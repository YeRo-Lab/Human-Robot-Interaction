# Identity flow (mixin): photo capture + recognition, greeting, name/identity confirmation, opener.
# Uses state from MemoryListenerModule.__init__ and the other mixins.
import re
import threading
import time

import config
import server_client

def _drop_trailing_questions(text):
    """Removes question sentences from the end of text, e.g. "Nice! Want to chat? Anything fun?" -> "Nice!"."""
    protected = re.sub(r'\b(Dr|Mr|Mrs|Ms|Prof|St)\.', r'\1<dot>', text.strip())  # "Dr." must not end a sentence
    sentences = re.split(r'(?<=[.!?])\s+', protected)
    while sentences and sentences[-1].endswith("?"):
        sentences.pop()
    return " ".join(sentences).replace("<dot>", ".")

class IdentityMixin(object):

    # Captures photos and asks the server who it is, setting pending_photo_paths and candidate_face_id.
    # Discards its result if a newer engagement superseded engagement_id during the slow steps.
    def _capture_and_recognize(self, engagement_id, speech_id=None):  # NEW: added engagement_id param (was no params)
        self._start_processing_leds(speech_id)
        t0 = time.time()
        photo_paths = self._capture_face_photo(engagement_id)
        t1 = time.time()
        self._log("Capture phase took: %.2fs" % (t1 - t0))

        if self._is_stale(engagement_id):
            self._stop_processing_leds()
            self._log("Stale capture (session", engagement_id, "superseded), discarding.")  # NEW
            return

        candidate = server_client.check_recognition(photo_paths) if photo_paths else None
        t2 = time.time()
        self._log("Recognition phase took: %.2fs" % (t2 - t1))
        self._log("Recognition result:", candidate)

        self._stop_processing_leds()

        if self._is_stale(engagement_id):
            self._log("Stale recognition (session", engagement_id, "superseded), discarding.")  # NEW
            return

        self.pending_photo_paths = photo_paths
        self.candidate_face_id = candidate


    # Greeting: "Are you X?" (awaiting_confirmation = "identity") if recognized, else asks for a name.
    # Then starts listening.
    def _respond_to_engagement(self, engagement_id):  # NEW: added engagement_id param (was no params)
        self._log("Responding to engagement: starting.")
        self.leds.fadeRGB("FaceLeds", 0.0, 1.0, 0.0, 0.3)

        if self.candidate_face_id:
            self.awaiting_confirmation = "identity"
            self.animated_speech.say("Are you " + self.candidate_face_id + "?", self.speech_config)
        else:
            self.awaiting_confirmation = None
            self.animated_speech.say("Hi, I don't think we've met. What's your name?", self.speech_config)

        self._log("Responding to engagement - done.")

        if self._is_stale(engagement_id):  # NEW: was just "if not self.human_engaged:"
            self.leds.fadeRGB("FaceLeds", 1.0, 1.0, 1.0, 0.3)
            self._log("Human left during greeting, skipping listening.")
            return

        self.leds.fadeRGB("FaceLeds", 0.0, 0.0, 1.0, 0.3)
        self._start_listening(engagement_id)


    # Guest mode (config.GUEST_MODE): no photos or recognition. Speaks the generic opener, then listens.
    def _greet_guest(self, engagement_id):
        self.pending_face_id = config.GUEST_NAME  # any non-empty id sends replies to the LLM, not the name flow
        self.leds.fadeRGB("FaceLeds", 0.0, 1.0, 0.0, 0.3)
        opener = self._get_conversation_opener("", engagement_id)

        if self._is_stale(engagement_id):
            self._log("Human left before opener was ready, staying silent.")
            return

        self.animated_speech.say(opener, self.speech_config)
        self._continue_or_idle(engagement_id)


    # RECOGNIZE_WHILE_GREETING: greets with the generic opener right away while a background thread captures and
    # recognizes. Body language stays off during the greeting so the camera frames stay sharp. The result waits in
    # candidate_face_id until _ask_pending_identity confirms it after a turn.
    def _greet_and_recognize(self, engagement_id):
        self.pending_face_id = config.GUEST_NAME  # replies go to the LLM until the identity is confirmed
        self.candidate_face_id = None
        self.pending_photo_paths = None
        self.leds.fadeRGB("FaceLeds", 0.0, 1.0, 0.0, 0.3)

        worker = threading.Thread(target=self._recognize_in_background, args=(engagement_id,))
        worker.daemon = True
        worker.start()

        opener = self._get_conversation_opener("", engagement_id)

        if self._is_stale(engagement_id):
            self._log("Human left before opener was ready, staying silent.")
            return

        self.animated_speech.say(opener, self.speech_config_still)
        self._continue_or_idle(engagement_id)


    # Capture + recognition without touching the LEDs or speech. Stores the result only if this engagement is still current.
    def _recognize_in_background(self, engagement_id):
        t0 = time.time()
        photo_paths = self._capture_face_photo(engagement_id)

        if self._is_stale(engagement_id):
            self._log("Background capture: engagement ended, discarding.")
            return

        candidate = server_client.check_recognition(photo_paths) if photo_paths else None
        self._log("Background recognition took: %.2fs, result:" % (time.time() - t0), candidate)

        with self.lock:
            if self._is_stale(engagement_id):
                self._log("Background recognition: engagement ended, discarding result.")
                return
            self.pending_photo_paths = photo_paths
            self.candidate_face_id = candidate

    # The confirmed name of the person, or None while they are still a guest / unconfirmed.
    def _known_name(self):
        name = self.pending_face_id
        return name if name and name != config.GUEST_NAME else None

    # Speaks the LLM reply. If background recognition found someone who isn't confirmed yet, it also asks
    # "are you <name>?" afterwards. The reply's own trailing question is dropped first, so the person never
    # has two questions to answer and a "yes"/"no" can't be taken for the wrong one.
    def _speak_reply(self, answer, engagement_id):
        name = self.candidate_face_id
        ask = bool(config.RECOGNIZE_WHILE_GREETING and name and not self._is_stale(engagement_id)
                   and self.awaiting_confirmation is None and self.pending_face_id == config.GUEST_NAME)
        if ask:
            answer = _drop_trailing_questions(answer)
        if answer:
            self.animated_speech.say(answer, self.speech_config)
        if ask and not self._is_stale(engagement_id):
            self.awaiting_confirmation = "identity"
            lead = "Just to make sure, are you " if answer else "Before we go on, are you "
            self.animated_speech.say(lead + name + "?", self.speech_config)


    # Handles the reply to "Did you say <name>?". Yes: save photos under candidate_name via /learn_face and
    # set pending_face_id. No: ask again. Unclear: repeat the question.
    def _handle_name_confirmation(self, filename, engagement_id):
        self._start_processing_leds()
        t0 = time.time()
        confirmed = server_client.extract_yes_no(filename)
        self._log("Yes/no extraction took: %.2fs" % (time.time() - t0))
        self._log("Name confirmation result:", confirmed)
        self._stop_processing_leds()

        if self._is_stale(engagement_id):
            self._log("Human left before name confirmation could be handled, staying silent.")
            return

        if confirmed is True:
            self.awaiting_confirmation = None
            self._start_processing_leds()
            t0 = time.time()
            success = server_client.learn_face(self.pending_photo_paths, self.candidate_name) if self.pending_photo_paths else False
            self._log("Learn face call took: %.2fs" % (time.time() - t0))
            self._stop_processing_leds()

            if self._is_stale(engagement_id):
                self._log("Human left during learn_face call, staying silent.")
                self.candidate_name = None
                return

            if success:
                self.pending_face_id = self.candidate_name
                self.animated_speech.say("Nice to meet you, " + self.candidate_name + "!", self.speech_config)
            else:
                self.animated_speech.say("Sorry, I had trouble remembering your face. Let's continue anyway.", self.speech_config)
            self.candidate_name = None

        elif confirmed is False:
            self.candidate_name = None
            self.awaiting_confirmation = None
            self.animated_speech.say("Sorry about that, what's your name?", self.speech_config)

        else:
            self.animated_speech.say("Sorry, was that a yes or no?", self.speech_config)

        self._continue_or_idle(engagement_id)


    # Handles the reply to "Are you <name>?". Yes: adopt candidate_face_id, add photos via /learn_face, speak
    # the opener. No: re-capture once; after the second miss, ask for a name instead.
    def _handle_identity_confirmation(self, filename, engagement_id):  # NEW: added engagement_id param (was def _handle_identity_confirmation(self, filename):)
        self._start_processing_leds()
        t0 = time.time()
        confirmed = server_client.extract_yes_no(filename)
        self._log("Yes/no extraction took: %.2fs" % (time.time() - t0))
        self._stop_processing_leds()
        self._log("Identity confirmation result:", confirmed)

        if self._is_stale(engagement_id):
            self._log("Human left before identity confirmation could be handled, staying silent.")
            return

        if confirmed is True:
            self.awaiting_confirmation = None
            self.pending_face_id = self.candidate_face_id

            if self.pending_photo_paths:
                self._start_processing_leds()
                t0 = time.time()
                server_client.learn_face(self.pending_photo_paths, self.pending_face_id)
                self._log("Learn face call took: %.2fs" % (time.time() - t0))
                self._stop_processing_leds()

                if self._is_stale(engagement_id):
                    self._log("Human left during learn_face call, staying silent.")
                    self.candidate_face_id = None
                    return

            if config.RECOGNIZE_WHILE_GREETING:
                opener = "Good to see you again, " + self.pending_face_id + "!"  # mid-conversation, so no fresh greeting
            else:
                opener = self._get_conversation_opener(self.pending_face_id, engagement_id)

            if self._is_stale(engagement_id):
                self._log("Human left before opener was ready, staying silent.")
                self.candidate_face_id = None
                return

            self.animated_speech.say(opener, self.speech_config)
            self.candidate_face_id = None

        elif confirmed is False:
            self.identity_confirm_attempts += 1
            self.candidate_face_id = None

            if self.identity_confirm_attempts >= 2:
                self.awaiting_confirmation = None
                self.pending_face_id = None
                self.animated_speech.say("Sorry about that. What's your name?", self.speech_config)
            else:
                self.animated_speech.say("Sorry, let me look again.", self.speech_config)
                self._capture_and_recognize(engagement_id)

                if self._is_stale(engagement_id):
                    self._log("Human left during re-capture, staying silent.")
                    return

                if self.candidate_face_id:
                    self.awaiting_confirmation = "identity"
                    self.animated_speech.say("Are you " + self.candidate_face_id + "?", self.speech_config)
                else:
                    self.awaiting_confirmation = None
                    self.pending_face_id = None
                    self.animated_speech.say("I couldn't tell. What's your name?", self.speech_config)

        else:
            self.animated_speech.say("Sorry, was that a yes or no?", self.speech_config)

        self._continue_or_idle(engagement_id)


    # Conversation opener for a recognized person. The HTTP call lives in
    # server_client.py; this wraps it in the thinking-LED feedback.
    def _get_conversation_opener(self, name, engagement_id):
        self._start_processing_leds()
        t0 = time.time()
        opener = server_client.get_conversation_opener(name, engagement_id)
        self._log("Conversation opener took: %.2fs" % (time.time() - t0))
        self._stop_processing_leds()
        return opener