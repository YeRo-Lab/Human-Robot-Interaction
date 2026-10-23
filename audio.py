# Audio capture, WAV saving and energy-based VAD turn-taking (mixin). Uses state set up in
# MemoryListenerModule.__init__. processRemote stays on that class because NAOqi calls it.
import time
import threading
import wave

import numpy as np

import config


class AudioMixin(object):

    # AUDIO PROCESSING:

    # Starts one listening turn: audio stream, hard-cap timer (recording_duration) and the VAD poll.
    def _start_listening(self, engagement_id):
        if self._is_stale(engagement_id):
            self._log("Not starting to listen: engagement ended.")
            return
        self.recording_session_id += 1
        recording_id = self.recording_session_id

        self.start_audio_stream()

        self._vad_has_spoken = False
        self._vad_silence_start = None
        self._recording_start_time = time.time()

        self.recording_timer = threading.Timer(
            self.recording_duration, self._finish_recording, args=[recording_id, engagement_id]
        )
        self.recording_timer.start()

        self._vad_timer = threading.Timer(
            self.vad_poll_interval, self._vad_check, args=[recording_id, engagement_id]
        )
        self._vad_timer.start()

    # Stops and saves the recording, then processes it (or skips if silent). Fired by the hard-cap timer, VAD
    # or onHumanLost; claims the recording under self.lock so only one caller runs it.
    def _finish_recording(self, recording_id, engagement_id):
        with self.lock:
            if recording_id != self.recording_session_id or not self.is_recording:
                return
            self.is_recording = False
            heard_speech = self._vad_has_spoken
        if self._vad_timer is not None:
            self._vad_timer.cancel()
        self.stop_audio_stream()
        filename = "audio/" + str(recording_id) + "captured_audio.wav"
        self.save_audio_to_wav(filename)
        self._log("Recording saved:", filename)

        if self._is_stale(engagement_id):
            self._log("Engagement ended, discarding recording.")
            return

        # if self._is_silent(filename):
        #     self._log("Silence detected, skipping STT/LLM.")
        #     self._continue_or_idle(engagement_id)
        #     return
        silent = self._is_silent(filename)
        if silent and heard_speech:
            self._log("Whole-clip RMS is low but the VAD heard speech, keeping the recording.")
        elif silent:
            self._log("Silence detected, skipping STT/LLM.")
            self._continue_or_idle(engagement_id)
            return

        self._process_and_respond(filename, engagement_id)

    # True if the saved WAV's front-channel RMS is below threshold.
    def _is_silent(self, filename, threshold=config.VAD_THRESHOLD):
        wf = wave.open(filename, 'rb')
        n_channels = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())
        wf.close()

        audio = np.frombuffer(raw, dtype=np.int16)
        if len(audio) == 0:  # guard against empty buffer before computing anything
            self._log("Empty audio buffer, treating as silent.")
            return True

        if n_channels > 1:
            audio = audio.reshape(-1, n_channels)
            audio = audio[:, config.AUDIO_FRONT_CHANNEL]

        rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))
        self._log("Audio RMS level: %.1f" % rms)
        return rms < threshold
    
    # Every vad_poll_interval, measures RMS of the last vad_window_seconds. After speech has been heard,
    # vad_silence_duration of silence ends the turn early. No cutoff before vad_min_recording_seconds.
    # Reschedules itself while this recording is still the live one.
    def _vad_check(self, recording_id, engagement_id):
        if recording_id != self.recording_session_id or not self.is_recording:
            return

        elapsed = time.time() - self._recording_start_time
        if elapsed < self.vad_min_recording_seconds:
            self._vad_timer = threading.Timer(self.vad_poll_interval, self._vad_check, args=[recording_id, engagement_id])
            self._vad_timer.start()
            return

        raw = b"".join(self.audio_frames[:])  # snapshot before slicing, audio_frames is still being appended to

        bytes_per_second = config.AUDIO_SAMPLE_RATE * config.AUDIO_CHANNELS * config.AUDIO_BYTES_PER_SAMPLE
        window_bytes = int(self.vad_window_seconds * bytes_per_second)
        window_bytes -= window_bytes % (config.AUDIO_CHANNELS * config.AUDIO_BYTES_PER_SAMPLE)  # keep sample-aligned (4 channels * 2 bytes each)
        window_raw = raw[-window_bytes:]
        window_audio = np.frombuffer(window_raw, dtype=np.int16)

        if len(window_audio) == 0:
            rms = 0.0
        else:
            window_audio = window_audio.reshape(-1, config.AUDIO_CHANNELS)
            channel = window_audio[:, config.AUDIO_FRONT_CHANNEL]
            rms = np.sqrt(np.mean(channel.astype(np.float64) ** 2))

        now = time.time()

        if rms >= self.vad_threshold:
            if not self._vad_has_spoken:
                self._log("VAD: speech detected, RMS=%.1f" % rms)
            self._vad_has_spoken = True
            self._vad_silence_start = None
        elif self._vad_has_spoken:
            if self._vad_silence_start is None:
                self._vad_silence_start = now
                self._log("VAD: trailing silence started, RMS=%.1f" % rms)
            elif now - self._vad_silence_start >= self.vad_silence_duration:
                self._log("VAD: silence held %.2fs after speech, ending turn early." % (now - self._vad_silence_start))
                if self.recording_timer is not None:
                    self.recording_timer.cancel()
                self._finish_recording(recording_id, engagement_id)
                return

        self._vad_timer = threading.Timer(self.vad_poll_interval, self._vad_check, args=[recording_id, engagement_id])
        self._vad_timer.start()
    
    # Clears the buffer and subscribes to ALAudioDevice.
    def start_audio_stream(self):
        self.audio_frames = []
        self.audio_service.setClientPreferences(self.getName(), config.AUDIO_SAMPLE_RATE, 0, 0)
        self.audio_service.subscribe(self.getName())
        self.is_recording = True
        self._log("Recording started.")

    # Unsubscribes from ALAudioDevice.
    def stop_audio_stream(self):
        self.audio_service.unsubscribe(self.getName())
        self.is_recording = False
        # self.listening_movement.setEnabled(False)

    # Writes audio_frames out as a 4-channel, 16-bit, 48 kHz WAV.
    def save_audio_to_wav(self, filename):
        raw_data = b"".join(self.audio_frames)
        wf = wave.open(filename, 'wb')
        wf.setnchannels(config.AUDIO_CHANNELS)
        wf.setsampwidth(config.AUDIO_BYTES_PER_SAMPLE)
        wf.setframerate(config.AUDIO_SAMPLE_RATE)
        wf.writeframes(raw_data)
        wf.close()