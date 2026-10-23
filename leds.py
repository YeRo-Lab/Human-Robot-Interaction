# "Thinking" LED indicator (mixin). Uses self.leds, self.animated_speech and the _processing_* state.
import threading


class LedsMixin(object):

    # Starts the yellow "thinking" eyes on a daemon thread. If wait_for_id is given, waits for that speech first.
    def _start_processing_leds(self, wait_for_id=None):
        self._processing_active = True

        def _spin():
            if wait_for_id is not None:
                try:
                    self.animated_speech.wait(wait_for_id, 0)
                except Exception as e:
                    self._log("Wait on speech id failed:", e)
            yellow = 0x00FFFF00
            while self._processing_active:
                self.leds.rotateEyes(yellow, 0.5, 0.5)  # one quick spin cycle, repeats while flag is True

        self._processing_thread = threading.Thread(target=_spin)
        self._processing_thread.daemon = True
        self._processing_thread.start()

    # Stops the "thinking" eyes.
    def _stop_processing_leds(self):
        self._processing_active = False
        if self._processing_thread is not None:
            self._processing_thread.join(timeout=1.0)