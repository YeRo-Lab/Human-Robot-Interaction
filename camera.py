# Face photo capture (mixin). Uses self.video_device and self.human_engaged.
import time

import numpy as np
import cv2


class CameraMixin(object):

    # IMAGE PROCESSING:

    # Grabs up to num_photos sharp frames (Laplacian variance >= blur_threshold) from the top camera, stopping
    # early if the person leaves or the engagement is superseded. Falls back to the sharpest frame;
    # returns None if every attempt fails.
    def _capture_face_photo(self, engagement_id, filename_prefix=None, max_attempts=8, blur_threshold=100.0, num_photos=3):
        AL_kTopCamera = 0
        AL_kVGA = 2
        AL_kBGRColorSpace = 13

        capture_client = self.video_device.subscribeCamera(
            "FaceCaptureClient", AL_kTopCamera, AL_kVGA, AL_kBGRColorSpace, 10
        )

        if filename_prefix is None:
            filename_prefix = "face_capture_" + str(int(time.time()))

        good_filepaths = []
        best_image = None
        best_sharpness = -1

        for attempt in range(1, max_attempts + 1):
            if self._is_stale(engagement_id):
                self._log("Human left mid-capture, stopping early.")
                break

            result = self.video_device.getImageRemote(capture_client)

            if result is None:
                self._log("Capture attempt", attempt, "failed: no image returned.")
                continue

            width = result[0]
            height = result[1]
            array = result[6]

            image = np.frombuffer(array, dtype=np.uint8).reshape((height, width, 3))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

            self._log("Capture attempt", attempt, "sharpness score: %.1f" % sharpness)

            if sharpness > best_sharpness:
                best_sharpness = sharpness
                best_image = image

            if sharpness >= blur_threshold:
                filepath = "captures/" + filename_prefix + "_" + str(len(good_filepaths) + 1) + ".jpg"
                cv2.imwrite(filepath, image)
                good_filepaths.append(filepath)
                self._log("Good photo", len(good_filepaths), "of", num_photos, "saved:", filepath)

                if len(good_filepaths) >= num_photos:
                    break

        self.video_device.unsubscribe(capture_client)

        if len(good_filepaths) > 0:
            return good_filepaths

        if best_image is not None:
            fallback_path = "captures/" + filename_prefix + "_fallback.jpg"
            cv2.imwrite(fallback_path, best_image)
            self._log("No image cleared threshold; saved best available (sharpness: %.1f):" % best_sharpness, fallback_path)
            return [fallback_path]

        self._log("All capture attempts failed.")
        return None