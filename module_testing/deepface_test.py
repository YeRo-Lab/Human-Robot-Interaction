from deepface import DeepFace
import os
import traceback

captures_dir = "captures"
images = [os.path.join(captures_dir, f) for f in os.listdir(captures_dir) if f.endswith(".jpg")]

print("Found", len(images), "images to test")

for i in range(len(images)):
    for j in range(i + 1, len(images)):
        try:
            result = DeepFace.verify(img1_path=images[i], img2_path=images[j], enforce_detection=False)
            print(images[i], "vs", images[j], "-> verified:", result["verified"], "distance:", result["distance"])
        except Exception as e:
            print(images[i], "vs", images[j], "-> ERROR:")
            traceback.print_exc()