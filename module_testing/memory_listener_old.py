from naoqi import ALProxy, ALBroker, ALModule
import time
import threading
import urllib2
import json

# robot_ip = "127.0.0.1" #localhost
robot_ip = "10.201.168.249"
robot_port = 9559

myBroker = ALBroker("myBroker", "0.0.0.0", 0, robot_ip, robot_port)

class MemoryListenerModule(ALModule):
    def __init__(self, name):
        ALModule.__init__(self, name)

        # ALMemory setup for subscriptions/testing events
        memory = ALProxy("ALMemory")
        memory.subscribeToEvent("MyTestEvent", "MemoryListener", "onTestEvent")

        # text-to-speech setup
        self.tts = ALProxy("ALTextToSpeech")

        # Face detecton config/setup
        memory.subscribeToEvent("FaceDetected", "MemoryListener", "onFaceDetected")
        self.last_announced_time = 0
        self.cooldown_seconds = 5
        self.lock = threading.Lock()

        # Tracking setup (registered but not started yet)
        self.motion = ALProxy("ALMotion")
        self.tracker = ALProxy("ALTracker")
        self.motion.wakeUp()
        self.tracker.registerTarget("Face", 0.1)
        self.is_tracking = False

        # LED setup
        self.leds = ALProxy("ALLeds")

    # test event, manually triggered
    def onTestEvent(self, key, value, message):
        print("Event fired! Key:", key, "Value:", value, "Message:", message)
        self.tts.say("I noticed a memory update")

    # method exectued on face detection
    def onFaceDetected(self, key, value, message):
        face_info_array = value[1] if len(value) > 1 else []
        if face_info_array and len(face_info_array) > 0:

            # if not tracking current face, start tracking
            if not self.is_tracking:
                self.tracker.track("Face")
                self.is_tracking = True
                print("Face detected - tracking started.")

            # if face detected, and cooldown has passed, respond/call llm
            should_respond = False
            with self.lock:
                now = time.time()
                if now - self.last_announced_time > self.cooldown_seconds:
                    self.last_announced_time = now
                    should_respond = True

            # one thread will handle llm response
            if should_respond:
                # print(now, "New face detected!")
                self.leds.fadeRGB("FaceLeds", 0.0, 1.0, 0.0, 0.3)  # green = about to speak
                self.tts.say("Face detected")
                self.leds.fadeRGB("FaceLeds", 0.0, 0.0, 1.0, 0.3)  # blue = listening/idle again
                # answer = self.get_llm_response("test")
                # print(now, "llm_response: " + answer)
                # self.tts.say(answer)

    # make POST request to local server (python 3.13) to prompt llm and send response
    def get_llm_response(self, text):
        url = "http://127.0.0.1:5000/respond"
        payload = json.dumps({"text": text})
        req = urllib2.Request(url, payload, {'Content-Type': 'application/json'})
        response = urllib2.urlopen(req)
        result = json.loads(response.read())
        # print(result["answer"].encode('utf-8'))
        return result["answer"].encode('utf-8')

MemoryListener = MemoryListenerModule("MemoryListener")

print("Subscribed and listening...")
try:
    while True:
        time.sleep(1)
# allow ctrl-C to auto stop ALtracker, and rest motors
except KeyboardInterrupt:
    print("Stopping tracker...")
    MemoryListener.tracker.stopTracker()
    MemoryListener.tracker.unregisterAllTargets()
    MemoryListener.motion.rest()
    print("Cleaned up. Exiting.")