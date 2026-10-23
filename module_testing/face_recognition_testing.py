from naoqi import ALProxy
import time

robot_ip = "10.201.168.249"
robot_port = 9559

motion = ALProxy("ALMotion", robot_ip, robot_port)
motion.wakeUp()

face_detection = ALProxy("ALFaceDetection", robot_ip, robot_port)
memory = ALProxy("ALMemory", robot_ip, robot_port)

face_detection.subscribe("FaceTest")

print("Waiting for a face to be detected before learning...")
try:
    face_confirmed = False
    while not face_confirmed:
        value = memory.getData("FaceDetected")
        if value and len(value) > 1 and value[1]:
            print("Face detected, learning now.")
            success = face_detection.learnFace("TestUser")
            print("Learn success:", success)
            face_confirmed = True
        time.sleep(0.5)

    print("Now checking recognition... (Ctrl+C to stop)")
    while True:
        value = memory.getData("FaceDetected")
        print(time.time(), value)
        time.sleep(1)
except KeyboardInterrupt:
    print("KeyboardInterrupt: Stopping...")
    face_detection.unsubscribe("FaceTest")
    motion.rest()
    print("Cleaned up. Exiting.")