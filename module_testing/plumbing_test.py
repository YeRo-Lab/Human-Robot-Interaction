from naoqi import ALProxy
robot_ip = "10.201.168.249"
robot_port = 9559
tts = ALProxy("ALTextToSpeech", robot_ip, robot_port)
tts.say("Hello from Python")
print("Success: robot spoke the line")