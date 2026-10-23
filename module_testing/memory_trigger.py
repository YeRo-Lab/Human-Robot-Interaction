from naoqi import ALProxy
robot_ip = "10.201.168.249"
robot_port = 9559
tracker = ALProxy("ALTracker", robot_ip, robot_port)
tracker.stopTracker()
# memory = ALProxy("ALMemory", robot_ip, robot_port)
# memory.raiseEvent("MyTestEvent", "hello from trigger script")