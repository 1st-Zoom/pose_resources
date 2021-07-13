import numpy as np
import enum

class BodyPart:
	NOSE = 1
	LEFT_EY = 2
	RIGHT_EYE = 3
	LEFT_EAR = 4
	RIGHT_EAR = 5
	LEFT_SHOULDER = 6
	RIGHT_SHOULDER = 7
	LEFT_ELBOW = 8
	RIGHT_ELBOW = 9
	LEFT_WRIST = 10
	RIGHT_WRIST = 11
	LEFT_HIP = 12
	RIGHT_HIP = 13
	LEFT_KNEE = 14
	RIGHT_KNEE = 15
	LEFT_ANKLE = 16
	RIGHT_ANKLE = 17

class estimateSinglePose:
	def __init__(self, bitmap):
		self.bitmap = bitmap

	def Position(self):
		self.x = 0
		self.y = 0

	def KeyPoint(self):
		self.bodyPart = BodyPart.NOSE.name
		self.position = Position()
		self.score = 0.0

	def Person(self):
		self.keyPoints = list((KeyPoint()));
		self.score = 0.0
		


		