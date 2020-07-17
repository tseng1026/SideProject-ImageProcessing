import argparse
import glob
import attr
import numpy as np
import cv2

class AddNoise():
	def __init__ (self, img):
		self.img   = img.astype("float")
		self.shape = img.shape

	def Gaussian(self):
		noise = np.random.normal(0, 20, self.shape)
		return noise

	def Uniform(self):
		noise = np.random.uniform(-127, 128, self.shape)
		return noise

	def Impulse(self):
		noise = self.Uniform()
		_, noise = cv2.threshold(noise, 100, 255, cv2.THRESH_BINARY)
		return noise

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--input" , default = "input", help = "path to the imput  image file directory")
	parser.add_argument("-o", "--output", default = "input", help = "path to the output image file directory")
	args = parser.parse_args()

	methods   = ["Gaussian", "Uniform", "Impulse"]
	filenames = glob.glob("{}/input_*.jpg".format(args.input)) + glob.glob(args.input + "/*.png")

	for filename in filenames:
		img = cv2.imread(filename)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = cv2.resize(img, (640, 640))
		add_noise = AddNoise(img)

		for method in methods:
			noise = getattr(add_noise, method)()
			noise = noise * 0.5 + img.astype("float")

			cv2.imwrite("{}/input_{}.jpg".format(args.output, method), noise)