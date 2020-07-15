import argparse
import glob
import attr
import numpy as np
import cv2

class ImageEnhancement():
	def __init__ (self, img):
		self.img   = img.astype("float")
		self.shape = img.shape

	def Negative(self):
		img = self.img.copy()

		neg = 255 - img
		return neg

	def Power(self, p = 2):
		img = self.img.copy().astype("float")
		img = img / 255

		pow = img ** p
		pow = pow * 255
		return pow

	def Logarithmic(self, a = 1):
		img = self.img.copy().astype("float")
		img = img / 255

		log = np.log(1 + a * img) / np.log(2)
		log = log * 255
		return log

	def Histogram(self):
		# eql = cv2.equalizeHist(img)

		img = self.img.copy()

		hst, _ = np.histogram(img.flatten(), 256, [0,256])
		cdf = np.cumsum(hst)
		cdf = np.ma.masked_equal(cdf, 0)

		cdf = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
		cdf = np.ma.filled(cdf, 0).astype('uint8')
		eql = cdf[img.astype("int")]
		return eql

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--input" , default = "input" , help = "path to the imput  image file directory")
	parser.add_argument("-o", "--output", default = "output", help = "path to the output image file directory")
	args = parser.parse_args()

	methods   = ["Negative", "Power", "Logarithmic", "Histogram"]
	filenames = glob.glob("{}/input_*.jpg".format(args.input)) + glob.glob(args.input + "/*.png")

	for filename in filenames:
		img = cv2.imread(filename)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = cv2.resize(img, (640, 640))

		image_enhancement = ImageEnhancement(img)

		for method in methods:
			clear = getattr(image_enhancement, method)()
			cv2.imwrite("{}/output_{}.jpg".format(args.output, method), clear)