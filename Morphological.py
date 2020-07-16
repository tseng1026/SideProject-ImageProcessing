import argparse
import glob
import attr
import numpy as np
import cv2

class Morphological():
	def __init__ (self, img):
		self.img   = img.astype("float")
		self.shape = img.shape

	def Dilation(self, img = None, n = 3):
		# morph = cv2.dilate(self.img, np.ones((n, n)))

		if type(img) == type(None): img = self.img.copy()
		mask = np.ones((n, n))
		filt = cv2.filter2D(img, ddepth = -1, kernel = mask)

		morph = np.where(filt != 0, 255, 0)
		return morph

	def Erosion(self, img = None, n = 3):
		# morph = cv2.erode(self.img, np.ones((n, n)))

		if type(img) == type(None): img = self.img.copy()
		mask = np.ones((n, n))
		filt = cv2.filter2D(img, ddepth = -1, kernel = mask)

		morph = np.where(filt != 9 * 255, 0, 255)
		return morph

	def Opening(self, img = None, n = 3):
		# morph = cv2.morphologyEx(self.img, cv2.MORPH_OPEN, np.ones((n, n)))

		if type(img) == type(None): img = self.img.copy()

		morph = self.Erosion (img  , n).astype("float")
		morph = self.Dilation(morph, n).astype("float")
		return morph

	def Closing(self, img = None, n = 3):
		# morph = cv2.morphologyEx(self.img, cv2.MORPH_CLOSE, np.ones((n, n)))

		if type(img) == type(None): img = self.img.copy()

		morph = self.Dilation(img  , n).astype("float")
		morph = self.Erosion (morph, n).astype("float")
		return morph

	def Gradient(self, n = 3):
		# morph = cv2.morphologyEx(self.img, cv2.MORPH_GRADIENT, np.ones((n, n)))

		img = self.img.copy()
		tmp1 = self.Erosion (img, n).astype("float")
		tmp2 = self.Dilation(img, n).astype("float")

		morph = abs(tmp1 - tmp2)
		return morph

	def TopHat(self, n = 3):
		# morph = cv2.morphologyEx(self.img, cv2.MORPH_TOPHAT, np.ones((n, n)))
		
		img = self.img.copy()
		tmp = self.Opening(img, n).astype("float")

		morph = abs(img - tmp)
		return morph
	
	def BlackHat(self, n = 3):
		# morph = cv2.morphologyEx(self.img, cv2.MORPH_BLACKHAT, np.ones((n, n)))

		img = self.img.copy()
		tmp = self.Closing(img, n).astype("float")

		morph = abs(img - tmp)
		return morph


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--input" , default = "input" , help = "path to the imput  image file directory")
	parser.add_argument("-o", "--output", default = "output", help = "path to the output image file directory")
	args = parser.parse_args()

	methods   = ["Dilation", "Erosion", "Opening", "Closing", "Gradient", "TopHat", "BlackHat"]
	filenames = glob.glob("{}/input_*.jpg".format(args.input)) + glob.glob(args.input + "/*.png")

	for filename in filenames:
		img = cv2.imread(filename)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = cv2.resize(img, (640, 640))
		_, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

		morphological = Morphological(img)
		for method in methods:
			clear = getattr(morphological, method)()
			cv2.imwrite("{}/{}.jpg".format(args.output, method), clear)