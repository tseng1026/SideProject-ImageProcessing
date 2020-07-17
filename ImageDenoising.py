import argparse
import glob
import attr
import numpy as np
import cv2

class ImageDenoising():
	def __init__ (self, img):
		self.img   = img.astype("float")
		self.shape = img.shape

	def LowPass(self, n = 2):
		img = self.img.copy()

		mask = np.array([[1, n, 1], [n, n**2, n], [1, n, 1]])
		filt = cv2.filter2D(img, ddepth = -1, kernel = mask)
		clear = cv2.normalize(filt, None, 0, 255, cv2.NORM_MINMAX)
		return clear

	def Outlier(self, e = 2):
		img = self.img.copy()

		mask = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
		filt = cv2.filter2D(img, ddepth = -1, kernel = mask)

		clear = np.where(abs(9 * img - filt) < e, img, filt / 9)
		return clear

	def Median(self, n = 3):
		# clear = cv2.medianBlur(self.img.astype("float32"), 3)

		hlf = n // 2
		img   = self.img.copy().astype("float")
		clear = self.img.copy().astype("float")
		
		for i in range(hlf, self.shape[0] - hlf):
			for j in range(hlf, self.shape[1] - hlf):
				clear[i][j] = np.median(img[i-hlf:i+hlf+1, j-hlf:j+hlf+1])

		return clear

	def PseudoMedian(self, n = 5):
		hlf = n // 2
		img   = self.img.copy().astype("float")
		clear = self.img.copy().astype("float")

		for i in range(hlf, self.shape[0] - hlf):
			for j in range(hlf, self.shape[1] - hlf):
				tmp1 = np.max(np.min(img[i-hlf:i+hlf+1, j-hlf:j+hlf+1], axis = 0))
				tmp2 = np.max(np.min(img[i-hlf:i+hlf+1, j-hlf:j+hlf+1], axis = 1))
				tmp3 = np.min(np.max(img[i-hlf:i+hlf+1, j-hlf:j+hlf+1], axis = 0))
				tmp4 = np.min(np.max(img[i-hlf:i+hlf+1, j-hlf:j+hlf+1], axis = 1))
				clear[i][j] = 0.5 * np.max([tmp1, tmp2]) + 0.5 * np.min([tmp3, tmp4])
		return clear

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--input" , default = "input" , help = "path to the imput  image file directory")
	parser.add_argument("-o", "--output", default = "output", help = "path to the output image file directory")
	args = parser.parse_args()

	methods   = ["LowPass", "Outlier", "Median", "PseudoMedian"]
	filenames = glob.glob("{}/input_*.jpg".format(args.input)) + glob.glob(args.input + "/*.png")

	for filename in filenames:
		img = cv2.imread(filename)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = cv2.resize(img, (640, 640))

		image_denoising = ImageDenoising(img)
		for method in methods:
			clear = getattr(image_denoising, method)()
			cv2.imwrite("{}/{}_{}.jpg".format(args.output, filename.split("/")[-1][6:-4], method), clear)