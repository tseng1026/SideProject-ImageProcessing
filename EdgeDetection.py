import argparse
import glob
import attr
import numpy as np
import cv2

class EdgeDetector():
	def __init__ (self, img):
		self.img   = img.astype("float")
		self.shape = img.shape

	def Forward(self, n = 10):
		img = self.img.copy()
		mask = np.array([[0, 0, 0], [0, n, 0], [0, -n, 0]])
		grad = cv2.filter2D(img, ddepth = -1, kernel = mask)

		edges = np.clip(grad, 0, 255)
		return edges

	def Backward(self, n = 10):
		img = self.img.copy()
		mask = np.array([[0, -n, 0], [0, n, 0], [0, 0, 0]])
		grad = cv2.filter2D(img, ddepth = -1, kernel = mask)
		
		edges = np.clip(grad, 0, 255)
		return edges

	def Prewitt(self):
		img = self.img.copy()
		mask = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
		gradx = cv2.filter2D(img, ddepth = -1, kernel = mask)
		grady = cv2.filter2D(img, ddepth = -1, kernel = mask.T)
		grad  = np.sqrt(gradx**2 + grady**2)
		
		edges = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX)
		return edges

	def Sobel(self):
		# gradx = cv2.Sobel(self.img, ddepth = -1, 1, 0)
		# grady = cv2.Sobel(self.img, ddepth = -1, 0, 1)
		# edges = cv2.bitwise_or(gradx, grady)

		img = self.img.copy()
		mask = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
		gradx = cv2.filter2D(img, ddepth = -1, kernel = mask)
		grady = cv2.filter2D(img, ddepth = -1, kernel = mask.T)
		grad  = np.sqrt(gradx**2 + grady**2)
		
		edges = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX)
		return edges

	def Canny(self, thresholdL = 30, thresholdH = 90):
		# edges = cv2.Canny(self.img, thresholdL, thresholdH)

		img = self.img.copy()
		img = cv2.GaussianBlur(img, (3, 3), 0)

		mask = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
		gradx = cv2.filter2D(img, ddepth = -1, kernel = mask)
		grady = cv2.filter2D(img, ddepth = -1, kernel = mask.T)
		grad  = np.sqrt(gradx**2 + grady**2)

		theta = np.arctan2(gradx, grady) / np.pi * 180
		theta = np.where(theta >= 0, theta, theta + 180)

		# non maximum supression
		for i in range(1, self.shape[0] - 1):
			for j in range(1, self.shape[1] - 1):
				if    22.5 > theta[i][j] and grad[i][j] >= grad[i-1][j] and grad[i][j] >= grad[i+1][j]: img[i][j] = grad[i][j]
				elif 157.5 < theta[i][j] and grad[i][j] >= grad[i-1][j] and grad[i][j] >= grad[i+1][j]: img[i][j] = grad[i][j]
				elif  22.5 < theta[i][j] <  67.5 and grad[i][j] >= grad[i-1][j-1] and grad[i][j] >= grad[i+1][j+1]: img[i][j] = grad[i][j]
				elif  67.5 < theta[i][j] < 112.5 and grad[i][j] >= grad[i  ][j-1] and grad[i][j] >= grad[i  ][j+1]: img[i][j] = grad[i][j]
				elif 112.5 < theta[i][j] < 157.5 and grad[i][j] >= grad[i-1][j+1] and grad[i][j] >= grad[i+1][j-1]: img[i][j] = grad[i][j]
				else: img[i][j] = 0

		edges = np.zeros(self.shape)
		for i in range(1, self.shape[0] - 1):
			for j in range(1, self.shape[1] - 1):
				if thresholdL < img[i][j] < thresholdH:
					if np.sum(np.where(img[i-1:i+2, j-1:j+2] >= thresholdH, 1, 0)) > 0:
						edges[i][j] = 1
				elif img[i][j] >= thresholdH: edges[i][j] = 1
				elif img[i][j] <= thresholdL: edges[i][j] = 0
		edges = edges * img
		return edges

	def Laplacian(self):
		img = self.img.copy()
		mask = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
		grad = cv2.filter2D(img, ddepth = -1, kernel = mask)
		
		edges = np.clip(grad, 0, 255)
		return edges

	def LaplacianOfGaussian(self, sigma = 0.8):
		# edges = cv2.Laplacian(img, ddepth = -1)

		img = self.img.copy()

		y,x = np.ogrid[-3:4,-3:4]
		mask = np.exp((x**2 + y**2) / (-2 * sigma ** 2))
		mask = mask * (x**2 + y**2  -   2 * sigma ** 2) / sigma**4
		mask = mask / np.sqrt(2 * np.pi * sigma **2)
		grad = cv2.filter2D(img, ddepth = -1, kernel = mask)

		edges = np.clip(grad, 0, 255)
		return edges

	def DifferenceOfGaussian(self, sigma1 = 0.001, sigma2 = 1000):
		img = self.img.copy()
		sigma1, sigma2 = max(sigma1, sigma2), min(sigma1, sigma2)

		tmp1 = cv2.GaussianBlur(img, (5, 5), sigmaX = sigma1, sigmaY = sigma1)
		tmp2 = cv2.GaussianBlur(img, (5, 5), sigmaX = sigma2, sigmaY = sigma2)
		grad = tmp1 - tmp2

		edges = np.clip(grad, 0, 255)
		return edges

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--input" , default = "input" , help = "path to the imput  image file directory" )
	parser.add_argument("-o", "--output", default = "output", help = "path to the output image file directory")
	args = parser.parse_args()

	methods   = ["Forward", "Backward", "Prewitt", "Sobel", "Canny", \
				 "Laplacian", "LaplacianOfGaussian", "DifferenceOfGaussian"]
	filenames = glob.glob("{}/input_*.jpg".format(args.input)) + glob.glob(args.input + "/*.png")
	
	for filename in filenames:
		img = cv2.imread(filename)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = cv2.resize(img, (640, 640))
		edge_detector = EdgeDetector(img)

		for method in methods:
			edges = getattr(edge_detector, method)()
			cv2.imwrite("{}/output_{}.jpg".format(args.output, method), edges)
