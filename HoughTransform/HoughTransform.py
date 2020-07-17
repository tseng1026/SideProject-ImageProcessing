import argparse
import glob
import attr
import numpy as np
import cv2

class Hough():
	def __init__ (self, img):
		self.img = img
		self.shape = img.shape

		theta = np.arange(180)
		self.sink = np.sin(np.deg2rad(theta))
		self.cosk = np.cos(np.deg2rad(theta))
		self.dist  = np.ceil(np.sqrt(img.shape[0] ** 2 + img.shape[1] ** 2)).astype("int")

	def preprocessing(self):
		img = self.img.copy()
		img = cv2.GaussianBlur(img, (5, 5), 1)
		img = cv2.Canny(img, 150, 200)
		_, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
		# self.img = img.astype("float")
		self.img = img
		return self.img

	def HoughTransform(self, threshold = 240):
		# lines = cv2.HoughLines(self.img, 1, np.pi / 180, threshold)[:,0,:]
		# lines[:,1] = np.rad2deg(lines[:,1])
		# lines = lines.astype("int")
		
		img = self.img.copy()

		hough = np.zeros((self.dist * 2, 180))
		for i in range(self.shape[0]):
			for j in range(self.shape[1]):
				if img[i][j] == 0: continue

				rho = np.round(i * self.sink + j * self.cosk).astype("int")
				hough[rho + self.dist, np.arange(180)] += 1
				
		lines = np.array(np.where(hough > threshold)).T
		lines[:,0] -= self.dist

		hough = cv2.equalizeHist(hough.astype("uint8"))
		cv2.imwrite("{}/Hough.jpg". format("output"), hough)
		print (lines)
		return lines

	def DrawLines(self, img, lines):
		for rho, theta in lines:
			### i*sin = rho * (sin**2) + 1000 * sin * cos
			### j*cos = rho * (cos**2) - 1000 * sin * cos
			i1 = np.round(rho * self.cosk[theta] - 2000 * self.sink[theta]).astype("int")
			j1 = np.round(rho * self.sink[theta] + 2000 * self.cosk[theta]).astype("int")

			### i*sin = rho * (sin**2) - 1000 * sin * cos
			### j*cos = rho * (cos**2) + 1000 * sin * cos
			i2 = np.round(rho * self.cosk[theta] + 2000 * self.sink[theta]).astype("int")
			j2 = np.round(rho * self.sink[theta] - 2000 * self.cosk[theta]).astype("int")
			cv2.line(img, (i1, j1), (i2, j2), (0, 0, 255), 2)
		return img

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--input" , default = "input" , help = "path to the imput  image file directory")
	parser.add_argument("-o", "--output", default = "output", help = "path to the output image file directory")
	args = parser.parse_args()

	filenames = glob.glob("{}/input_*.jpg".format(args.input)) + glob.glob(args.input + "/*.png")

	for filename in filenames:
		img = cv2.imread(filename)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = cv2.resize(img, (640, 640))

		res = cv2.imread(filename)
		res = cv2.resize(res, (640, 640))

		hough = Hough(img)

		img   = hough.preprocessing()
		lines = hough.HoughTransform()
		res   = hough.DrawLines(res, lines)

		cv2.imwrite("{}/Canny.jpg".format(args.output), img)
		cv2.imwrite("{}/Lines.jpg".format(args.output), res)