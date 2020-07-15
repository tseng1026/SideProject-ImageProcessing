import argparse
import glob
import numpy as np
import cv2

def sobel (img):
	gradx = cv2.Sobel(img, -1, 0, 1)
	grady = cv2.Sobel(img, -1, 1, 0)
	return cv2.bitwise_or(gradx, grady)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--input" , default = "input" , help = "path to the imput  image file directory" )
	parser.add_argument("-o", "--output", default = "output", help = "path to the output image file directory")
	args = parser.parse_args()

	filenames = glob.glob("{}/input_*.jpg".format(args.input)) + glob.glob(args.input + "/*.png")

	for filename in filenames:
		img = cv2.imread(filename)
		img = cv2.resize(img, (640, 640))

		inv = 255 - img
		inv = cv2.GaussianBlur(inv, (5, 5), 0)

		col = cv2.divide(img, 255 - inv, scale = 256)
		gry = cv2.cvtColor(col, cv2.COLOR_BGR2GRAY)
		
		# gry, col = cv2.pencilSketch(img, sigma_s = 60, sigma_r = 0.2, shade_factor = 0.05)
		cv2.imwrite("{}/output_Gray.jpg". format(args.output), gry)
		cv2.imwrite("{}/output_Color.jpg".format(args.output), col)