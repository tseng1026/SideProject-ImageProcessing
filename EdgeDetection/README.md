# SideProject - EdgeDetection

## Basic Execution
- **Platform:** Unix (Macbook)
- **Language:** Python3
- **Environment:** CPU
- **Usage:**
	- ``python EdgeDetection.py -i <input dir> -o <output dir>``
- **Requirements:**
	- python3.6
	- glob
	- attr
	- opencv ``pip install opencv-python``
	- numpy ``pip install numpy``

## Algorithm
- forward / backward
- prewitt
- sobel
- canny
- laplacian
- laplacian of gaussian (LoG)
- difference of gaussian (DoG)

## Application
- pencil sketch
	- find the inversed image
	- apply Gaussian blur toward inversed image
	- dodge blend the original image and blurred inverted image

## Reference
- *Object Enhancement and Extraction*, J. M. S. Prewitt, Picture processing and Psychopictorics 1970
- *An Isotropic 3x3 Image Gradient Operator*, I. Sobel, G. Feldman, Pattern Classification and Scene Analysis, 1973
- *A computational approach to edge detection*, J. Canny, IEEE 1996
- *Fast convolution with Laplacian-of-Gaussian masks*, J. S. Chen, A. Huertas, G. Medioni, IEEE 1987
