# SideProject - ImageProcessing

## Basic Execution

- **Platform:** Unix (Macbook)
- **Language:** Python3
- **Environment:** CPU
- **Library**: Opencv
  - Install ``pip install opencv-python``.
  - This repository mainly use the library ``opencv``, the tutoiral of the library can be found via the [link](https://opencv-python-tutroals.readthedocs.io/en/latest/index.html). 
  - Most of the functions are provided by OpenCV, but to understand the concepts more, I implement the details independently.

## Introduction

- **Image Enhancement**
  - The goal is to make images more appealing.
  - The main idea is to rescale the original image so that the intensity differences between pixels become larger.
  - There are two approaches to achieve the target, contrast manipulation (negative, power law transformation, and logarithmic point transformation) as well as histogram modification (histogram equalization).
  - Other worth reading topics would be gamma correction, tone mapping, and high dynamic range.

- **Image Restoration**
  - In the project, I more focus on noise-cleaning.
  - The key concept is to remove the noise caused by electrical sensors, photographic grain noise, channel error, etc.
  - The noise is usually discrete, not spatially correlated, and high spatial frequency; noise can be categorized into two types: uniform (additive uniform noise, Gaussian noise) and impulse noise (salt and pepper noise).
  - There are two strategies respectively to deal with these noises, low pass filtering for uniform noise and non-linear filtering for impulse noise.
  - Other interesting references would be EPLL and BM3D.

- **Edge Detection**
  - Edge detection is motivated since human eyes are more sensitive to edges; besides, it is a fundamental step in image analysis.
  - The essential characteristic of the edge is that the intensity differences, so-called gradient, are significant on opposite sides of the edge.
  - There are two methods; one rarely used method is a model-based method, and the other is non-parametric approaches.
  - *Question: Is there an algorithm identifying the image is sharp or not?*

- **Hough Transform**
  - This procedure is usually implemented after edge detection to detect lines; hough transformation can tolerate gaps in the edges and is relatively unaffected by noise.
  - A more useful representation for line in hough transformation is 
  - <img src="https://render.githubusercontent.com/render/math?math=\rho = x\cos\theta %2B y\sin\theta">
  - The concept is to vote for the parameters permutation and select the terms with the votes greater than the threshold.

- **Morphological Processing**
  - Morphology is a post-processing process on the binary image.
  - Simple morphological processing is implemented based on binary hit or miss rules, whereas for advanced ones, the conditional array (or mask) is introduced. 
  - These processing strategies are helpful for boundary extraction, hole filling, and connected component labeling.

## Acknowledgment

This project is based on the course *NTU CSIE 5612 Digital Image Processing*, taught by MingSui (Amy) Lee. I want to show my sincere gratitude to Prof. Lee. It was my first course combining computer science and image and spurred me to research vision topics deeper.
