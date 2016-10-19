# CS4243 - Lab 3
# Name: Lau Yun Hui Barry
# Matric no.: A0111682M
# Fri 10am session

import cv2
import numpy as np


""" Perform convolution on image with selected filter """
def MyConvolve(img, ff):
	result = np.zeros(img.shape)
	x_len = img.shape[0]
	y_len = img.shape[1]

	ff = np.flipud(np.fliplr(ff))

	# Apply filter to pixels
	for x in range(1, x_len - 1):
		for y in range(1, y_len - 1):
			# Left column
			top_left = img[x-1, y-1] * ff[0, 0]
			left = img[x, y-1] * ff[1, 0]
			btm_left = img[x+1, y-1] * ff[2, 0]
			# Middle column
			top = img[x-1, y] * ff[0, 1]
			middle = img[x, y] * ff[1, 1]
			btm = img[x+1, y] * ff[2, 1]
			# Right column
			top_right = img[x-1, y+1] * ff[0, 2]
			right = img[x, y+1] * ff[1, 2]
			btm_right = img[x+1, y+1] * ff[2, 2]

			result[x, y] = top_left + left + btm_left + top + middle + btm + top_right + right + btm_right

	return result		


""" Perform Prewitt edge detection """
def PrewittEdgeDetector(img):
	result = np.zeros(img.shape)
	x_len = img.shape[0]
	y_len = img.shape[1]

	vertical_filter = np.array([[-1, 0, 1]
							   ,[-1, 0, 1]
							   ,[-1, 0, 1]])
	horizontal_filter = np.array([[1, 1, 1]
							     ,[0, 0, 0]
							     ,[-1, -1, -1]])

	# Apply both horizontal and vertical filter to image
	x_result = MyConvolve(img, vertical_filter)
	y_result = MyConvolve(img, horizontal_filter)

	# Combine results to form image
	for x in range(1, x_len - 1):
		for y in range(1, y_len - 1):
			result[x, y] = np.sqrt(x_result[x, y] ** 2 + y_result[x, y] ** 2) / 3

	return result


""" Perform Sobel edge detection """
def SobelEdgeDetector(img):
	result = np.zeros(img.shape)
	x_len = img.shape[0]
	y_len = img.shape[1]

	vertical_filter = np.array([[-1, 0, 1]
							   ,[-2, 0, 2]
							   ,[-1, 0, 1]])
	horizontal_filter = np.array([[1, 2, 1]
							     ,[0, 0, 0]
								 ,[-1, -2, -1]])	

	# Apply both horizontal and vertical filter to image
	x_result = MyConvolve(img, vertical_filter)
	y_result = MyConvolve(img, horizontal_filter)

	# Combine results to form image
	for x in range(1, x_len - 1):
		for y in range(1, y_len - 1):
			result[x, y] = np.sqrt(x_result[x, y] ** 2 + y_result[x, y] ** 2) / 4

	return result	


""" Perform edge thinning on image """
def EdgeThinning(img):
	result = np.zeros(img.shape)
	x_len = img.shape[0]
	y_len = img.shape[1]

	for x in range(1, x_len - 1):
		for y in range(1, y_len - 1):
			# Build horizontal and vertical edge list
			vertical_edge_list = [img[x-1, y], img[x+1, y]]
			horizontal_edge_list = [img[x, y-1], img[x, y+1]]
			
			# Get max edge of horizontal and vertical of pixel
			vertical_max_edge = max(vertical_edge_list, key=tuple)
			horizontal_max_edge = max(horizontal_edge_list, key=tuple)

			if img[x, y, 0] >= vertical_max_edge[0] or img[x, y, 0] >= horizontal_max_edge[0]:
				result[x, y] = img[x, y]
			else:
				result[x, y] = np.zeros(result[x, y].shape)

	return result								


""" Main function """
def main():
	# Build image list for edge detection and thinning
	img_list = ['test1', 'test2', 'test3']

	for img_name in img_list:
		# Perform Prewitt edge detection on images
		img = cv2.imread(img_name + '.jpg', 0)
		img = PrewittEdgeDetector(img)
		cv2.imwrite(img_name + '_prewitt_result.jpg', img)

		# Perform Sobel edge detection on images
		img = cv2.imread(img_name + '.jpg', 0)		
		img = SobelEdgeDetector(img)
		cv2.imwrite(img_name + '_sobel_result.jpg', img)

		# Perform edge thinning on images
		img = cv2.imread(img_name + '_prewitt_result.jpg')
		img = EdgeThinning(img)
		cv2.imwrite(img_name + '_thinned_prewitt_result.jpg', img)

		img = cv2.imread(img_name + '_sobel_result.jpg')
		img = EdgeThinning(img)
		cv2.imwrite(img_name + '_thinned_sobel_result.jpg', img)


# Execute main function
main()		
