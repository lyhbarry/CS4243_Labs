# CS4243 - Lab2
# Name: Lau Yun Hui Barry
# Matric no.: A0111682M

import cv2
import numpy as np


""" Calculate required values for rgb-hsv conversion """
def rgb2hsv_get_vars(r, g, b):
	r_prime = r/255.0
	g_prime = g/255.0
	b_prime = b/255.0

	c_max = max(r_prime, g_prime, b_prime)
	c_min = min(r_prime, g_prime, b_prime)
	max_min_diff = c_max - c_min

	return r_prime, g_prime, b_prime, c_max, c_min, max_min_diff


""" Calculate required values for hsv-rgb conversion """
def hsv2rgb_get_vars(h, s, v):	
	c = s * v
	x = c * (1 - abs((h / 60.0) % 2 - 1))
	m = v - c

	r_prime = 0
	g_prime = 0
	b_prime = 0

	# h = int(h)

	if h >= 0 and h < 60:
		r_prime = c
		g_prime = x
		b_prime = 0
	elif h >= 60 and h < 120:
		r_prime = x
		g_prime = c
		b_prime = 0		
	elif h >= 120 and h < 180:
		r_prime = 0
		g_prime = c
		b_prime = x		
	elif h >= 180 and h < 240:
		r_prime = 0
		g_prime = x
		b_prime = c
	elif h >= 240 and h < 300:
		r_prime = x
		g_prime = 0
		b_prime = c	
	elif h >= 300 and h <= 360:
		r_prime = c
		g_prime = 0
		b_prime = x
	else:	
		print "Unexpected value encounter ):"

	return r_prime, g_prime, b_prime, m


""" Perform rgb-hsv conversion """
def rgb_to_hsv(image):	
	x_len = image.shape[0]
	y_len = image.shape[1]

	# Inititate zero arrays for writing
	converted_image = np.zeros(image.shape)	

	# Perform rgb to hsv conversion on each pixel of image
	for x in range(x_len):
		for y in range(y_len):
			# Initialise R, G and B values from image
			r = image[x][y][2]
			g = image[x][y][1]
			b = image[x][y][0]

			r_prime, g_prime, b_prime, c_max, c_min, max_min_diff = rgb2hsv_get_vars(r, g, b)
			
			# Get Hue (H)
			if max_min_diff == 0:
				h = 0.0
			elif c_max == r_prime:
				h = 60.0 * (((g_prime - b_prime)/max_min_diff) % 6)
			elif c_max == g_prime:
				h = 60.0 * (((b_prime - r_prime)/max_min_diff) + 2)
			elif c_max == b_prime:
				h = 60.0 * (((r_prime - g_prime)/max_min_diff) + 4)
			else:
				print "Unexpected value encountered :("	

			# Get Saturation (S)
			if c_max == 0:
				s = 0.0
			else:
				s = max_min_diff / c_max

			# Get Value (V)	
			v = c_max	

			# Normalise H, S and V to 0-255 range
			converted_image[x][y][2] = (h / 360.0) * 255
			converted_image[x][y][1] = 255 * s
			converted_image[x][y][0] = 255 * v

	return converted_image


""" Perform hsv-rgb conversion """
def hsv_to_rgb(image):
	x_len = image.shape[0]
	y_len = image.shape[1]

	# Initiate zero arrays for writing
	converted_image = np.zeros(image.shape)

	# Build images list for conversion
	for x in range(x_len):
		for y in range(y_len):
			# Initialise H, S and V values from images
			h = (image[x][y][2] / 255.0) * 360
			s = image[x][y][1] / 255.0
			v = image[x][y][0] / 255.0

			r_prime, g_prime, b_prime, m = hsv2rgb_get_vars(h, s, v)

			r = (r_prime + m) * 255.0
			g = (g_prime + m) * 255.0
			b = (b_prime + m) * 255.0

			# Normalise H, S and V to 0-255 range
			converted_image[x][y][2] = r
			converted_image[x][y][1] = g
			converted_image[x][y][0] = b

	return converted_image					


""" Perform histogram equalization """
def hist_eq(image_hue, image_saturation, image_brightness, image_name):
	# Read images
	image_hue = cv2.imread(image_hue)
	image_saturation = cv2.imread(image_saturation)
	image_brightness = cv2.imread(image_brightness)

	x_len = image_brightness.shape[0]
	y_len = image_brightness.shape[1]		

	histeq_image = np.zeros(image_brightness.shape)

	# Get frequency of gray levels in brightness image with PMF
	gray_level_freq = np.bincount(image_brightness[:,:,2].flatten())

	# Perform CDF on gray levels in brightness image
	cdf = []
	for i in range(256):
		if i != 0:
			prev_value = cdf[i - 1]
		else:
			prev_value = 0
		val = float(gray_level_freq[i])/np.sum(gray_level_freq) + prev_value
		cdf.append(val)

	# Replaces current gray level values with the new gray level values obtained	
	for x in range(x_len):
		for y in range(y_len):
			gray_level = image_brightness[x][y][2]
			histeq_image[x][y][0] = int(cdf[gray_level] * 255)

	histeq_image[:,:,1] = image_saturation[:,:,1]
	histeq_image[:,:,2] = image_hue[:,:,0]

	return histeq_image


""" Main function to perform all conversions and equalization """
def main():
	# Build list of images name for conversion
	name_list = ['concert', 'sea1', 'sea2']

	for name in name_list:
		# Perform RGB to HSV conversion, 
		# then write H, S and V images seperately
		image = cv2.imread(name + '.jpg')
		rgb2hsv_image = rgb_to_hsv(image)

		cv2.imwrite(name + "_hue.jpg", rgb2hsv_image[:,:,2])
		cv2.imwrite(name + "_saturation.jpg", rgb2hsv_image[:,:,1])
		cv2.imwrite(name + "_brightness.jpg", rgb2hsv_image[:,:,0])

		# Perform HSV to RGB conversion,
		# then write image
		image_hue = cv2.imread(name + '_hue.jpg')
		image_saturation = cv2.imread(name + '_saturation.jpg')
		image_brightness = cv2.imread(name + '_brightness.jpg')			

		image = np.zeros(image_hue.shape)
		image[:,:,2] = image_hue[:,:,2]
		image[:,:,1] = image_saturation[:,:,1]
		image[:,:,0] = image_brightness[:,:,0]		

		hsv2rgb_image = hsv_to_rgb(image)
		cv2.imwrite(name + "_hsv2rgb.jpg", hsv2rgb_image)

		# Perform histogram equalization, HSV to RGB conversion
		# then write image
		histeq_image = hist_eq(name + '_hue.jpg', name + '_saturation.jpg', name + '_brightness.jpg', name)
		histeq_image = hsv_to_rgb(histeq_image)
		cv2.imwrite(name + '_histeq.jpg', histeq_image)
	

# Run program
main()	

