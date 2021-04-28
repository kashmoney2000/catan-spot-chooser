import cv2
import imutils
import sys
import math
import tkinter as tk
import numpy as np
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
# import pickle

np.set_printoptions(threshold=sys.maxsize)

PERFECT_HEXAGON = [(52, 491), (309, 46), (824, 46), (1081, 491), (824, 939), (309, 938)]

HEXAGON_CENTERS = [(158, 76), (221, 76), (289, 76), (124, 129), (191, 131), (257, 133), (321, 131), (355, 191), (289, 189), (221, 188), (157, 189), (90, 185), (122, 245), (189, 246), (253, 245), (322, 245), (289, 305), (221, 304), (153, 303)]
BASE_WIDTH = 445
BASE_HEIGHT = 385

BOARD_PIXEL_RANGE_LOWER = (100, 75, 0)
BOARD_PIXEL_RANGE_HIGHER = (160, 110, 80)

NUMBER_PIXEL_RANGE_LOWER = (15, 110, 220)
NUMBER_PIXEL_RANGE_HIGHER = (25, 160, 260)

TEST_AREA = 411323
RADIUS = 20

# brick, wood, wheat, sheep, stone
RESOURCES = [[205, 126, 64],[106, 92, 42],[216, 148, 49],[158, 162, 53],[172, 132, 96]]
NAMES = ["brick", "wood", "grain", "sheep", "ore"]

truth = " black.jpg"
ONE_DIGIT_CONTOURS = {}
RED_ONE_DIGIT_CONTOURS = {}
TWO_DIGIT_CONTOURS = {}


def calculate_bounding_area(cnt):
	rect = cv2.boundingRect(cnt)
	area = rect[2] * rect[3]
	return area

def color_diff(color1, color2):
	color1_rgb = sRGBColor(color1[0]/255, color1[1]/255, color1[2]/255)
	color2_rgb = sRGBColor(color2[0]/255, color2[1]/255, color2[2]/255)
	color1_lab = convert_color(color1_rgb, LabColor)
	color2_lab = convert_color(color2_rgb, LabColor)
	delta_e = delta_e_cie2000(color1_lab, color2_lab)
	return delta_e

def normalized(rgb):
	norm=np.zeros((*rgb.shape[:2],3),np.float32)

	rgb = rgb.astype('int32')

	b=rgb[:,:,0]
	g=rgb[:,:,1]
	r=rgb[:,:,2]

	sum=b+g+r

	sum[sum == 0] = 1

	norm[:,:,0]=b/sum*255.0
	norm[:,:,1]=g/sum*255.0
	norm[:,:,2]=r/sum*255.0

	norm_rgb=cv2.convertScaleAbs(norm)
	return norm_rgb

def detect_corners(center, points, tolerance=50):
	dist = sorted([(math.hypot(center[0] - x[0], center[1] - x[1]), x) for x in points])[::-1]
	corners = []
	for each in dist:
		each = each[1]
		good = True
		for pt in corners:
			if(math.hypot(pt[0] - each[0], pt[1] - each[1]) < tolerance):
				good = False
				break
		if(good):
			corners.append(tuple(each))
		if(len(corners) == 6):
			return corners

def normalize_board_orientation(image, debug=False):
	normed = normalized(image)
	# hsv = cv2.cvtColor(normed, cv2.COLOR_BGR2HSV)
	# cv2.imwrite("normed_test.png", normed)
	#display_image(normed)

	blue_mask = cv2.inRange(normed, BOARD_PIXEL_RANGE_LOWER, BOARD_PIXEL_RANGE_HIGHER)

	
	output = cv2.bitwise_and(normed, normed, mask=blue_mask)
	gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
	gray = cv2.bilateralFilter(gray, 7, 24, 24)
	thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1]

	cnts = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	cnts = sorted(cnts, key = calculate_bounding_area, reverse = True)
	board_outline = cnts[0]
	M = cv2.moments(board_outline)
	cX = int(M["m10"] / M["m00"])
	cY = int(M["m01"] / M["m00"])
	# cv2.circle(image, (cX, cY), 5, (0, 255, 0), -1)
	points = [list(x[0]) for x in list(board_outline)]
	corners = detect_corners((cX, cY), points)
	corners.sort(key=lambda p: math.atan2(p[1]-cY,p[0]-cX))
	x, y, width, height = cv2.boundingRect(board_outline)
	h, status = cv2.findHomography(np.float32(corners),np.float32(PERFECT_HEXAGON))

	# print((width, height))
	im_out = cv2.warpPerspective(image, h, (1131, 980))
	display_image(im_out)
	if(debug):
		cv2.rectangle(image, (x, y), (x+width, y+height), (0, 255, 0), 2)
		cv2.drawContours(image, [board_outline], -1, (0,255,0))
		cv2.imwrite("normalized" + str(option) + ".png", normed)
		cv2.imwrite("mask" + str(option) + ".png", output)
		cv2.imwrite("output" + str(option) + ".png", image)
		cv2.imwrite("warp_test" + str(option) + ".png", im_out)

	return im_out

def calculate_bounding_box(image):
	normed = normalized(image)

	cv2.imwrite("normed_test.png", normed)

	# hsv = cv2.cvtColor(normed, cv2.COLOR_BGR2HSV)

	blue_mask = cv2.inRange(normed, BOARD_PIXEL_RANGE_LOWER, BOARD_PIXEL_RANGE_HIGHER)

	output = cv2.bitwise_and(image, image, mask=blue_mask)
	gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
	gray = cv2.bilateralFilter(gray, 7, 24, 24)
	thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1]

	cnts = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	cnts = sorted(cnts, key = calculate_bounding_area, reverse = True)
	board_outline = cnts[0]
	x, y, w, h = cv2.boundingRect(board_outline)
	# print((w, h))
	return (x, y, w, h)

def display_image(image):
	cv2.imshow("image", image)
	k = cv2.waitKey(0)
	if k == 27:
		cv2.destroyAllWindows()

def get_digit_contour(ROI):
	orig = ROI.copy()
	ROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)

	ROI[ROI == 0] = 255
	ROI[ROI >= 155] = 255


	mask = np.zeros((ROI.shape[0],ROI.shape[1]), np.uint8)
	cv2.circle(mask, (int(ROI.shape[0]/2), int(ROI.shape[1]/2)), int(ROI.shape[0]/2), (0,0,0), 5)
	cv2.circle(mask, (int(ROI.shape[0]/2), int(ROI.shape[1]/2)), int(ROI.shape[0]/2)-3, (255,255,255), -1)
	#display_image(mask)
	sample = cv2.bitwise_and(ROI, ROI, mask=mask)
	sample[sample == 0] = 255
	sample[sample != 255] = 0
	#display_image(ROI)
	#display_image(sample)

	kernel = np.ones((2,2),np.uint8)
	sample = cv2.erode(sample, kernel, iterations = 1)
	#bolden/pronounce the number

	# sample = cv2.dilate(sample, kernel, iterations = 1)

	#display_image(sample)

	cnts = cv2.findContours(sample.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	cnts = imutils.grab_contours(cnts)
	# detects square as the largest contour so parse it out
	cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[1:]
	candidates = []
	for cnt in cnts:
		if(cv2.contourArea(cnt) > 75):
			candidates.append(cnt)
	# assert len(candidates) <= 2 and len(candidates) > 0
	if(len(candidates) == 0):
		return None
	if(len(candidates) == 1):
		return candidates
	digit1 = candidates[0]
	digit2 = candidates[1]
	if(abs(cv2.contourArea(digit1) - cv2.contourArea(digit2)) >= 150):
		return [max(digit1, digit2, key=cv2.contourArea)]
	return candidates

def calculate_point_value(image, circle_center, circle_radius):
	circle_img = np.zeros((image.shape[0],image.shape[1]), np.uint8)
	crop_img = image[(circle_center[1] - circle_radius):(circle_center[1] + circle_radius), (circle_center[0] - circle_radius):(circle_center[0] + circle_radius)]

	hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)

	lower_bound = np.array([15,100,230])
	upper_bound = np.array([25,160,260])

	mask = cv2.inRange(hsv, lower_bound, upper_bound)

	kernel = np.ones((2, 2), np.uint8)

	mask = cv2.erode(mask, kernel, iterations=1)
	mask = cv2.dilate(mask, kernel, iterations=1)

	#display_image(mask)

	cnts = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	cnt = None
	for each in cnts:
		area = calculate_bounding_area(each)
		if(area > 1500 and area < 2500):
			# print(area)
			cnt = each
			break

	# cv2.drawContours(crop_img, [cnt], -1, (255,0,0))
	# display_image(mask)
	# display_image(crop_img)

	if(cnt is not None):
		x,y,w,h = cv2.boundingRect(cnt)
		(x_cent,y_cent), r = cv2.minEnclosingCircle(cnt)
		center = (int(x_cent),int(y_cent))
		r = int(r)
		output = np.zeros((crop_img.shape[0],crop_img.shape[1]), np.uint8)
		cv2.circle(output, center, r, (255,255,255), -1)
		crop_img = cv2.bitwise_and(crop_img, crop_img, mask=output)
		ROI = crop_img[y:y+h, x:x+w]

		gray_ROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
		hsv = cv2.cvtColor(ROI, cv2.COLOR_BGR2HSV)

		lower_bound = np.array([4, 158, 210])
		upper_bound = np.array([16, 202, 236])

		mask = cv2.inRange(hsv, lower_bound, upper_bound)
		output = cv2.bitwise_and(gray_ROI, gray_ROI, mask=mask)

		#display_image(output)

		output[output != 0] = 255
		white_pixs = np.sum(output == 255)

		ROI = cv2.resize(ROI, (50, 50))

		#display_image(ROI)

		digit_contour = get_digit_contour(ROI)

		# display_image(digit_contour)

		cv2.drawContours(ROI, digit_contour, -1, (0, 255, 0), 1)

		#display_image(ROI)

		if(digit_contour == None):
			return -1, 1.0

		map = None
		if(len(digit_contour) > 1):
			map = TWO_DIGIT_CONTOURS
		elif(white_pixs > 10):
			map = RED_ONE_DIGIT_CONTOURS
		else:
			map = ONE_DIGIT_CONTOURS

		min_diff_num = -1
		min_diff = 1000000
		for key in map:
			diff = diff_contours(digit_contour, map[key])
			if(diff < min_diff):
				min_diff = diff
				min_diff_num = key

		# print(min_diff_num)

		# cv2.drawContours(ROI, digit_contour, -1, (255, 0, 0))
		# display_image(ROI)


		return min_diff_num, min_diff
	else:
		return -1, 1.0

def diff_contours(contour1, contour2, con1img=None, con2img=None):
	if(len(contour1) == 1 and len(contour2) == 1):
		return cv2.matchShapes(contour1[0], contour2[0], cv2.CONTOURS_MATCH_I1, 0.5) + abs(cv2.contourArea(contour1[0]) - cv2.contourArea(contour2[0])) * 0.001
	else:

		op1 = cv2.matchShapes(contour1[0], contour2[0], cv2.CONTOURS_MATCH_I1, 0.5) + abs(cv2.contourArea(contour1[0]) - cv2.contourArea(contour2[0])) * 0.001 + cv2.matchShapes(contour1[1], contour2[1], cv2.CONTOURS_MATCH_I1, 0.5) + abs(cv2.contourArea(contour1[1]) - cv2.contourArea(contour2[1])) * 0.001


		op2 = cv2.matchShapes(contour1[0], contour2[1], cv2.CONTOURS_MATCH_I1, 0.5) + abs(cv2.contourArea(contour1[0]) - cv2.contourArea(contour2[1])) * 0.001 + cv2.matchShapes(contour1[1], contour2[0], cv2.CONTOURS_MATCH_I1, 0.5) + abs(cv2.contourArea(contour1[1]) - cv2.contourArea(contour2[0])) * 0.001

		# testing op1
		if(con1img is not None and con2img is not None):
			con1img_1 = con1img.copy()
			con2img_1 = con2img.copy()
			con1img_2 = con1img.copy()
			con2img_2 = con2img.copy()
			cv2.drawContours(con1img_1, [contour1[0]], -1, (255, 0, 0))
			cv2.drawContours(con1img_1, [contour1[1]], -1, (0, 255, 0))
			cv2.drawContours(con2img_1, [contour2[0]], -1, (255, 0, 0))
			cv2.drawContours(con2img_1, [contour2[1]], -1, (0, 255, 0))
			v_stacked_1 = np.vstack([con1img_1, con2img_1])

			cv2.drawContours(con1img_2, [contour1[0]], -1, (255, 0, 0))
			cv2.drawContours(con1img_2, [contour1[1]], -1, (0, 255, 0))
			cv2.drawContours(con2img_2, [contour2[1]], -1, (255, 0, 0))
			cv2.drawContours(con2img_2, [contour2[0]], -1, (0, 255, 0))
			v_stacked_2 = np.vstack([con1img_2, con2img_2])
			cv2.imshow("debug", np.hstack([v_stacked_1, v_stacked_2]))

			print("op1: " + str(op1))
			print("op2: " + str(op2))

			cv2.waitKey(0)

		if(abs(op1 - op2) < 0.015 and op1 < 1 and op2 < 1):
			return (op1 + op2) / 2

		return min(op1, op2)

def calculate_average_value(image, circle_center, circle_radius, ignore=[0,0,0]):
	# hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	circle_img = np.zeros((image.shape[0],image.shape[1]), np.uint8)
	cv2.circle(circle_img , circle_center, circle_radius, (255,255,255), 20)
	sample = cv2.bitwise_and(image, image, mask=circle_img)
	# cv2.imshow("dog", sample)
	# cv2.waitKey(0)
	# display_image(sample)
	rgb = cv2.mean(image, circle_img)[::-1]
	return [int(x) for x in rgb[1:]]

def main():
	print("started")
	for x in range(2, 13):
		if(x == 7):
			continue
		filename = str(x) + truth
		# print(filename)
		img = cv2.imread(filename)
		digit_good_contour = get_digit_contour(img)
		print(filename)
		if(x < 10):
			if(x == 6 or x == 8):
				RED_ONE_DIGIT_CONTOURS[x] = digit_good_contour
			else:
				ONE_DIGIT_CONTOURS[x] = digit_good_contour
		else:
			TWO_DIGIT_CONTOURS[x] = digit_good_contour
		cv2.drawContours(img, digit_good_contour, -1, (0, 255, 0))
		#display_image(img)



	# for option in range(7, 10):
	# 	option = 8
	image = cv2.imread("test8.jpg")
	image = imutils.resize(image, height=1024)
	im_out = normalize_board_orientation(image)
	# im_out = normalized(im_out)
	display_image(im_out)

	number_results = []
	data = {}

	x, y, width, height = calculate_bounding_box(im_out)
	for ctr in HEXAGON_CENTERS:
		center = (x + int(ctr[0] * (width / BASE_WIDTH)), y + int(ctr[1] * (height / BASE_HEIGHT)))
		radius = int(RADIUS * ((width * height) / TEST_AREA))
		pt, confidence = calculate_point_value(im_out, center, int(radius * 1.25))

		avg = calculate_average_value(im_out, center, radius)
		# print(avg)
		cv2.circle(im_out, center, 5, (255, 0, 0), -1)
		min_dist = 2147483647
		min_dist_idx = -1
		for z in range(len(RESOURCES)):
			dist = color_diff(avg, RESOURCES[z])
			if(dist < min_dist):
				min_dist = dist
				min_dist_idx = z
		resource_name = NAMES[min_dist_idx]
		if(resource_name == "grain" and min_dist > 4):
			resource_name = "desert"

		number_results.append((pt, confidence))
		data[(pt, confidence)] = (center, int(radius * 1.25))
		print((pt, confidence))
		cv2.putText(im_out, f"{resource_name} on {pt}", center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
		# print((center, resource_name, pt))
	display_image(im_out)
	# break
if __name__ == "__main__":
    main()
