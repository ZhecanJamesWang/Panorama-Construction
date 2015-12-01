from PanoramaConstruction_TwoSidesMergingMethod import ImageStiching

import os
import string
import cv2

def string_integer_splitter(sample):
	"""
	example: "8.0.jpg" -> 8.0
	"""
	return int(string.split(sample, '.')[0])

class SelectAndMerge(object):
	"""docstring for ClassName"""
	def __init__(self):
		self.raw_path = 'street_view_images/raw'
		self.get_sorted_raw_int()
		

	def get_sorted_raw_int(self):
		self.raw = os.listdir(self.raw_path)
		raw_int = map(string_integer_splitter, self.raw)
		self.sorted_raw_int = sorted(raw_int)
		return self.sorted_raw_int

	def foo(self):
		patch_counter = 0
		patch_number = 0
		for filename in self.sorted_raw_int:
			path = os.path.join(self.raw_path, str(filename)+".0.jpg")
			print path
			img = cv2.imread(path)
			# if img != None:	
			# 	cv2.imshow("test", img)
	  #       	cv2.waitKey(5)
	        

SelectAndMerge().foo()
	# # merge
# folderName = "street_view_images/working"
# imageName = "image"
# patchNumber = 5
# imageStiching = ImageStiching(folderName, imageName, patchNumber)
# imageStiching.run()

# panorama.ImageStitching