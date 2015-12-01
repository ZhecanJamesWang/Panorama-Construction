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
		self.indexList = []
		self.total = len(self.sorted_raw_int)
		self.diff = 2
		self.start_offset = 2
		self.patchNumber = 5	

	def get_sorted_raw_int(self):
		self.raw = os.listdir(self.raw_path)
		raw_int = map(string_integer_splitter, self.raw)
		self.sorted_raw_int = sorted(raw_int)
		return self.sorted_raw_int

	@staticmethod
	def patch_selection(start, n_patches, diff):
		"""
		Returns the patch indexes [0, 2, 4, 6, 8]
		for start = 0, n_patches = 5, diff = 2
		"""
		return range(start, start + n_patches*diff, diff)

	def generatePatchBatches(self):

		for start in range(0, self.total - self.patchNumber, self.start_offset):
			patch_select = self.patch_selection(start, self.patchNumber, self.diff)
			print patch_select
			self.indexList.append(patch_select)

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
	        

SelectAndMerge().generatePatchBatches()
	# # merge
# folderName = "street_view_images/working"
# imageName = "image"
# patchNumber = 5
# imageStiching = ImageStiching(folderName, imageName, patchNumber)
# imageStiching.run()

# panorama.ImageStitching