import os
import string
import cv2
from panoramaGenerator import ImageStitching

def filename2number(filename):
	"""
	example: "8.0.jpg" -> 8.0
	"""
	return int(string.split(filename, '.')[0])

def number2filename(number):
	"""
	example: 8 -> "8.0.jpg"
	"""
	# note: making it integer would make it more straightforward
	return str(number) + ".0.jpg"

def test_number2filename():
	print number2filename(8) == "8.0.jpg" 
	print number2filename(100) == "100.0.jpg" 
# print test_number2filename()

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
		raw_int = map(filename2number, self.raw)
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
			self.indexList.append(patch_select)

	def feedToStitching(self):
		"""get images from each of the batches and feed into image stiching algorithm

		then save each panorama ... (maybe in this function)
		"""
		for batch in self.indexList:
			print batch
			image_fns = map(number2filename, [self.sorted_raw_int[i] for i in batch])
			image_paths = [os.path.join(self.raw_path, fn) for fn in image_fns]
			im_st = ImageStitching(image_paths)
			print map(type, im_st.working_images)
			panorama = im_st.run()# call panorama stiching algorithm
			print type(panorama), panorama.shape
			cv2.imshow('window', panorama)
			cv2.waitKey(0)

			break

	def readImageBatch(self, ):
		patch_counter = 0
		patch_number = 0
		for filename in self.sorted_raw_int:
			path = os.path.join(self.raw_path, str(filename)+".0.jpg")
			print path
			img = cv2.imread(path)


			# if img != None:	
			# 	cv2.imshow("test", img)
	  #       	cv2.waitKey(5)
	        
sm = SelectAndMerge()
sm.generatePatchBatches()
sm.feedToStitching()
	# # merge
# folderName = "street_view_images/working"
# imageName = "image"
# patchNumber = 5
# imageStiching = ImageStiching(folderName, imageName, patchNumber)
# imageStiching.run()

# panorama.ImageStitching