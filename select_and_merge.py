import os
import string
import cv2
import numpy as np
from panoramaGenerator import ImageStitching
from ConnectImagetoSqlite import Database


def filename2number(filename):
	"""
	example: "8.jpg" -> 8
	"""
	return int(string.split(filename, '.')[0])

def number2filename(number):
	"""
	example: 8 -> "8.0.jpg"
	"""
	# note: making it integer would make it more straightforward
	return str(number) + ".jpg"

def test_number2filename():
	print number2filename(8) == "8.jpg" 
	print number2filename(100) == "100.jpg" 

# print test_number2filename()
def get_center_degree(num_array):
	return np.mean(num_array)

def test_get_middle():
	print "Expected 4: Got ", get_center_degree([0, 2, 4, 6, 8])
	print "Expected 3: Got ", get_center_degree([0, 2, 4, 6])

test_get_middle()
class SelectAndMerge(object):
	"""docstring for ClassName"""
	def __init__(self,pose_x,pose_y):
		self.pose_x = pose_x
		self.pose_y = pose_y
		self.raw_path = 'street_view_images/raw/{x},{y}'.format(x=pose_x, y=pose_y)
		self.save_path = 'street_view_images/final/iter8' # remove hardcoding later
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

			batch_numbers = [self.sorted_raw_int[i] for i in batch]
			image_fns = map(number2filename, batch_numbers)
			image_paths = [os.path.join(self.raw_path, fn) for fn in image_fns]

			im_st = ImageStitching(image_paths)
			panorama = im_st.run() # call panorama stiching algorithm
			print type(panorama)
			fn = '{x},{y},{theta}.jpg'\
				.format(x=self.pose_x, y=self.pose_y, theta=get_center_degree(batch_numbers))
			cv2.imwrite(os.path.join(self.save_path, fn), panorama)

	def readImageBatch(self):
		patch_counter = 0
		patch_number = 0
		for filename in self.sorted_raw_int:
			path = os.path.join(self.raw_path, str(filename)+".0.jpg")
			print path
			img = cv2.imread(path)

	def run(self):
			self.generatePatchBatches()
			self.feedToStitching()
			Database (self.save_path).run()

			# if img != None:	
			# 	cv2.imshow("test", img)
	  #       	cv2.waitKey(5)
	        
sm = SelectAndMerge(0.0, -0.0)
sm.run()
# # merge
# folderName = "street_view_images/working"
# imageName = "image"
# patchNumber = 5
# imageStiching = ImageStiching(folderName, imageName, patchNumber)
# imageStiching.run()

# panorama.ImageStitching