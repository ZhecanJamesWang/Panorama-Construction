    # coding: utf-8
import cv2, numpy as np
import os
import Image

class ImageStiching(object):
  """Stiching multiple images together"""
  def __init__(self, folderName, fileName):
    self.folderName = folderName
    self.fileName = fileName
    self.fileNumber = 0
    self.DEBUG = False
    self.DEBUG1 = False
    self.patchNumber = 5
    
    
  def extract_features_and_descriptors(self, image):

    ## Convert image to grayscale (for SURF detector).
   
    keypoints = []
    descriptors = []
    gray= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # sift = cv2.xfeatures2d.SIFT_create()
    # keypoints, descriptors = sift.detectAndCompute(image,None)

    surf = cv2.SURF(400)
    keypoints, descriptors = surf.detectAndCompute(image,None)
    ## Detect SURF features and compute descriptors.

    
    return (keypoints, descriptors)
    
  ## --------------------------------------------------------------------
  def detect_features(self, grey_image):
    surf = cv2.FeatureDetector_create("SURF")
    surf.setDouble("hessianThreshold", 1000)
    return surf.detect(grey_image)
    
  def extract_descriptors(self, grey_image, keypoints):
    surf = cv2.DescriptorExtractor_create("SURF")
    return surf.compute(grey_image, keypoints)[1]
    

  ## --------------------------------------------------------------------
  ## Find corresponding features between the images. ----------------
  def find_correspondences(self, keypoints1, descriptors1, keypoints2, descriptors2):

    ## Find corresponding features.
    match = self.match_flann(descriptors1, descriptors2)
    # print match
    # good = []
    # # good = match
    # for m,n in match:
    #     if m.distance < 0.7*n.distance:
    #         good.append(m)
    # print len(match)
    index1 = []
    index2 = []
    for indexPair in match:
      ind_1, ind_2 = indexPair
      index1.append(ind_1)
      index2.append(ind_2)

    points1 = []
    points2 = []
    # points1 = np.float32([ keypoints1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    # points2 = np.float32([ keypoints1[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    ## Look up corresponding keypoints.



    for kp1 in keypoints1:
      points1.append(kp1.pt)
    for kp2 in keypoints2:
      points2.append(kp2.pt)

    newPoints1 = []
    newPoints2 = []

    for ind in index1:
      newPoints1.append(points1[ind])
    for ind in index2:
      newPoints2.append(points2[ind])

    return (newPoints1, newPoints2)


  ## ---------------------------------------------------------------------
  ##  Calculate the size and offset of the stitched panorama. --------
  def calculate_size(self, size_image1, size_image2, homography):
    
    # ## Calculate the size and offset of the stitched panorama.
    # ## (Overwrite the following 2 lines with your answer.)

    (h1, w1) = size_image1[:2]
    (h2, w2) = size_image2[:2]

    #remap the coordinates of the projected image onto the panorama image space
    top_left = np.dot(homography,np.asarray([0,0,1]))
    top_right = np.dot(homography,np.asarray([w2,0,1]))
    bottom_left = np.dot(homography,np.asarray([0,h2,1]))
    bottom_right = np.dot(homography,np.asarray([w2,h2,1]))

    if self.DEBUG:
      print top_left
      print top_right
      print bottom_left
      print bottom_right

    #normalize
    top_left = top_left/top_left[2]
    top_right = top_right/top_right[2]
    bottom_left = bottom_left/bottom_left[2]
    bottom_right = bottom_right/bottom_right[2]

    if self.DEBUG:
      print np.int32(top_left)
      print np.int32(top_right)
      print np.int32(bottom_left)
      print np.int32(bottom_right)

    pano_left = int(min(top_left[0], bottom_left[0], 0))
    pano_right = int(max(top_right[0], bottom_right[0], w1))
    W = pano_right - pano_left

    pano_top = int(min(top_left[1], top_right[1], 0))
    pano_bottom = int(max(bottom_left[1], bottom_right[1], h1))
    H = pano_bottom - pano_top

    size = (W, H)

    if self.DEBUG:
      print 'Panodimensions'
      print pano_top
      print pano_bottom

    # offset of first image relative to panorama
    X = int(min(top_left[0], bottom_left[0], 0))
    Y = int(min(top_left[1], top_right[1], 0))
    offset = (-X, -Y)

    if self.DEBUG:
      print 'Calculated size:'
      print size
      print 'Calculated offset:'
      print offset
        
    ## Update the homography to shift by the offset
    # does offset need to be remapped to old coord space?
    # print homography
    # homography[0:2,2] += offset

    return (size, offset)


  ## ---------------------------------------------------------------------
  ##  Combine images into a panorama. --------------------------------
  def merge_images(self, image1, image2, homography, size, offset, keypoints):
    ## Combine the two images into one.
    ## (Overwrite the following 5 lines with your answer.)
    (h1, w1) = image1.shape[:2]
    (h2, w2) = image2.shape[:2]
    
    panorama = np.zeros((size[1], size[0], 3), np.uint8)
    
    (ox, oy) = offset
    
    translation = np.matrix([
      [1.0, 0.0, ox],
      [0, 1.0, oy],
      [0.0, 0.0, 1.0]
    ])
    
    if self.DEBUG:
      print homography
    homography = translation * homography
    # print homography
    
    # draw the transformed image2
    cv2.warpPerspective(image2, homography, size, panorama)
    
    panorama[oy:h1+oy, ox:ox+w1] = image1  
    # panorama[:h1, :w1] = image1  

    ## Draw the common feature keypoints.

    return panorama

    
  def place_image(self, output, image, x, y):
    minx = max(x,0)
    miny = max(y,0)
    maxx = min(x+image.shape[1],output.shape[1])
    maxy = min(y+image.shape[0],output.shape[0])
    output[miny:maxy, minx:maxx] = image[miny-y:maxy-y, minx-x:maxx-x]

  def match_flann(self, desc1, desc2, r_threshold = 0.5):
    'Finds strong corresponding features in the two given vectors.'
    ## Adapted from <http://stackoverflow.com/a/8311498/72470>.
    
    if len(desc1) == 0 or len(desc2) == 0:
      print "No features passed into match_flann"
      return []

    ## Build a kd-tree from the second feature vector.
    FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
    flann = cv2.flann_Index(desc2, {'algorithm': FLANN_INDEX_KDTREE, 'trees': 4})

    ## For each feature in desc1, find the two closest ones in desc2.
    (idx2, dist) = flann.knnSearch(desc1, 2, params={}) # bug: need empty {}

    ## Create a mask that indicates if the first-found item is sufficiently
    ## closer than the second-found, to check if the match is robust.
    mask = dist[:,0] / dist[:,1] < r_threshold
    
    ## Only return robust feature pairs.
    idx1  = np.arange(len(desc1))
    pairs = np.int32(zip(idx1, idx2[:,0]))

    return [(i,j) for (i,j) in pairs[mask]]


  def draw_correspondences(self, image1, image2, points1, points2):
    'Connects corresponding features in the two images using yellow lines.'

    ## Put images side-by-side into 'image'.
    (h1, w1) = image1.shape[:2]
    (h2, w2) = image2.shape[:2]
    image = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    image[:h1, :w1] = image1
    image[:h2, w1:w1+w2] = image2
    
    ## Draw yellow lines connecting corresponding features.
    for (x1, y1), (x2, y2) in zip(np.int32(points1), np.int32(points2)):
      cv2.line(image, (x1, y1), (x2+w1, y2), (0, 255, 255), lineType=cv2.CV_AA)

    return image

  def show(self, name, im):
    if im.dtype == np.complex128:
      raise Exception("OpenCV can't operate on complex valued images")
    cv2.namedWindow(name)
    cv2.imshow(name, im)
    cv2.waitKey(1)
   

  def mergeTwoImages(self, imageIndex, heading):
    ## Load images.
    self.heading = heading
    self.imageIndex = imageIndex
    image1 = cv2.imread(self.folderName+"/"+self.fileName+str(self.imageIndex))
    image2 = cv2.imread(self.folderName+"/"+self.fileName+str(self.imageIndex+1))
    if self.heading == "left":
      tmp = image1
      image1 = image2
      image2 = tmp

    print self.folderName+"/"+self.fileName+str(imageIndex)
    print self.folderName+"/"+self.fileName+str(imageIndex+1)
    if self.DEBUG1:
      print image1.shape
      print image2.shape

    ## Detect features and compute descriptors.
    keypoints1, descriptors1 = self.extract_features_and_descriptors(image1)
    keypoints2, descriptors2 = self.extract_features_and_descriptors(image2)
    if self.DEBUG1:
      print len(keypoints1), "features detected in image1"
      print len(keypoints2), "features detected in image2"
    
    # show("Image1 features", cv2.drawKeypoints(image1, keypoints1, color=(0,0,255)))
    # show("Image2 features", cv2.drawKeypoints(image2, keypoints2, color=(0,0,255)))
    
    ## Find corresponding features.
    points1, points2 = self.find_correspondences(keypoints1, descriptors1, keypoints2, descriptors2)
    points1 = np.array(points1, dtype=float)
    points2 = np.array(points2, dtype=float)
    if self.DEBUG1:
      print len(points1), "features matched"

    ## Visualise corresponding features.
    correspondences = self.draw_correspondences(image1, image2, points1, points2)
    # cv2.imwrite(folderName+"/correspondences.jpg", correspondences)
    # cv2.imshow('correspondences', correspondences)
    
    ## Find homography between the views.
    # if len(points1) < 4 or len(points2) < 4:
    #   print "Not enough features to find a homography"
      # homography = np.identity(3, dtype=float)
    # else:
    (homography, _) = cv2.findHomography(points2, points1, method=cv2.RANSAC)
    # homography = np.matrix(homography)
    # print "Homography = "
    # print homography
    
    ## Calculate size and offset of merged panorama.

    (size, offset) = self.calculate_size(image1.shape[:2], image2.shape[:2], homography)

    size = tuple(np.asarray(size).flatten().astype(int).tolist())
    offset = tuple(np.asarray(offset).flatten().astype(int).tolist())
    if self.DEBUG1:
      print "image1, ", image1.shape[:2], "image2: ", image2.shape[:2]
      print "output size: %ix%i" % size
    ## Finally combine images into a panorama.
    panorama = self.merge_images(image1, image2, homography, size, offset, (points1, points2))
    # cv2.imwrite(self.folderName+"/"+self.fileName+str(self.fileNumber+1)+".jpg", panorama)
    

    panorama = cv2.cvtColor(panorama,cv2.COLOR_BGR2RGB)
    panorama = Image.fromarray(panorama)
    print self.heading
    print imageIndex
    self.fileNumber = self.calculateFilesNumber()
    if self.heading == "left":
      if imageIndex == int(self.preFileNumber/2):
        panorama.save(self.folderName+"/"+self.fileName+str(self.fileNumber+1),'JPEG')
        print "output_save:", self.folderName+"/"+self.fileName+str(self.fileNumber+1)
      else:
        panorama.save(self.folderName+"/"+self.fileName+str(self.imageIndex+1),'JPEG')
        print "output_save:", self.folderName+"/"+self.fileName+str(self.imageIndex+1)
    else:
      if imageIndex == int(self.preFileNumber/2)+1:
        self.fileNumber = self.calculateFilesNumber()
        panorama.save(self.folderName+"/"+self.fileName+str(self.fileNumber+1),'JPEG')
        print "output_save:", self.folderName+"/"+self.fileName+str(self.fileNumber+1)
      elif imageIndex > self.preFileNumber:
        panorama.save(self.folderName+"/"+self.fileName+str(self.fileNumber+1),'JPEG')
        print "output_save:", self.folderName+"/"+self.fileName+str(self.fileNumber+1)
        print "---"
      else:
        panorama.save(self.folderName+"/"+self.fileName+str(self.imageIndex),'JPEG')
        print "output_save:", self.folderName+"/"+self.fileName+str(self.imageIndex)

  def calculateFilesNumber(self):
    fileNumber = 0
    for fileName in os.listdir(self.folderName+"/."):
      fileNumber +=1
    return fileNumber

  def run(self):
    # self.mergeTwoImages(6)
    self.fileNumber = self.calculateFilesNumber()
    self.preFileNumber = self.fileNumber
    if self.fileNumber%self.patchNumber != 0:
      raise Exception("The number of files cannot divided evenly by patchNumber")
    for i in range (int(self.preFileNumber/self.patchNumber)):
        for j in range (i*self.patchNumber, int(((i+1)*self.patchNumber)/2)):
          self.mergeTwoImages(j+1,"left")
        for j in range (i*self.patchNumber, int(((i+1)*self.patchNumber)/2)):
          print "j", j
          self.mergeTwoImages(self.preFileNumber - (j+1), "right")
        self.mergeTwoImages(self.preFileNumber+1, "right")

if __name__ == "__main__":
  folderName = "input14"
  imageName = "Image"
  imageStiching = ImageStiching(folderName, imageName)  
  imageStiching.run()

  # panorama = cv2.imread( folderName+"/panorama.jpg")
  # # cv2.imshow('panorama', panorama)     
  # import sys, select 
  # print "Press enter or any key on one of the images to exit"
  # while True:
  #   if cv2.waitKey(0) != -1:
  #     break
  #   # http://stackoverflow.com/questions/1335507/keyboard-input-with-timeout-in-python
  #   i, o, e = select.select( [sys.stdin], [], [], 0.1 )
  #   if i:
  #     break