# coding: utf-8
import cv2, numpy as np
DEBUG = False
## ---------------------------------------------------------------------
## 3.1  Extract SURF keypoints and descriptors from an image. ----------
def extract_features_and_descriptors(image):

  ## Convert image to grayscale (for SURF detector).
  ## TODO: 
  keypoints = []
  descriptors = []
  gray= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
  # sift = cv2.xfeatures2d.SIFT_create()
  # keypoints, descriptors = sift.detectAndCompute(image,None)

  surf = cv2.SURF(400)
  keypoints, descriptors = surf.detectAndCompute(image,None)
  ## Detect SURF features and compute descriptors.
  ## TODO
  
  return (keypoints, descriptors)
  
## --------------------------------------------------------------------
def detect_features(grey_image):
  surf = cv2.FeatureDetector_create("SURF")
  surf.setDouble("hessianThreshold", 1000)
  return surf.detect(grey_image)
  
def extract_descriptors(grey_image, keypoints):
  surf = cv2.DescriptorExtractor_create("SURF")
  return surf.compute(grey_image, keypoints)[1]
  

## --------------------------------------------------------------------
## 3.2 Find corresponding features between the images. ----------------
def find_correspondences(keypoints1, descriptors1, keypoints2, descriptors2):

  ## Find corresponding features.
  match = match_flann(descriptors1, descriptors2)
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


  ## TODO
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
## 3.3  Calculate the size and offset of the stitched panorama. --------
def calculate_size(size_image1, size_image2, homography):
  
  # ## Calculate the size and offset of the stitched panorama.
  # ## TODO: (Overwrite the following 2 lines with your answer.)
  # offset = (10, 10)
  # size   = (1400, 800)
  

  # offset = abs((homography*(size_image2[0]-1,size_image2[1]-1,1))[0:2,2]) 
  # print offset
  # size   = (size_image1[1] + int(offset[0]), size_image1[0] + int(offset[1]))
  # if (homography*(0,0,1))[0][2] > 0:
  #   offset[0] = 0
  # if (homography*(0,0,1))[1][2] > 0:
  #   offset[1] = 0

  # ## Update the homography to shift by the offset
  # homography[0,2] += offset[0]
  # homography[1,2] += offset[1]

  # return (size, offset)

  (h1, w1) = size_image1[:2]
  (h2, w2) = size_image2[:2]

  #remap the coordinates of the projected image onto the panorama image space
  top_left = np.dot(homography,np.asarray([0,0,1]))
  top_right = np.dot(homography,np.asarray([w2,0,1]))
  bottom_left = np.dot(homography,np.asarray([0,h2,1]))
  bottom_right = np.dot(homography,np.asarray([w2,h2,1]))

  if DEBUG:
    print top_left
    print top_right
    print bottom_left
    print bottom_right

  #normalize
  top_left = top_left/top_left[2]
  top_right = top_right/top_right[2]
  bottom_left = bottom_left/bottom_left[2]
  bottom_right = bottom_right/bottom_right[2]

  if DEBUG:
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

  if DEBUG:
    print 'Panodimensions'
    print pano_top
    print pano_bottom

  # offset of first image relative to panorama
  X = int(min(top_left[0], bottom_left[0], 0))
  Y = int(min(top_left[1], top_right[1], 0))
  offset = (-X, -Y)

  if DEBUG:
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
## 3.4  Combine images into a panorama. --------------------------------
def merge_images(image1, image2, homography, size, offset, keypoints):

  # ## Combine the two images into one.
  # ## TODO
  # # panorama = np.zeros((size[1], size[0], 3), np.uint8)  
  
  # # place_image(panorama, image1, offset[0], offset[1])
  # panorama = cv2.warpPerspective(image2,homography,size)
  # (h1, w1) = image1.shape[:2]

  # for h in range(h1):
  #   for w in range(w1):
  #       if image1[h][w][0] != 0 or image1[h][w][3] != 0 or image1[h][w][4] != 0:
  #           panorama[h+offset[1]][w + offset[0]] = image1[h][w]

  # return panorama
  ## TODO: Combine the two images into one.
  ## TODO: (Overwrite the following 5 lines with your answer.)
  (h1, w1) = image1.shape[:2]
  (h2, w2) = image2.shape[:2]
  
  panorama = np.zeros((size[1], size[0], 3), np.uint8)
  
  (ox, oy) = offset
  
  translation = np.matrix([
    [1.0, 0.0, ox],
    [0, 1.0, oy],
    [0.0, 0.0, 1.0]
  ])
  
  if DEBUG:
    print homography
  homography = translation * homography
  # print homography
  
  # draw the transformed image2
  cv2.warpPerspective(image2, homography, size, panorama)
  
  panorama[oy:h1+oy, ox:ox+w1] = image1  
  # panorama[:h1, :w1] = image1  

  ## TODO: Draw the common feature keypoints.

  return panorama










##------------------------------------------------------------------------
##------------------------------------------------------------------------
##------------------------------------------------------------------------
##---- No need to change anything below this point. ----------------------
##------------------------------------------------------------------------
##------------------------------------------------------------------------

  
def place_image(output, image, x, y):
  minx = max(x,0)
  miny = max(y,0)
  maxx = min(x+image.shape[1],output.shape[1])
  maxy = min(y+image.shape[0],output.shape[0])
  output[miny:maxy, minx:maxx] = image[miny-y:maxy-y, minx-x:maxx-x]

def match_flann(desc1, desc2, r_threshold = 0.5):
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


def draw_correspondences(image1, image2, points1, points2):
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




## ---------------------------------------------------------------------

def show(name, im):
  if im.dtype == np.complex128:
    raise Exception("OpenCV can't operate on complex valued images")
  cv2.namedWindow(name)
  cv2.imshow(name, im)
  cv2.waitKey(1)
  
if __name__ == "__main__":

  ## Load images.
  image1 = cv2.imread("input/Image1.jpg")
  image2 = cv2.imread("input/Image2.jpg")

  ## Detect features and compute descriptors.
  keypoints1, descriptors1 = extract_features_and_descriptors(image1)
  keypoints2, descriptors2 = extract_features_and_descriptors(image2)
  print len(keypoints1), "features detected in image1"
  print len(keypoints2), "features detected in image2"
  
  show("Image1 features", cv2.drawKeypoints(image1, keypoints1, color=(0,0,255)))
  show("Image2 features", cv2.drawKeypoints(image2, keypoints2, color=(0,0,255)))
  
  ## Find corresponding features.
  points1, points2 = find_correspondences(keypoints1, descriptors1, keypoints2, descriptors2)
  points1 = np.array(points1, dtype=float)
  points2 = np.array(points2, dtype=float)
  print len(points1), "features matched"

  ## Visualise corresponding features.
  correspondences = draw_correspondences(image1, image2, points1, points2)
  cv2.imwrite("correspondences.jpg", correspondences)
  cv2.imshow('correspondences', correspondences)
  
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
  (size, offset) = calculate_size(image1.shape[:2], image2.shape[:2], homography)
  size = tuple(np.asarray(size).flatten().astype(int).tolist())
  offset = tuple(np.asarray(offset).flatten().astype(int).tolist())
  print "output size: %ix%i" % size
  
  ## Finally combine images into a panorama.
  panorama = merge_images(image1, image2, homography, size, offset, (points1, points2))
  cv2.imwrite("panorama.jpg", panorama)
  cv2.imshow('panorama', panorama)
  
      
  
  import sys, select 
  print "Press enter or any key on one of the images to exit"
  while True:
    if cv2.waitKey(0) != -1:
      break
    # http://stackoverflow.com/questions/1335507/keyboard-input-with-timeout-in-python
    i, o, e = select.select( [sys.stdin], [], [], 0.1 )
    if i:
      break
