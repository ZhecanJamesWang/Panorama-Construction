# Panorama-Construction
Project of constructing panorama view from several images for Computational Robotics Fall 2015



#Story line 1


What’s pup! (This is where you say “not mutts”). This is James and Erika and we want to talk to you about a piece of the “Neato (Google) Street View” project: visualizing the google street view inside school campus. Our vision for the final product is to allow users to remotely visit our web interface and interact with our “Neato Google Street View”. 
    
Besides Slam, this is the another big piece of the project and mainly has three sections. First, the panorama. In order to create a “2.5 D” view, we think creating a panorama view will be useful; Second, the data storage. We need places to store all the pictures and load them to web interface; Third, the web interface. 
    
In this blog post we will be touching on the panorama portion of this project…
    
There are different kinds of panoramas varying with different angles of degrees. In our project, in order to visualise the map fully at specific waypoints we decided to explore making 360 degree panoramas using the neato. Since we don’t have a 360 degree camera built into the neato, our plan is to take many pictures and then stitch them together to form a panorama. We would then upload slices of the panorama into a database to view on an interactive web platform.
    
Stitching is usually done with needle and thread but you can also use a machine called a sewing machine. They’re pretty cool, but pretty hard to use on pictures so we decided to use our secret weapon supported by the famous guy, Paul Ruvolo ---------- programming tool, python instead. 
    
Intuitively, we chose to start stitching two images before moving on to multiple image stitching. Even though there are not so many resources online about image stitching we luckily found some resources. After reading some resources, we found that people start image stitching with finding the features in the images first. There are multiple feature detection algorithms online supported by OpenCV library like SIFT, ORB and so on. Based on our past experiences, we chose to use SURF for this part. 
    
After detecting all the features and getting the keypoints and descriptors of these features by SURF, we realized that we need to find a method to match those features. We learned online that we can use some match algorithms from OpenCV like FlannBasedMatcher. It only takes all the descriptors from the features and will produce the matched pairs of the keypoints which also identify the features.  
    
Then, we successfully got all the matched features. We were thinking about how to merge the images together based on the mathes by the proper perspectives. After searching online and getting hints from both the professors and online resources, we learned this new topic, homography. Homography gives us the transfer matrix to help us to merge the two images in the same perspective. 
    
Until here, we finally got two images stitched together. Some of the examples are as following:

![Image](https://github.com/ZhecanJamesWang/Panorama-Construction/blob/master/BlogImages/blogpic1.jpg)
[Figure 1: First of two images we’ll stitch together]

![Image](https://github.com/ZhecanJamesWang/Panorama-Construction/blob/master/BlogImages/blogpic2.jpg)
[Figure 2: Second of two images we’ll stitch together]

![Image](https://github.com/ZhecanJamesWang/Panorama-Construction/blob/master/BlogImages/blogpic3.jpg)
[Figure 3: The two images stitched together]

As in the photo, the photo are merged well. 

After stitching two images, we would like to move to multiple images. Essentially it will be the same idea. 
The following are the two methods we have explored.

The first one we tried was to stitch the first two images then stitch the new image with the third and so on [see Figure 4]. 

The second method we tried was to stitch every two images then from that new group of images stitch every two and so on until we would compile to one single image [see Figure 5]. 

![Image](https://github.com/ZhecanJamesWang/Panorama-Construction/blob/master/BlogImages/blogpic4.jpg)
[Figure 4: First Method of Image Stitching]

![Image](https://github.com/ZhecanJamesWang/Panorama-Construction/blob/master/BlogImages/blogpic5.jpg)
[Figure 5: Second Method of Image Stitching]

![Image](https://github.com/ZhecanJamesWang/Panorama-Construction/blob/master/BlogImages/blogpic6.jpg)
[Figure 6: Multiple stitched images using the first stitching method]

![Image](https://github.com/ZhecanJamesWang/Panorama-Construction/blob/master/BlogImages/blogpic7.jpg)
[Figure 7: Multiple stitched images using the second stitching method]

As the photos above, from both methods, the because of the homography transfer, as the images get stitched more and more, the output images will get more and more skewed in a nonuniform way. This problem will make the match algorithm unable to detect corresponding matches anymore since the photos are getting more unscaled. Each iteration would magnify any errors in the compiling of the image and would create large black lines in the middle of the images. Additionally the panorama would increase in size as it stitched pictures.

After running countless tests and discussing with the professor, we started to think that image stitching to generate panoramas might not be the most effective way to capture a 360 degree shot on the neato. We’ve now identified other options and will be moving forward with testing these new ideas and then will move to finding the most efficient way of storing these panoramas.


#Story Line 2


After previously failing many times in image stitching, we planned to start looking for other methods of making panorama. However after discussing with Genius Professor, Paul Ruvolo the Great, a passionate young man, Zhecan Wang in our team got inspired and found a new way of stitching images together which may solve all the problems. 

##The New stitching method (two side stitching method)
Diagram of how method works for 5 images
![Image](https://github.com/ZhecanJamesWang/Panorama-Construction/blob/master/BlogImages/Screenshot%20from%202015-12-07%2023:49:24.png)
[Figure 7: Multiple stitched images using the second stitching method]

