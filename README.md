$Introduction:

In this project, I explored the applications of visual interfaces in augmented reality (AR)
systems by creating a demo for an augmented reality game inspired by first person shooters
(FPS). First person shooter games generally involve multiple players navigating through a
computer generated environment in first person perspective. To score a point, a player must align
a reticle on their screen over the model of another player while simultaneously registering an
input (typically a mouse click or button press). The game I have developed is a reimagining of
the FPS genre within an augmented reality context. Instead of being computer generated, the
three dimensional environment that the players navigate through is the real world; each player’s
perspective and reticle is defined using the imagery captured by a camera on one of their devices.

The full version of the game would play out as follows. Multiple players would move
around a bounded environment while holding their phones in front of them. Each player would
be running a mobile application, with an interface resembling that of a typical camera
application. Notably, there will be a reticle and a button overlaid onto the display (a display
which depicts the real world, as captured by the phone’s camera); if a player aligns the reticle on
their device over the body of another player while pressing the button, they will deal damage to
that player. If a player takes enough damage, they are eliminated from the game.

The most novel and challenging aspect of implementing an augmented reality FPS is
designing a system that can recognize when a shot has been successfully landed. There are two
problems we must solve when attempting to build such a system, neither of which are trivial.
First, we must be able to detect whether or not the shot has hit a human (iow. if the reticle is
aligned over an image resembling a human). Then, if we have concluded that the shot has hit a
human, we must be able to determine whether or not that human is playing the game. For the
purposes of this project, I developed a simplified demo of the game involving a player and a
target, in order to evaluate the hit detection system that I built. In this demo, a user can simulate
the game from a single-player perspective, by taking shots at a human who has been labeled with
a colored symbol, designated as the target. In return, the system will inform the player whether 
their shot missed, hit a bystander, or successfully hit their opponent. The user is given the option
to analyze their last shot taken, in order to help them better understand why the hit detection
pipeline might be failing. In addition, the user is able to designate a new target opponent, by
calibrating the system to a different colored symbol.

Through deep-learning based instance segmentation, my system was able to reliably
detect whether or not the reticle was aligned over an image resembling a human. There are many
possible ways to go about the next step, determining whether or not that human is our target. My
approach involved domain engineering; a colored square would be taped onto the human
designated as the opponent. If the system has determined that the shot hit a human, it would then
analyze the body of that human and scan for the colored square by evaluating color and shape
similarity. More details will be included in the implementation section.

My image description system was written using Python. The application was created and
tested using my M1 MacBook Pro (which runs on macOS V entura). For ease of use, I connected
my computer to a lightweight webcam, which was used to capture all imagery. I used the
PyAutoGUI and PySimpleGUI libraries to implement the controller for my application. I used
the OpenCV library to display a live feed from the webcam and save images to my computer.
The OpenCV library was also used in the hit detection system, alongside the numpy library. A
prebuilt deep-learning model was loaded in and used for instance segmentation. Finally, the
matplotlib library was used to display several visualizations to the user.

Implementation:

STEP 0 — User Interface:
The first thing that I needed to implement was an interface that allows a user to view a
live camera feed and take pictures. I used cv2.VideoCapture() and cv2.imshow() to capture and
display a live camera feed. Then, I integrated source code from a public github repository1 to
create a simple controller for the application using the PyAutoGUI and PySimpleGUI libraries. I
linked the controller with an object of the HitDetector class, which would be responsible for
detecting whether or not a shot has successfully hit the target opponent. The implementation of
the HitDetector will be discussed in the following sections. Finally, I used an object of
PySimpleGUI’s Text class to display a prompt that describes the result of said shot.

STEP 1 — Mask R-CNN Instance Segmentation:
As discussed in the introduction, the hit detection system I am attempting to implement
follows a two step process. First, the system must be able to evaluate whether or not the reticle is
positioned over an image that resembles a human. In order to accomplish this, I used a
deep-learning instance segmentation model; more specifically, I loaded in a prebuilt Mask
R-CNN sourced from an online resource3 and adapted some of the source code.
Instance segmentation is an object detection algorithm used to categorize each pixel of an
image into a set of classes. In instance segmentation, unlike semantic segmentation, the separate
instances of each class are differentiated. The current, state of the art instance segmentation
algorithms rely on deep-learning approaches, typically involving convolutional neural nets.
Convolutional neural nets are used for image analysis and are composed of convolutional layers,
pooling layers, linear layers, and activation functions. Mask R-CNN is a R-CNN, a form of
convolutional network that is region-based, making it well suited for object detection. In a
R-CNN, multiple region proposals are generated, corresponding to potential objects; the
convolutional network is then applied separately on all Regions of Interest (ROI), allowing
different ROIs to be classified independently. R-CNN has two outputs for each detected object: a
class and a bounding-box. Mask R-CNN adds one additional output, producing a binary mask for 
each detected object. To be exact, Mask R-CNN is a Faster R-CNN, a type of R-CNN that has
been optimized for computational efficiency.

The shoot() method of the HitDetector class takes in an image and uses a Mask R-CNN
to predict bounding boxes, object classes, and binary masks for each object detected. First,
cv2.dnn.blobFromImage() was used to perform some preprocessing on the image matrix. The
resulting blob was passed as an input to the Mask R-CNN that was loaded in. In return, the
network detects any objects and outputs the corresponding bounding boxes, object classes, and
binary masks. From there, the system iterates through each output tuple: if the object is identified
as a human, the system will draw in the associated binary mask and bounding box, until
eventually, two images are produced. One is a combined binary mask of all the humans detected;
the other is a version of the original image with all of the relevant bounding boxes drawn in. The
system then checks to see whether the reticle is positioned over a human, by checking the
corresponding location in the combined binary mask. If the reticle is not positioned over a
human, then the shoot() returns 0 as its first output, indicating that the shot has missed.
Otherwise, if a hit has been landed, the system has to determine whether the human is the target,
and moves on to step 2. Step 1 did not require much fine-tuning; by relying on the power of the
segmentation masks produced by the Mask R-CNN, I was able to get reliable results.

Step 2 — Color Distance and Shape Detection:
As discussed in the introduction, I am using domain engineering to designate the target
opponent, by taping a colored square to their shirt. To detect the square, the system must know
what color it is; therefore, I implemented a calibration feature that allows a user to hover their
reticle over a color and designate that color as the ID. In order to determine if the human that was
shot was the target, the system analyzes the region defined by that person’s bounding box,
searching for the colored square. First, the system calculates the color distance between the
pixels of the region and the color ID; all pixels with a color distance less than some fine-tuned
threshold (< 750) are set to 1 and all other pixels are set to 0, creating a color mask.
Evaluating color similarity is not a trivial task. At first, I calculated color distance based
on the Euclidean distance between the RGB values of the two colors. While this worked to some
extent, the results I was getting were not good enough. In order to improve my color masks, I
attempted to evaluate color distance in a way that better matches human perception. I tried
translating the RGB values into the CIELAB color space and then evaluating Euclidean distance,
but to no avail. I also tried, unsuccessfully, to evaluate color distance by translating into the HSV
color space and comparing hue values. Eventually, I stumbled across a variant of the RGB
Euclidean distance formula that was weighted to better match human perception for color
(shown below).2 This definition of color distance gave me the most accurate color masks and
was the one I used in my final implementation.

If the colored square was present, it should have passed through the color mask. I used
cv2.findContours() to find the contours of the color mask and then iterated through them,
identifying the contour with the largest area. Then, I used cv2.boundingRect() to find the
rectangle bounding that contour; I divided the area of the contour by the area of the bounding
rectangle, as a measure of rectangularity. I also divided the larger side of the bounding rectangle
by the smaller side, as a measure of squareness. The rectangularity and squareness metrics would
then be compared to some fine tuned thresholds (rectangularity > 0.5 and squareness < 1.5) to
determine whether or not the colored square was present. If the square was determined to be
present, shoot() returns 1 as its first output, indicating that the shot has hit the target opponent.
Otherwise, shoot() returns 2, to indicate that a bystander has been hit.

Overall Pipeline:

First, the user positions their reticle over the colored square and presses calibrate to store
its color as an ID. Then, the user presses shoot and an image is taken. The image is passed to the
shoot() method, which detects whether or not the shot hit a human through Mask R-CNN
instance segmentation. If a human was hit, a color mask is generated from the region defined by
the corresponding bounding box, so that only pixels with colors similar to the ID pass through.
The color mask is then analyzed for any contours resembling a square; if one is identified, then
the system concludes a shot has been landed. Otherwise, the system will report that the shot has
hit a bystander. If no human was hit, the color mask will not be generated and the system will
report that the shot missed.

You may notice that the thresholds for the color distance, rectangularity, and squareness
are relatively lenient. These thresholds were fine-tuned through experimentation. If we were
identifying the target based on color or shape alone, we might want to make these thresholds
more strict. But since we are taking both color and shape into account when evaluating the target,
each individual threshold does not have to be as stringent.

Evaluation:
In order to evaluate my system, I took a series of shots and compared the expected status
(shot missed, shot hit, bystander hit) with the status predicted by the hit detector. I gathered 20
shots that missed, 20 shots that hit, and 20 shots that hit a bystander, identifying any
discrepancies between my expectations and the hit detector’s predictions. In doing so, I hoped to
uncover any unaccounted edge cases and/or deficiencies in my system. The results of my
evaluation have been reported below.

All 20 missed shots were correctly identified as missed shots, for an accuracy of 100%. This
result demonstrates the reliability of our instance segmentation algorithm. In addition, most of
the segmentation maps appear to be consistent with the original images, which is a good sign.

Out of 20 bystander hits, 18 were correctly recognized as bystander hits and 2 were incorrectly
recognized as missed shots, for an accuracy of 90%. I tested the bystander hits on individuals
standing up, sitting down, facing towards the camera, and facing away from the camera, to see
how instance segmentation would perform in these different cases. For the most part, the Mask
R-CNN instance segmentation algorithm worked well, although occasionally there were
segmentation maps produced with holes that should not be there. The two bystander hits that
were incorrectly recognized as missed shots had segmentation maps with these unwanted holes
in them. Our results further demonstrate the robustness of our instance segmentation algorithm.
In addition, the fact that no bystander hits were incorrectly recognized as target hits is reassuring
— it suggests that the color mask approach we are using is suitably selective.

Out of 20 target hits, 15 were correctly recognized as target hits, 3 were incorrectly recognized as
bystander hits, and 2 were labeled as misses, for an accuracy of 75%. Although this was worse
than the accuracy for the bystander hits and misses, it is still relatively good. I tested the
bystander hits on individuals standing up, sitting down, facing towards the camera, and facing
away from the camera, to test the robustness of our instance separation algorithm. While it
performed relatively well, the algorithm still made some mistakes, incorrectly recognizing two
target hits as misses. Our results also suggest that the color mask approach is working; we can
see a clear square pattern in the color masks of our hits. Nevertheless, the approach would
benefit from some finetuning; three target hits are still incorrectly recognized as bystander hits.

Conclusion:
Overall, I was really happy with the performance of my hit detection system. Out of 60
shots, 53 were labeled correctly, for a total accuracy of 88.33%. In addition, the visualizations
produced are consistent with my expectations, which suggests that the instance segmentation and
color mask algorithms are working fairly well.

Nevertheless, there are aspects of the hit detection system that I want to improve. The
system still struggles to assess color similarity in low light environments. I think it would be
worth exploring color spaces like HSV and CIELAB further, in an attempt to evaluate color
distance in a way that better matches human perception. It would also be worth investigating
color spaces that are robust to changes in light and shadow.

Throughout this project, I learned a lot about implementing deep-learning models. At
first, it was quite daunting; I was not familiar with instance segmentation and had never been
tasked with implementing a state of the art model before. Nevertheless, I found it really
satisfying to learn about the mechanisms behind Mask R-CNN and apply my newfound
knowledge to design a hit detection system. I also learned about the difficulties in assessing color
similarity, by experiencing them first hand. I researched multiple different metrics used to assess
color distance and tested each out, until I found one that worked well for my system.
Now that the hit detection system has been implemented, it is trivial to implement the full
game. All that is left to do is to translate the demo into a multiplayer context, by hosting a
network that allows the devices of multiple players to communicate with one another.
The code used for this assignment was quite long; it has been included in the .zip file of
my submission.

References:
[1] Akpythonyt, A. (2020, December 20). Camera.py. GitHub.
https://github.com/akpythonyt/AKpythoncodes/blob/main/Camera.py
[2] Antoniadis, P . (2022, November 4). How to Compute the Similarity of Colors. Baeldung.
https://www.baeldung.com/cs/compute-similarity-of-colours
[3] Canu, S. (2021, May 18). Instance Segmentation Mask R-CNN. Pysource.
https://pysource.com/2021/05/18/instance- segmentation-mask-r-cnn-with-python-and-opencv/
[4] “Instance Segmentation Algorithms Overview.” ReasonField Lab,
https://www.reasonfieldlab.com/post/instance-segmentation-algorithms-overview.
[5] Jacobs, D.W., et al. “Comparing Images under V ariable Illumination.” Proceedings. 1998
IEEE Computer Society Conference on Computer Vision and Pattern Recognition (Cat.
No.98CB36231), June 1998, https://doi.org/10.1109/cvpr.1998.698668.
22
[6] Kandekar, A. (2020, July 28). mask-RCNN: Segmentation and object detection for google
RVC 2020. GitHub. https://github.com/kandekar007/MASK-RCNN
[7] Khandelwal, Renu. “Computer Vision: Instance Segmentation with Mask R-CNN.” Medium,
27 Nov. 2019, towardsdatascience.com/computer-vision-instance-segmentation-with-mask -r-
cnn-7983502fcad1.
[8] Lundgren, Jonathan. Implementation and Evaluation of Hit Registration in Networked First
Person Shooters. 2021. Linköping University, Master of Science Thesis.
[9] Peddie, Jon. “Overview of Augmented Reality System Organization.” Augmented Reality,
2017, pp. 53–58., https://doi.org/10.1007/978-3-319-54502-8_4.
