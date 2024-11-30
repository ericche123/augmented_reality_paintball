import pyautogui
import PySimpleGUI as sg
import cv2
import numpy as np
import matplotlib.pyplot as plt

#variant of RGB Euclidean distance that is weighted to better fit human perception
#formula adapted from online resource: https://www.baeldung.com/cs/compute-similarity-of-colours
def color_distance(color0, color1):
    b0, g0, r0 = float(color0[0]), float(color0[1]), float(color0[2])
    b1, g1, r1 = float(color1[0]), float(color1[1]), float(color1[2])

    distance = 0.3 * (r1 - r0) * (r1 - r0) + 0.59 * (g1 - g0) * (g1 - g0) + 0.11 * (b1 - b0) * (b1 - b0) 
    return distance

class HitDetector:
    #hit detector class is responsible for determining whether a shot missed, hit its target, or hit a bystander

    def __init__(self):
        #load in prebuilt Mask R-CNN for instance segmentation
        #code adapted from online resource: https://pysource.com/2021/05/18/instance-segmentation-mask-r-cnn-with-python-and-opencv/ 
        self.net = cv2.dnn.readNetFromTensorflow("dnn/frozen_inference_graph_coco.pb", "dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")
        self.color_id = None

    def calibrate(self, img):
        #given image matrix, set color ID to color of pixel that reticle is pointing to 
        height, width, depth = img.shape
        reticle = ((height // 2), (width // 2))

        self.color_id = img[reticle[0]][reticle[1]]

    def shoot(self, img): 
        #given image matrix, determine whether a shot missed, hit its target, or hit a bystander
        height, width, depth = img.shape
        reticle = ((height // 2), (width // 2))

        # create black image
        black_image = np.zeros((height, width, 3), np.uint8)

        # perform preprocessing on image matrix to get blob
        #provide blob as input for Mask R-CNN
        blob = cv2.dnn.blobFromImage(img, swapRB=True)
        self.net.setInput(blob)

        #recieve outputs of Mask-RCNN
        boxes, masks = self.net.forward(["detection_out_final", "detection_masks"])
        detection_count = boxes.shape[2]

        hitboxes = []

        for i in range(detection_count):
            #iterate through outputs
            box = boxes[0, 0, i]
            class_id = box[1]
            score = box[2]

            if score < 0.5 or class_id != 0:
                #if score is too low, skip
                #class_id of 0 corresponds to human; if detected object is NOT a human, skip
                continue

            #get coordinates of bounding box
            ul = (int(box[3] * width), int(box[4] * height))
            lr = (int(box[5] * width), int(box[6] * height))

            #add bounding box to list of bounding boxes
            hitboxes.append((ul, lr))

            #define region of interest
            roi = black_image[ul[1] : lr[1], ul[0] : lr[0]]
            roi_height, roi_width, roi_depth = roi.shape

            #get masks, based on dimensions
            mask = masks[i, int(class_id)]
            mask = cv2.resize(mask, (roi_width, roi_height))
            _, mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)

            #draw bounding box
            cv2.rectangle(img, ul, lr, (255, 0, 0), 3)

            #get mask coordinates, draw mask
            contours, _ = cv2.findContours(np.array(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cv2.fillPoly(roi, [cnt], (255, 255, 255))

        #STEP 1: Determine if reticle is aligned over a human body
        #check black_image, which represents the combined binary mask of all the humans detected
        #if the position at the reticle is equal to [255, 255, 255], that means that the reticle is aligned over an image detected to be a human
        if np.array_equal(black_image[reticle[0]][reticle[1]], [255, 255, 255]):
            #if this is the case then we must move on to step 2
            #we know that reticle is aligned over a human body
            #STEP 2: Determine if that human body is the target

            #iterate through list of bounding boxes, hitboxes, looking for the one corresponding to the human that was hit
            hitbox_index = -1
            for i in range(len(hitboxes)):
                ul = hitboxes[i][0]
                lr = hitboxes[i][1]

                if (reticle[1] > ul[0] and reticle[1] < lr[0]) and (reticle[0] > ul[1] and reticle[0] < lr[1]):
                    hitbox_index = i

            ul = hitboxes[hitbox_index][0]
            lr = hitboxes[hitbox_index][1]

            #create a reframed view of the image matrix that focuses on the region defined by this bounding box
            reframed = img[(ul[1]):(lr[1]+1),(ul[0]):(lr[0]+1)]
            frame_height, frame_width, frame_depth = reframed.shape

            #create color_mask using color_distance
            #for all pixels in the region, compute the color distance between that pixel and the color ID
            #if it is less than 750 (finetuned threshold), set to corresponding position to 1; otherwise, default is zero
            color_mask = np.zeros((frame_height, frame_width), dtype = np.uint8)
            
            for i in range(frame_height):
                for j in range(frame_width):
                    color_dist = color_distance(reframed[i][j], self.color_id)
                    if color_dist < 750:
                        color_mask[i][j] = 1

            contours, hierarchy = cv2.findContours(color_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

            #sometimes there are no contours found; use an if statement to prevent errors
            if len(contours) == 0:
                return 2, black_image, color_mask, None, None

            contour = max(contours, key=lambda x: cv2.contourArea(x))
            contour_area = cv2.contourArea(contour)
            x,y,w,h = cv2.boundingRect(contour) 

            #compute metric of rectangularity and squareness
            #rectangularity: compare contour area to area of bounding rectangle
            #squareness: compare longer side of bounding rectangle to shorter side
            rectangularity = contour_area / (w * h)
            squareness = max(w, h)/min(w, h)

            #thresholds finetuned through experimentation
            #thresholds for the color distance, rectangularity, and squareness are relatively lenient
            #if we were identifying the target based on color or shape alone, we might want to make these thresholds more strict
            #since we are taking both color and shape into account when evaluating the target, each individual threshold does not have to be as stringent.
            if rectangularity > 0.5 and squareness < 1.5:
                cv2.rectangle(black_image, ul, lr, (255, 0, 0), 3)

                #returning 1 signifes that shot has landed
                return 1, black_image, color_mask, rectangularity, squareness

            #returning 2 signifes that bystander was hit
            return 2, black_image, color_mask, rectangularity, squareness

        #returning 0 signifies that shot missed
        return 0, black_image, None, None, None

        #also returns black_image, color_mask, rectangularity, squareness
        #these are visualiations and metrics, returned so that they can be easily viewed by the user 

def main():

    sg.theme('Black')

    #code for controller adapted from online resource: https://github.com/akpythonyt/AKpythoncodes/blob/main/Camera.py 
    #controller has calibrate, shoot, analyze, and exit button
    #also has text prompt that can be updated

    layout = [[ sg.Button('Calibrate',size=(10,1),font='Serif 14') ], [ sg.Button('Shoot',size=(10,1),font='Serif 14') ], [ sg.Button('Analyze',size=(10,1),font='Serif 14') ], [ sg.Button('Exit',size=(10,1),font='Serif 14') ], [sg.Text("", size=(0, 1), font='Serif 14', key='output')]]
    window = sg.Window('Controller', layout)

    #set up live video feed
    cap = cv2.VideoCapture(0)

    #instantiate object of HitDetector class, to be used to label shots
    hit_detector = HitDetector()
    analysis = None
    counter = 0

    while True:
        event, values = window.read(timeout=20)

        if event == 'Calibrate':
            #call calibrate method to calibrate color ID
            hit_detector.calibrate(shot)

        if event == 'Shoot': 
            if hit_detector.color_id is None:
                raise Exception("The color ID has not been set yet!")

            hit_status, analysis, color_mask, rectangularity, squareness = hit_detector.shoot(shot)

            height, width, depth = analysis.shape

            #drawing in crosshair onto analysis visualization
            for i in range(3, 20):
                mid_h = height // 2
                mid_w = width // 2

                analysis[mid_h + i][mid_w] = [255, 0, 0]
                analysis[mid_h - i][mid_w] = [255, 0, 0]

                analysis[mid_h][mid_w + i] = [255, 0, 0]
                analysis[mid_h][mid_w - i] = [255, 0, 0]

            #no matter what the hit_status is, write the appropriate images to their corresponding directory
            if hit_status == 0:
                #shot missed
                window['output'].update(value="Shot missed!")
                cv2.imwrite(r'shots/missed/shot_{}.png'.format(counter), frame) 
                cv2.imwrite(r'shots/missed/analysis_{}.png'.format(counter), analysis)
            elif hit_status == 1:
                #shot hit
                window['output'].update(value="Shot landed!")
                cv2.imwrite(r'shots/hit/shot_{}.png'.format(counter), frame) 
                cv2.imwrite(r'shots/hit/analysis_{}.png'.format(counter), analysis) 

                plt.imshow(color_mask)
                plt.savefig(r'shots/hit/mask_{}.png'.format(counter)) 
                plt.close()
            elif hit_status == 2:
                #shot hit "civilian"
                window['output'].update(value="Bystander hit!")
                cv2.imwrite(r'shots/bystander/shot_{}.png'.format(counter), frame) 
                cv2.imwrite(r'shots/bystander/analysis_{}.png'.format(counter), analysis) 

                plt.imshow(color_mask)
                plt.savefig(r'shots/bystander/mask_{}.png'.format(counter)) 
                plt.close()

            #counter used to update filenames automatically
            counter += 1

        if event == 'Analyze':
            if analysis is None:
                raise Exception("Shot has not been taken yet!")

            #show visualization of segmentations mask, color mask (if possible), and metrics of rectangularity and squareness (if possible)
            plt.imshow(analysis)
            plt.show()
            plt.close()

            if color_mask is not None:
                plt.imshow(color_mask)
                plt.show()
                plt.close()

            if rectangularity and squareness:
                print("Rectangularity: " + str(rectangularity))
                print("squareness: " + str(squareness))

        if event == 'Exit':
            #clean up and exit
            cv2.destroyAllWindows()
            cam.release()
            return

        #read in frame from live feed
        ret, frame = cap.read()

        if ret:
            #copy frame
            shot = frame.copy()

            #drawing in crosshair onto frame
            height, width, depth = frame.shape

            for i in range(3, 20):
                mid_h = height // 2
                mid_w = width // 2

                frame[mid_h + i][mid_w] = [255, 0, 0]
                frame[mid_h - i][mid_w] = [255, 0, 0]

                frame[mid_h][mid_w + i] = [255, 0, 0]
                frame[mid_h][mid_w - i] = [255, 0, 0]

            #display live feed
            cv2.imshow('Camera', frame)
            cv2.waitKey(1)
        
if __name__ == "__main__":
    main()