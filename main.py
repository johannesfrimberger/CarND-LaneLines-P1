#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os, glob

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

############################
## Common Functions ########
############################

"""
"""
class LaneDetection:
    def __init__(self):
        """
        Set parameters and thresholds used in LaneDetection class.
        Initialize internal variables
        """

        # Parameters for visualization
        self.showRawResults = False

        # Parameters for gaussian blur
        self.kernel_size = 5

        # Threshold used for canny edge detection
        self.low_threshold = 100
        self.high_threshold = 180

        # Parameters used in hough transformation
        self.rho = 2  # distance resolution in pixels of the Hough grid
        self.theta = np.pi / 180  # angular resolution in radians of the Hough grid
        self.hough_threshold = 5  # minimum number of votes (intersections in Hough grid cell)
        self.min_line_length = 10  # minimum number of pixels making up a line
        self.max_line_gap = 2  # maximum gap in pixels between connectable line segments

        # Parameters for
        self.update_rate = 0.05

        # Reset internal storage
        self.reset()

    def reset(self):
        """
        Reset all internal storage to default values
        """
        self._imageShape = (0, 0)
        self._init = False

    def run(self, img):
        """
        Run image processing pipeline to identify lane lines.
        This method will return a annotated version of the input img.
        """

        # Update internal storage
        self._imageShape = img.shape

        # Run image processing pipeline
        imgGrayscale = self.convertToGrayscale(img)
        imgGaussian = self.runGaussianBlur(imgGrayscale)
        imgCanny = self.runCanny(imgGaussian)
        imgMasked = self.maskImage(imgCanny)
        imgHoughLines = self.runHoughLines(imgMasked)
        imgWithLanes = self.weighted_img(imgHoughLines, img)

        # Set class to initialized
        self._init = True

        return imgWithLanes
        #return cv2.cvtColor(imgMasked, cv2.COLOR_GRAY2RGB)

    def convertToGrayscale(self, img):
        """Applies the Grayscale transform
        This will return an image with only one color channel
        but NOTE: to see the returned image as grayscale
        you should call plt.imshow(gray, cmap='gray')"""
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def runGaussianBlur(self, img):
        """Applies a Gaussian Noise kernel"""
        return cv2.GaussianBlur(img, (self.kernel_size, self.kernel_size), 0)

    def runCanny(self, img):
        """Applies the Canny transform"""
        return cv2.Canny(img, self.low_threshold, self.high_threshold)

    def region_of_interest(img, vertices):
        """
        Applies an image mask.

        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        """
        # defining a blank mask to start with
        mask = np.zeros_like(img)

        # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        # filling pixels inside the polygon defined by "vertices" with the fill color
        cv2.fillPoly(mask, vertices, ignore_mask_color)

        # returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    def region_of_interest(self, img, vertices):
        """
        Applies an image mask.

        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        """
        # defining a blank mask to start with
        mask = np.zeros_like(img)

        # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        # filling pixels inside the polygon defined by "vertices" with the fill color
        cv2.fillPoly(mask, vertices, ignore_mask_color)

        # returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    def maskImage(self, img):
        """
        Mask the input image
        """

        imshape = img.shape
        leftBoundary = 115
        top1 = 320
        top2 = 300
        leftTop = 440
        rightTop = 530

        vertices = np.array([[(leftBoundary, imshape[0]), (leftTop, top1), (rightTop, top2), (imshape[1], imshape[0])]],
                            dtype=np.int32)

        return self.region_of_interest(img, vertices)

    def draw_lines(self, img, lines, color=[255, 0, 0], thickness=5):
        """
        NOTE: this is the function you might want to use as a starting point once you want to
        average/extrapolate the line segments you detect to map out the full
        extent of the lane (going from the result shown in raw-lines-example.mp4
        to that shown in P1_example.mp4).

        Think about things like separating line segments by their
        slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
        line vs. the right line.  Then, you can average the position of each of
        the lines and extrapolate to the top and bottom of the lane.

        This function draws `lines` with `color` and `thickness`.
        Lines are drawn on the image inplace (mutates the image).
        If you want to make the lines semi-transparent, think about combining
        this function with the weighted_img() function below
        """

        # Split lines in three categories (left, right, unused)
        leftLane = []
        rightLane = []

        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = ((y2 - y1) / (x2 - x1))

                # Do not use almost horizontal lines
                if abs(slope) > 0.3:
                    colorRaw = [0, 255, 0]
                    if slope < 0:
                        leftLane.append([x1, y1])
                        leftLane.append([x2, y2])
                        colorRaw = [0, 0, 255]
                    else:
                        rightLane.append([x1, y1])
                        rightLane.append([x2, y2])

                    if self.showRawResults:
                        cv2.line(img, (x1, y1), (x2, y2), colorRaw, thickness)

        shape = img.shape

        leftLane = np.array(leftLane)
        fitLeft = np.polyfit(leftLane[:, 0], leftLane[:, 1], 1)

        rightLane = np.array(rightLane)
        fitRight = np.polyfit(rightLane[:, 0], rightLane[:, 1], 1)

        # Check if LaneDetection algorithm has been initialized and update line parameters
        if self._init:
            self._fitLeft[0] = ((1.0 - self.update_rate) * self._fitLeft[0]) + (self.update_rate * fitLeft[0])
            self._fitLeft[1] = ((1.0 - self.update_rate) * self._fitLeft[1]) + (self.update_rate * fitLeft[1])
            self._fitRight[0] = ((1.0 - self.update_rate) * self._fitRight[0]) + (self.update_rate * fitRight[0])
            self._fitRight[1] = ((1.0 - self.update_rate) * self._fitRight[1]) + (self.update_rate * fitRight[1])
        else:
            self._fitLeft = fitLeft
            self._fitRight = fitRight

        # Use a more appropriate model for the lanes than a simple line

        # Sort points in leftLane and rightLane by y-axis
        #leftLane = leftLane[np.argsort(leftLane[:, 1])]
        #rightLane = rightLane[np.argsort(rightLane[:, 1])]
        #splineLeft = interpolate.splrep(leftLane[:, 1], leftLane[:, 0], s=0)
        #splineRight = interpolate.splrep(rightLane[:, 1], rightLane[:, 0], s=0)

        # These parameters give the top and bottom position of the extrapolation (Could be made adaptive)
        drawingRangeTop = 320
        drawingRangeBottom = shape[0]

        # Find top and bottom points to draw lines inbetween
        lineLeftTop = (int((drawingRangeTop - self._fitLeft[1]) / self._fitLeft[0]), drawingRangeTop)
        lineLeftBottom = (int((drawingRangeBottom - self._fitLeft[1]) / self._fitLeft[0]), drawingRangeBottom)
        lineRightTop = (int((drawingRangeTop - self._fitRight[1]) / self._fitRight[0]), drawingRangeTop)
        lineRightBottom = (int((drawingRangeBottom - self._fitRight[1]) / self._fitRight[0]), drawingRangeBottom)

        if not(self.showRawResults):
            cv2.line(img, lineLeftTop, lineLeftBottom, color, thickness)
            cv2.line(img, lineRightTop, lineRightBottom, color, thickness)

    def runHoughLines(self, img):
        """
        `img` should be the output of a Canny transform.

        Returns an image with hough lines drawn.
        """
        lines = cv2.HoughLinesP(img, self.rho, self.theta, self.hough_threshold, np.array([]),
                                minLineLength=self.min_line_length, maxLineGap=self.max_line_gap)
        line_img = np.zeros((*img.shape, 3), dtype=np.uint8)

        self.draw_lines(line_img, lines)

        return line_img

    def weighted_img(self, img, initial_img, α=0.8, β=1., λ=0.):
        """
        `img` is the output of the hough_lines(), An image with lines drawn on it.
        Should be a blank image (all black) with lines drawn on it.

        `initial_img` should be the image before any processing.

        The result image is computed as follows:

        initial_img * α + img * β + λ
        NOTE: initial_img and img must be the same shape!
        """
        return cv2.addWeighted(initial_img, α, img, β, λ)

# Not best practice but use a global variable to keep information from previous frame
# with fl_image function not taking multiple parameters
laneDetection = LaneDetection()

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # you should return the final output (image with lines are drawn on lanes)
    return laneDetection.run(image)

def main():

    # Find all images
    imageStreamFolder = "test_images"
    resultFolder = "test_images/results"
    imageDataType = "jpg"
    test_image = glob.glob(os.path.join(imageStreamFolder, "*.%s" % (imageDataType)))

    # Process each image
    for filename in test_image:
        filenameResults = (os.path.join(resultFolder, os.path.basename(filename)))
        img = mpimg.imread('%s' % (filename))

        laneDetection = LaneDetection()
        results = laneDetection.run(img)
        mpimg.imsave(filenameResults, results)


    white_output = 'white.mp4'
    clip1 = VideoFileClip("solidWhiteRight.mp4")
    white_clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)

    laneDetection.reset()
    yellow_output = 'yellow.mp4'
    clip2 = VideoFileClip('solidYellowLeft.mp4')
    yellow_clip = clip2.fl_image(process_image)
    yellow_clip.write_videofile(yellow_output, audio=False)

    laneDetection.reset()
    challenge_output = 'extra.mp4'
    clip2 = VideoFileClip('challenge.mp4')
    challenge_clip = clip2.fl_image(process_image)
    challenge_clip.write_videofile(challenge_output, audio=False)

if __name__ == "__main__":
    main()