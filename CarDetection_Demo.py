import numpy as np
import cv2   as cv
import os.path as path
import pickle
import glob
from matplotlib import pyplot as plt

from matplotlib import pyplot as plt

#Global Constant Variables
RUN_DEMO = True         #Run Application based on Test Files in REL_PATH
RUN_LIVE = not RUN_DEMO  #Run Application LIVE using the default web-camera option

CAR_CASCADE = 'cars.xml' #Harr Cascade xml file for object detection

NORMAL_MODE = 0 #No Pre-processing required
CLOUDY_MODE = 1 #Increase Brightness & Contrast of image
SUNNY_MODE  = 2 #Decrease Brightness & Contrast of image
NIGHT_MODE  = 3 #Filter Ambient Light Sources

#For Demo
REL_PATH = 'TestClips\\'  #Relative Sub-Directory
FILE_CNT = 4 #Number of Test Files to be Analysed
VID_NAME = ['TestClip1.mp4','TestClip2.mp4','TestClip3.mp4','TestClip4.mp4']
VID_MODE = [CLOUDY_MODE,SUNNY_MODE,NIGHT_MODE,NIGHT_MODE]
RF_90DCW = [False,False,True,True] #Rotate Frame 90 Degrees Clockwise

#For Live
STREAM_SOURCE = 0 #Default Web-camera
VIDEO_PREPROC = NORMAL_MODE#CLOUDY_MODE #SUNNY_MODE #NIGHT_MODE #NORMAL_MODE
ROTATEF_90DCW = False #Rotate Frame 90 Degrees Clockwise

#Processing Variables - Can be changed to produce different outcome

    # *** WARNING ONLY FOR PROS *** #

#Lane Detection Threshold
lower_yellow = np.array([20, 100, 100], dtype = "uint8")  # Lowest  Yellow value defined in BGR Color Space
upper_yellow = np.array([30, 255, 255], dtype = "uint8")  # Highest Yellow value defined in BGR Color Space
lower_white = 200
upper_white = 255

#Funcion Deceleration
def preprocess_normal(video_frame):
    return video_frame
def preprocess_sunny (video_frame):
    #Conver Image to grey scaled Image
    img = cv.cvtColor(video_frame, cv.COLOR_BGR2GRAY)
    #Blur Image
    blur = cv.GaussianBlur(img, (5, 5), 0)
    # find normalized_histogram, and its cumulative distribution function
    hist = cv.calcHist([blur], [0], None, [256], [0, 256])
    hist_norm = hist.ravel() / hist.max()
    Q = hist_norm.cumsum()
    bins = np.arange(256)
    fn_min = np.inf
    thresh = -1
    for i in range(1, 256):
        p1, p2 = np.hsplit(hist_norm, [i])  # probabilities
        q1, q2 = Q[i], Q[255] - Q[i]  # cum sum of classes
        b1, b2 = np.hsplit(bins, [i])  # weights
        # finding means and variances
        m1, m2 = np.sum(p1 * b1) / q1, np.sum(p2 * b2) / q2
        v1, v2 = np.sum(((b1 - m1) ** 2) * p1) / q1, np.sum(((b2 - m2) ** 2) * p2) / q2
        # calculates the minimization function
        fn = v1 * q1 + v2 * q2
        if fn < fn_min:
            fn_min = fn
            thresh = i
    # find otsu's threshold value with OpenCV function
    ret, otsu = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    cv.imshow("otsu",otsu)
    return otsu
def preprocess_cloudy(video_frame):
    frame_h   = np.size(video_frame,0)
    frame_w   = np.size(video_frame,1)

    top_left  = (int(frame_w * 0.0), int(frame_h * 0.0))
    cen_left  = (int(frame_w * 0.0), int(frame_h * 0.5))
   #bot_left  = (int(frame_w * 0.0), int(frame_h * 0.8))
   #top_right = (int(frame_w * 1.0), int(frame_h * 0.0))
    cen_right = (int(frame_w * 1.0), int(frame_h * 0.5))
    bot_right = (int(frame_w * 1.0), int(frame_h * 0.8))

    top_half  = video_frame[top_left[1] : cen_right[1], top_left[0] : cen_right[0]]
    bot_half  = video_frame[cen_left[1]: bot_right[1], cen_left[0]: bot_right[0]]

    top_hsv   = cv.cvtColor(top_half, cv.COLOR_BGR2HSV)
    bot_hsv   = cv.cvtColor(bot_half, cv.COLOR_BGR2HSV)

    ht,st,vt,_= np.uint8(cv.mean(top_hsv))
    hb,sb,vb,_= np.uint8(cv.mean(bot_hsv))

    #cv.imshow(   "Top Half of Frame",top_half)
    #cv.imshow("Bottom Half of Frame",bot_half)

    if vt > vb:
        v = vb
    elif vb > vt:
        v = vt
    else:
        v = 0

    threshold = 150

    if  threshold > v:
        brightness_ = threshold -  v
    else:
        brightness_ = threshold

    # Increase Brightness On image
    brightness_v = brightness_
    hsv_frame = cv.cvtColor(video_frame, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv_frame)
    max_limit = 255 - brightness_v
    v[v >= brightness_] = 255
    v[v <= brightness_] = 0
    v[v > max_limit] = 255
    final_hsv = cv.merge((h, s, v))
    new_image = cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)

    return new_image
def preprocess_night (video_frame):
    # -----Reading the image-----------------------------------------------------    
    #cv.imshow("img", video_frame)

    # -----Converting image to LAB Color model-----------------------------------
    lab = cv.cvtColor(video_frame, cv.COLOR_BGR2LAB)
    #cv.imshow("lab", lab)

    # -----Splitting the LAB image to different channels-------------------------
    l, a, b = cv.split(lab)
    #cv.imshow('l_channel', l)
    #cv.imshow('a_channel', a)
    #cv.imshow('b_channel', b)

    # -----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv.createCLAHE(clipLimit = 3.0, tileGridSize=(5, 5))
    cl = clahe.apply(l)
    #cv.imshow('CLAHE output', cl)

    # -----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv.merge((cl, a, b))
    #cv.imshow('limg', limg)

    # -----Converting image from LAB Color model to RGB model--------------------
    final = cv.cvtColor(limg, cv.COLOR_LAB2BGR)
    #cv.imshow('final', final)
    return final

def lazy_calibration(func):
    """
    Decorator for calibration function to avoid re-computing calibration every time.
    """
    calibration_cache = 'CameraCal/calibration_data.pickle'

    def wrapper(*args, **kwargs):
        if path.exists(calibration_cache):
            print('Loading cached camera calibration...', end=' ')
            with open(calibration_cache, 'rb') as dump_file:
                calibration = pickle.load(dump_file)
        else:
            print('Computing camera calibration...', end=' ')
            calibration = func(*args, **kwargs)
            with open(calibration_cache, 'wb') as dump_file:
                pickle.dump(calibration, dump_file)
        print('Done.')
        return calibration

    return wrapper
@lazy_calibration
def calibrate_camera(calib_images_dir, verbose=False):
    """
    Calibrate the camera given a directory containing calibration chessboards.

    :param calib_images_dir: directory containing chessboard frames
    :param verbose: if True, draw and show chessboard corners
    :return: calibration parameters
    """

    assert path.exists(calib_images_dir), '"{}" must exist and contain calibration images.'.format(calib_images_dir)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(path.join(calib_images_dir, 'calibration*.jpg'))

    # Step through the list and search for chessboard corners
    for filename in images:

        img = cv.imread(filename)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chessboard corners
        pattern_found, corners = cv.findChessboardCorners(gray, (9, 6), None)

        if pattern_found is True:
            objpoints.append(objp)
            imgpoints.append(corners)

            if verbose:
                # Draw and display the corners
                img = cv.drawChessboardCorners(img, (9, 6), corners, pattern_found)
                cv.imshow('img',img)
                cv.waitKey(500)

    if verbose:
        cv.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return ret, mtx, dist, rvecs, tvecs
def undistort(frame, mtx, dist, verbose=False):
    """
    Undistort a frame given camera matrix and distortion coefficients.
    :param frame: input frame
    :param mtx: camera matrix
    :param dist: distortion coefficients
    :param verbose: if True, show frame before/after distortion correction
    :return: undistorted frame
    """
    frame_undistorted = cv.undistort(frame, mtx, dist, newCameraMatrix=mtx)
    '''
    if verbose:
        fig, ax = plt.subplots(nrows=1, ncols=2)
        ax[0].imshow(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
        ax[1].imshow(cv.cvtColor(frame_undistorted, cv.COLOR_BGR2RGB))
        plt.show()
    '''
    return frame_undistorted

def preproccess_frame(frame,mode):

    if mode == NORMAL_MODE:
        color_img = preprocess_normal(frame)
        grey_img = cv.cvtColor(color_img, cv.COLOR_BGR2GRAY)
    elif mode == CLOUDY_MODE:
        color_img = preprocess_cloudy(frame)
        grey_img = cv.cvtColor(color_img, cv.COLOR_BGR2GRAY)
    elif mode == SUNNY_MODE:
        color_img = frame
        grey_img = preprocess_sunny(frame)
    elif mode == NIGHT_MODE:
        color_img = preprocess_night(frame)
        grey_img = cv.cvtColor(color_img, cv.COLOR_BGR2GRAY)
    else:
        color_img = frame
        grey_img = cv.cvtColor(color_img, cv.COLOR_BGR2GRAY)

    return color_img , grey_img

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    try:
        return cv.addWeighted(initial_img, α, img, β, λ)
    except:
        return img
def mse         (imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err
def get_slope   (x1,y1,x2,y2):
    return (y2-y1)/(x2-x1)
cache = 1
first_frame = 1
def draw_lines  (img, lines, color=[51, 204, 51], thickness=6):
    """workflow:
    1) examine each individual line returned by hough & determine if it's in left or right lane by its slope
    because we are working "upside down" with the array, the left lane will have a negative slope and right positive
    2) track extrema
    3) compute averages
    4) solve for b intercept
    5) use extrema to solve for points
    6) smooth frames and cache
    """
    global cache
    global first_frame

    #first_frame = 1;

    y_global_min = img.shape[0]  # min will be the "highest" y value, or point down the road away from car
    y_max = img.shape[0]
    l_slope, r_slope = [], []
    l_lane, r_lane = [], []
    det_slope = 0.4
    α = 0.2
    # i got this alpha value off of the forums for the weighting between frames.
    # i understand what it does, but i dont understand where it comes from
    # much like some of the parameters in the hough function
    try:
        for line in lines:
            # 1
            for x1, y1, x2, y2 in line:
                slope = get_slope(x1, y1, x2, y2)
                if slope > det_slope:
                    r_slope.append(slope)
                    r_lane.append(line)
                elif slope < -det_slope:
                    l_slope.append(slope)
                    l_lane.append(line)
            # 2
            y_global_min = min(y1, y2, y_global_min)

        # to prevent errors in challenge video from dividing by zero
        if ((len(l_lane) == 0) or (len(r_lane) == 0)):
            print ('no lane detected')
            return 1

        l_slope = [x for x in l_slope if ~(x == float("inf") or x == float("-inf"))]
        r_slope = [x for x in r_slope if ~(x == float("inf") or x == float("-inf"))]

        # 3
        l_slope_mean = np.mean(l_slope, axis=0)
        r_slope_mean = np.mean(r_slope, axis=0)
        l_mean = np.mean(np.array(l_lane), axis=0)
        r_mean = np.mean(np.array(r_lane), axis=0)

        if ((r_slope_mean == 0) or (l_slope_mean == 0)):
            print('dividing by zero')
            return 1

        if ((r_slope_mean == "NaN") or (l_slope_mean == "NaN")):
            print('infinite slope')
            return 1

        # 4, y=mx+b -> b = y -mx
        l_b = l_mean[0][1] - (l_slope_mean * l_mean[0][0])
        r_b = r_mean[0][1] - (r_slope_mean * r_mean[0][0])

        if ((l_b == "NaN") or (l_b == "NaN")):
           return 1

        if ((r_mean == "NaN") or (l_mean == "NaN")):
            return 1

        # 5, using y-extrema (#2), b intercept (#4), and slope (#3) solve for x using y=mx+b
        # x = (y-b)/m
        # these 4 points are our two lines that we will pass to the draw function



        l_x1 = int((y_global_min - l_b) / l_slope_mean)
        l_x2 = int((y_max - l_b) / l_slope_mean)
        r_x1 = int((y_global_min - r_b) / r_slope_mean)
        r_x2 = int((y_max - r_b) / r_slope_mean)

        # 6
        if l_x1 > r_x1:
            l_x1 = int((l_x1 + r_x1) / 2)
            r_x1 = l_x1
            l_y1 = int((l_slope_mean * l_x1) + l_b)
            r_y1 = int((r_slope_mean * r_x1) + r_b)
            l_y2 = int((l_slope_mean * l_x2) + l_b)
            r_y2 = int((r_slope_mean * r_x2) + r_b)
        else:
            l_y1 = y_global_min
            l_y2 = y_max
            r_y1 = y_global_min
            r_y2 = y_max

        current_frame = np.array([l_x1, l_y1, l_x2, l_y2, r_x1, r_y1, r_x2, r_y2], dtype="float32")

        if first_frame == 1:
            next_frame = current_frame
            first_frame = 0
        else:
            prev_frame = cache
            next_frame = (1 - α) * prev_frame + α * current_frame

        cv.line(img, (int(next_frame[0]), int(next_frame[1])), (int(next_frame[2]), int(next_frame[3])), color, thickness)
        cv.line(img, (int(next_frame[4]), int(next_frame[5])), (int(next_frame[6]), int(next_frame[7])), color, thickness)

        cache = next_frame
        return 0
    except:
        print('Error')
        return 1

def detect_lanes(video_frame,gray_frame):

    mask_yellow = cv.inRange   (video_frame, lower_yellow, upper_yellow)
    mask_white  = cv.inRange   (gray_frame , lower_white , upper_white )
    mask_yw     = cv.bitwise_or(mask_white , mask_yellow)

    # Lane Image with combined color space mask applied
    mask_yw_image = cv.bitwise_and(gray_frame, mask_yw)

    kernel_size = 5
    gauss_gray = cv.GaussianBlur(mask_yw_image, (kernel_size, kernel_size), 0)

    low_threshold  = 50
    high_threshold = 150
    canny_edges = cv.Canny(gauss_gray, low_threshold, high_threshold)


    # rho and theta are the distance and angular resolution of the grid in Hough space
    # same values as quiz
    rho = 4
    theta = np.pi / 180
    # threshold is minimum number of intersections in a grid for candidate line to go to output
    threshold = 30
    min_line_len = 50#100
    max_line_gap = 90#180

    hough_lines = cv.HoughLinesP(canny_edges, rho, theta, threshold, np.array([]), minLineLength=min_line_len,maxLineGap=max_line_gap)
    line_image = np.zeros((canny_edges.shape[0], canny_edges.shape[1], 3), dtype="uint8")
    notdrawn = draw_lines(line_image, hough_lines)
    if notdrawn:
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(line_image, 'Error : No Lane Detected', (0, 30), font, 1, (0, 0, 255), 2, cv.LINE_AA)
    '''
    else :
        left_half, right_half = split_frame(line_image,axis = 0)

        cv.imshow("Video Capture Final   left_half", left_half)
        cv.imshow("Video Capture Final   right_half", right_half)

        histr = cv.calcHist([left_half], [1], None, [256], [0, 256])
        plt.plot(histr, color='g')
        plt.xlim([0, 256])
        plt.show()

        histr = cv.calcHist([right_half], [1], None, [256], [0, 256])
        plt.plot(histr, color='g')
        plt.xlim([0, 256])
        plt.show()

        histr = cv.calcHist([line_image], [1], None, [256], [0, 256])
        plt.plot(histr, color='g')
        plt.xlim([0, 256])
        plt.show()
    '''
    #histogram = np.sum(canny_edges[canny_edges.shape[0] // 2:, :], axis=0)
    #out_img = np.dstack((canny_edges, canny_edges, canny_edges)) * 255

    cv.imshow("Video Capture Pre-processed", video_frame)
    cv.imshow("Video Capture yw img", mask_yw_image)
    cv.imshow("Video Capture Gray Blur", gauss_gray)
    cv.imshow("Video Capture Canny Edge", canny_edges)
    cv.imshow("Video Capture line img", line_image)
    return line_image
def detect_cars (video_frame,classifier_file):

    imgClassifier = cv.CascadeClassifier(classifier_file)

    # detect cars using haar cascade
    imgs = imgClassifier.detectMultiScale(video_frame, 1.1, 2)

    nimgs = 0
    for (x, y, w, h) in imgs:
        cv.rectangle(video_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    nimgs = nimgs + 1

    return video_frame

def split_frame(video_frame, axis = 0):
    frame_h = np.size(video_frame, 0)
    frame_w = np.size(video_frame, 1)

    if axis == 0:
        top_left  = (int(frame_w * 0.0), int(frame_h * 0.0))
        cen_left  = (int(frame_w * 0.0), int(frame_h * 0.5))
        bot_left  = (int(frame_w * 0.0), int(frame_h * 1.0))
        top_right = (int(frame_w * 1.0), int(frame_h * 0.0))
        cen_right = (int(frame_w * 1.0), int(frame_h * 0.5))
        bot_right = (int(frame_w * 1.0), int(frame_h * 1.0))

        top_frame = video_frame[top_left[1]: cen_right[1], top_left[0]: cen_right[0]]
        bot_frame = video_frame[cen_left[1]: bot_right[1], cen_left[0]: bot_right[0]]

        #recombine = np.concatenate((top_frame,bot_frame), axis=0)
        #cv.imshow("ROI Top Section of Frame", top_frame)
        #cv.imshow("ROI Bottom Section of Frame", bot_frame)
        #cv.imshow("ROI Full Section of Frame", recombine)
        return top_frame,bot_frame
    else:
        top_left  = (int(frame_w * 0.0), int(frame_h * 0.0))
        top_cen   = (int(frame_w * 0.5), int(frame_h * 0.0))
        top_right = (int(frame_w * 1.0), int(frame_h * 0.0))

        bot_left  = (int(frame_w * 0.0), int(frame_h * 1.0))
        bot_cen   = (int(frame_w * 0.5), int(frame_h * 1.0))
        bot_right = (int(frame_w * 1.0), int(frame_h * 1.0))

        left_frame  = video_frame[top_left[1]: bot_cen  [1], top_left[0]: bot_cen  [0]]
        right_frame = video_frame[top_cen [1]: bot_right[1], top_cen [0]: bot_right[0]]

        # recombine = np.concatenate((top_frame,bot_frame), axis=0)
        # cv.imshow("ROI Top Section of Frame", top_frame)
        # cv.imshow("ROI Bottom Section of Frame", bot_frame)
        # cv.imshow("ROI Full Section of Frame", recombine)
        return left_frame, right_frame

def recombine_frame(top_half,bot_half,axis = 0):
    return np.concatenate((top_half,bot_half), axis=axis)
def apply_lane_roi(video_frame):
    frame_h = np.size(video_frame, 0)
    frame_w = np.size(video_frame, 1)

    t = 0
    c = 0.65
    b = 0.85

    top_left  = (int(frame_w * 0.0), int(frame_h * t))
    cen_left  = (int(frame_w * 0.0), int(frame_h * c))
    bot_left  = (int(frame_w * 0.0), int(frame_h * b))
    top_right = (int(frame_w * 1.0), int(frame_h * t))
    cen_right = (int(frame_w * 1.0), int(frame_h * c))
    bot_right = (int(frame_w * 1.0), int(frame_h * b))

    hexa_top_l = (int(frame_w * 0.05), int(frame_h * c))
    hexa_top_r = (int(frame_w * 0.95), int(frame_h * c))
    hexa_bot_l = (int(frame_w * 0.00), int(frame_h * (b - (b - c)/2)))
    hexa_bot_r = (int(frame_w * 1.00), int(frame_h * (b - (b - c)/2)))

    roi_mask     = np.zeros(video_frame.shape, dtype="uint8")
    roi_vertices = np.array([[hexa_top_l, hexa_top_r, hexa_bot_r, bot_right,bot_left,hexa_bot_l]], dtype="int32")
    #roi_vertices = np.array([[cen_left, cen_right, bot_right, bot_left]], dtype="int32")

    cv.fillPoly(roi_mask, roi_vertices, (255,255, 255))
    masked_image = cv.bitwise_and(video_frame, roi_mask)

    '''
    top_half = masked_image[top_left[1]: cen_right[1], top_left[0]: cen_right[0]]
    bot_half = masked_image[cen_left[1]: bot_right[1], cen_left[0]: bot_right[0]]

    recombine = np.concatenate((top_half , bot_half), axis = 0)
    cv.imshow("ROI    Top Half of Frame", top_half)
    cv.imshow("ROI Bottom Half of Frame", bot_half)
    cv.imshow("ROI recomc Half of Frame", recombine)
    '''
    return masked_image
def apply_objs_roi(video_frame):
    #Not Implemented
    return video_frame

def run_app(source_file,mode,rotate_frame):

    #Define Video Capture from Source File
    video_capture = cv.VideoCapture(source_file)

    ret, mtx, dist, rvecs, tvecs = calibrate_camera(calib_images_dir='camera_cal')
    #Grab Video Frame if Video Steam has successfully been opened
    if video_capture.isOpened():
        valid, video_frame = video_capture.read()
    else:
        valid = False

    while valid:
        video_frame = undistort(video_frame, mtx, dist, verbose=False)
        #Rotate Frame 90 Degrees Clockwise if required
        if rotate_frame:
            video_frame = cv.rotate(video_frame, cv.ROTATE_90_CLOCKWISE)

        #top_frame,cen_frame,bot_frame = split_frame(video_frame)

        #Pre-process Video Frame based on current capture lighting
        color_frame , grey_frame = preproccess_frame(video_frame,mode)
        # Apply Region of interest for lane detection
        roi_frame_c = apply_lane_roi(color_frame)
        roi_frame_g = apply_lane_roi(grey_frame)
        #Detect car lanes inside defined region of interest
        line_image = detect_lanes(roi_frame_c,roi_frame_g)

        #Detect Cars in original video frame
        top_half, bot_half = split_frame(video_frame)
        new_b_half = detect_cars(bot_half,CAR_CASCADE)
        cv.imshow("Car detection new Image", new_b_half)
        img_detect = recombine_frame(top_half, new_b_half)

        #Draw Detected lanes over original video frame
        result_frame = weighted_img(line_image, img_detect, α=0.8, β=1., λ=0.)

        #cv.imshow("Video Capture Orignal Image", video_frame )
        cv.imshow("Video Capture Final   Image", result_frame)

        valid, video_frame = video_capture.read()
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release() # Close video stream
    return valid

if RUN_DEMO:
    for i in range(0,FILE_CNT):
        run_app(REL_PATH + VID_NAME[i],VID_MODE[i],RF_90DCW[i])
elif RUN_LIVE:
    run_app(STREAM_SOURCE, VIDEO_PREPROC, ROTATEF_90DCW)


