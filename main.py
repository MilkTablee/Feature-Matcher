'''
THIS ASSIGNMENT WAS DONE ON PyCharm WITH PYTHON VERSION 3.6.6 AND THE PACKAGE
opencv-contrib-python VERSION 3.4.2.16
'''
import cv2


# Function to show image
def cv2_imshow(text, image):
    cv2.imshow(text, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Function to convert image to grayscale
def grayScale(image):
    output = image.copy()
    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    return gray


# Read images
img1 = cv2.imread('./victoria.jpg')
img2 = cv2.imread('./victoria2.jpg')

# Convert images to grayscale
gray1 = grayScale(img1)
gray2 = grayScale(img2)


def bfMatcher(image1, keypoints1, descriptors1, image2, keypoints2, descriptors2):
    # BFMatcher with default parameters
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    img_result = cv2.drawMatchesKnn(image1, keypoints1, image2, keypoints2, good, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    return img_result


def bfMatcherSift():
    # Keypoint matching using bf matcher with SIFT
    # Create SIFT object
    sift = cv2.xfeatures2d.SIFT_create()

    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    img_result = bfMatcher(gray1, keypoints1, descriptors1, gray2, keypoints2, descriptors2)
    #cv2.imwrite('SIFTbfm.jpg', img_result)
    cv2_imshow('Feature Method - SIFT', img_result)


def bfMatcherSurf():
    # Keypoint matching using bf matcher with SURF
    # Create SURF Feature Detector object
    hessianThreshold = 500
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold)

    # Only features with hessian larger than hessianThreshold are retained by the detector
    keypoints1, descriptors1 = surf.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = surf.detectAndCompute(gray2, None)

    img_result = bfMatcher(gray1, keypoints1, descriptors1, gray2, keypoints2, descriptors2)
    #cv2.imwrite('SURFbfm.jpg', img_result)
    cv2_imshow('Feature Method - SURF', img_result)


def bfMatcherOrb():
    # Keypoint matching using bf matcher with ORB
    # Create ORB Feature Detector object
    orb = cv2.ORB_create()

    # Determine key points
    keypoints1 = orb.detect(gray1, None)
    keypoints2 = orb.detect(gray2, None)

    # Get the descriptors
    keypoints1, descriptors1 = orb.compute(gray1, keypoints1)
    keypoints2, descriptors2 = orb.compute(gray2, keypoints2)

    img_result = bfMatcher(gray1, keypoints1, descriptors1, gray2, keypoints2, descriptors2)
    #cv2.imwrite('ORBbfm.jpg', img_result)
    cv2_imshow('Feature Method - ORB', img_result)


def main():
    # Get user input on which Feature Detector to use
    options = ["A", "B", "C"]
    userInput = ""

    while userInput.upper() not in options:
        print("\nFeature Detectors available: A: SIFT, B: SURF, C: ORB")
        userInput = input("Please select a Feature Detector: (A/B/C) ")
    if userInput.upper() == "A":
        bfMatcherSift()
    elif userInput.upper() == "B":
        bfMatcherSurf()
    elif userInput.upper() == "C":
        bfMatcherOrb()


main()
