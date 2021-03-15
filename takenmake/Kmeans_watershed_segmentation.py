def image_segmentation(image_file):
    """This function takes the input of a single image
    of the inside of a fridge. It performs k-means and 
    the watershed algorithm to segment the image into
    individual images of each food item within the fridge
    and returns these images as the variable all_foods"""

    from skimage import io
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2
    import os
    print('IMAGE SEG')
    print(os.getcwd())
    try:
        os.chdir(r"static/uploads") 
    except:
        pass
    print(os.getcwd())

    # Read in the image and make a copy for future use
    print('in segmentation:',  image_file)
    whole_fridge = io.imread(image_file)
    image = cv2.imread(image_file)

    # Change color to RGB (from BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Reshaping the image into a 2D array of pixels and 3 color values (RGB)
    pixel_vals = image.reshape((-1, 3))
    # Convert to float type
    pixel_vals = np.float32(pixel_vals)

    # the below line of code defines the criteria for the algorithm to stop running,
    # which will happen is 100 iterations are run or the epsilon (which is the required accuracy)
    # becomes 85%
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)

    # then perform k-means clustering wit h number of clusters defined as 2
    # also random centres are initally chosed for k-means clustering
    k = 2  # using 2 allows us to separate food from fridge backgound
    retval, labels, centers = cv2.kmeans(
        pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # convert data into 8-bit values
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]

    # reshape data into the original image dimensions
    segmented_image = segmented_data.reshape((image.shape))

    # use watershed algorigthm on the kmeans image output
    img = segmented_image  
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # get for sure background area using dilation
    sure_bg = cv2.dilate(opening, kernel, iterations=20)

    # Usually watershed uses erosion to get the sure forground. but what if we use
    # dilation but just less than what's used to find the sure background

    # "sure" foreground area
    sure_fg = cv2.dilate(opening, kernel, iterations=2)

    # Finding unknown region- region btwn sure foreground and background
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]

    # now we want to get the markers as individual images
    # getting mask with connectComponents

    all_foods = []

    for marker in range(1, ret):
        mask = np.array(markers, dtype=np.uint8)

        Num_pixels_for_marker = np.sum(mask == marker)
        # take masks only larger than 0.3% of the total image size
        if Num_pixels_for_marker > (0.003*mask.shape[0]*mask.shape[1]):

            indices = np.where(mask == marker)
            all_foods.append(whole_fridge[np.min(indices[0]):np.max(
                indices[0]), np.min(indices[1]):np.max(indices[1])])

    return all_foods
