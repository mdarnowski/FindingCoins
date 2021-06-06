import numpy as np
import cv2 as cv


def find_tray(image):
    img = cv.imread(image)
    # convert color space
    conv_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # remove outlier pixels
    thresh = cv.threshold(conv_img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
    # morphology
    kernel = np.ones((5, 5), np.uint8)
    morph = cv.morphologyEx(thresh, cv.MORPH_ELLIPSE, kernel, iterations=1)
    # sure background area
    sure_bg = cv.dilate(morph, kernel, iterations=20)
    # finding sure foreground area
    dist_transform = cv.distanceTransform(morph, cv.DIST_L2, 5)
    # finding unknown region
    sure_fg = np.uint8(cv.threshold(dist_transform,
                                    0.993 * dist_transform.max(), 255, 0)[1])
    unknown = cv.subtract(sure_bg, sure_fg)
    # marker labelling
    markers = cv.connectedComponents(sure_fg)[1]
    # add one to all labels so that sure background is 1 instead of 0
    markers = markers + 1
    # mark the region of unknown with 0
    markers[unknown == 255] = 0
    markers = cv.watershed(img, markers)

    return markers


def find_coins(image, markers):
    img = cv.imread(image)
    # convert color space
    conv_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # remove outlier pixels
    blur = cv.GaussianBlur(conv_img, (5, 5), 2)
    # calculate the threshold for a small regions of the image
    thresh = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv.THRESH_BINARY, 11, 3)
    # morphology
    kernel = np.ones((4, 3), np.uint8)
    k_open = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
    # detection of the edges
    outline = cv.Canny(k_open, 100, 100, L2gradient=True)
    # Hough Circles
    coins = cv.HoughCircles(outline, cv.HOUGH_GRADIENT, 1.8,
                            minDist=30, minRadius=13, maxRadius=39)
    if coins is not None:
        coins = np.round(coins[0, :]).astype("int")
        for (x, y, r) in coins:
            cv.circle(img, (x, y), r, (0, 0, 0), 6)
            if markers[y][x] == 1:
                cv.circle(img, (x, y), 5, (0, 255, 0), 5)
    else:
        print('No circles found')

    # sort
    coins = coins[np.argsort(coins[:, 2])]
    # paint fives
    cv.circle(img, (coins[-1][0], coins[-1][1]), coins[-1][2], (0, 30, 200), 3)
    cv.circle(img, (coins[-2][0], coins[-2][1]), coins[-2][2], (0, 30, 200), 3)
    # mark results of the watershed algorithm
    img[markers == -1] = [0, 255, 0]
    # write to file
    filename = 'trays_res/'+image[len(image) - 9:]
    cv.imwrite(filename, img)
    # show
    cv.imshow(filename, img)


if __name__ == '__main__':
    find_coins("trays/tray1.jpg", find_tray("trays/tray1.jpg"))
    find_coins("trays/tray2.jpg", find_tray("trays/tray2.jpg"))
    find_coins("trays/tray3.jpg", find_tray("trays/tray3.jpg"))
    find_coins("trays/tray4.jpg", find_tray("trays/tray4.jpg"))
    find_coins("trays/tray5.jpg", find_tray("trays/tray5.jpg"))
    find_coins("trays/tray6.jpg", find_tray("trays/tray6.jpg"))
    find_coins("trays/tray7.jpg", find_tray("trays/tray7.jpg"))
    find_coins("trays/tray8.jpg", find_tray("trays/tray8.jpg"))

    cv.waitKey(0)
    cv.destroyAllWindows()
