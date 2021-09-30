import cv2
import numpy as np
 
def grey(image):
    image = np.asarray(image)
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

  
def gauss(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

  
def canny(image):
    edges = cv2.Canny(image,50,150)
    return edges
def region(image):
    height, width = image.shape

    triangle = np.array([
                       [(100, height), (475, 325), (width, height)]
                       ])
    mask = np.zeros_like(image)
    mask = cv2.fillPoly(mask, triangle, 255)
    mask = cv2.bitwise_and(image, mask)
    return mask

def display_lines(image, lines):
    lines_image = np.zeros_like(image)
    #make sure array isn't empty
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line
            #draw lines on a black image
            cv2.line(lines_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return lines_image
def average(image, lines):
    left = []
    right = []
    for line in lines:
        print(line)
        x1, y1, x2, y2 = line.reshape(4)
        #fit line to points, return slope and y-int
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        print(parameters)
        slope = parameters[0]
        y_int = parameters[1]
        #lines on the right have positive slope, and lines on the left have neg slope
        if slope < 0:
            left.append((slope, y_int))
        else:
            right.append((slope, y_int))
    #takes average among all the columns (column0: slope, column1: y_int)
    right_avg = np.average(right, axis=0)
    left_avg = np.average(left, axis=0)
    #create lines based on averages calculates
    left_line = make_points(image, left_avg)
    right_line = make_points(image, right_avg)
    return np.array([left_line, right_line])
def make_points(image, average):
    print(average)
    slope, y_int = average
    y1 = image.shape[0]
    y2 = int(y1 * (3/5))
    #determine algebraically
    x1 = int((y1 - y_int) // slope)
    x2 = int((y2 - y_int) // slope)
    return np.array([x1, y1, x2, y2])

video = "/Users/shreyashrivastava/Desktop/sample.mp4"
cap = cv2.VideoCapture(video)
print("here")
num = 0
while(cap.isOpened()): 
    _, frame = cap.read()
    print(num)
    num += 1

    gaus = gauss(frame)
    edges = cv2.Canny(gaus,50,150)
    isolated = region(edges)
    lines = cv2.HoughLinesP(isolated, 2, np.pi/180, 50, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average(frame, lines)
    black_lines = display_lines(frame, averaged_lines)
    lanes = cv2.addWeighted(frame, 0.8, black_lines, 1, 1)
    cv2.imshow("frame", lanes)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release() 
cv2.destroyAllWindows()