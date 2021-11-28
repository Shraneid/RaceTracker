import time
import math
import cv2
import mss
import numpy as np

lowWhite = np.array([0,0,0])
highWhite = np.array([30,55,255])

initialised = False

center_left = (0,0)
center_right = (0,0)
old_lines = []

def get_points(lines):
    rho = lines[i][0][0]
    theta = lines[i][0][1]
    a = math.cos(theta)
    b = math.sin(theta)
    x0 = a * rho
    y0 = b * rho
    pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
    pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
    return pt1, pt2

def slope(pt1, pt2):
    if pt2[0]-pt1[0] != 0:
        return -(pt2[1]-pt1[1])/(pt2[0]-pt1[0])
    return 0

def difference(line, pt1, pt2):
    old_slope = slope(line[0],line[1])
    new_slope = slope(pt1, pt2)
    x_diff = abs(abs(line[0][0])-abs(pt1[0]))
    slope_diff = abs(abs(old_slope)-abs(new_slope))
    return ((x_diff * 5) + (slope_diff * 20))
    
with mss.mss() as sct:
    monitor = {"top": 240, "left": 20, "width": 660, "height": 190}

    while "Screen capturing":
        last_time = time.time()
        img = np.array(sct.grab(monitor))
        hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        threshold = cv2.inRange(hsv,lowWhite,highWhite)

        '''
        edges = cv2.Canny(threshold,100,200)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilation = cv2.dilate(edges, kernel, iterations = 1)
        kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        erosion = cv2.erode(dilation, kernel1, iterations = 1)

        kernel = np.ones((3,3),np.uint8)        
        morph = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
        '''
        
        lines = cv2.HoughLines(threshold, 1, np.pi / 180, 50, None, 0, 0)
        
        line_image = np.copy(img) * 0
        
        if lines is not None:
            if not initialised:
                old_lines = []
                high_slope = 0
                low_slope = 0
                
                for i in range(len(lines)):
                    rho = lines[i][0][0]
                    theta = lines[i][0][1]
                    a = math.cos(theta)
                    b = math.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                    pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))

                    slopeV = slope(pt1,pt2)
                    
                    if slopeV > high_slope:
                        high_slope = slopeV
                    elif slopeV < low_slope:
                        low_slope = slopeV
                
                for i in range(0, len(lines)):
                    rho = lines[i][0][0]
                    theta = lines[i][0][1]
                    a = math.cos(theta)
                    b = math.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                    pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))

                    slopeV = slope(pt1,pt2)
                    if (slopeV < high_slope and slopeV > low_slope):
                        pass
                    elif len(old_lines) < 2:
                        if len(old_lines) == 1 and difference(old_lines[0], pt1,pt2) > 10:
                            cv2.line(line_image, pt1, pt2, (0,0,255), 3)
                            old_lines.append((pt1,pt2))
                        elif len(old_lines) == 0:
                            cv2.line(line_image, pt1, pt2, (0,0,255), 3)
                            old_lines.append((pt1,pt2))
                
            else:
                new_lines = []
                for j in range(2):
                    error = 10000
                    actual_line = old_lines[j]
                    #print("actual {0}\n".format(actual_line))
                    #print(len(lines))
                    for i in range(len(lines)):
                        rho = lines[i][0][0]
                        theta = lines[i][0][1]
                        a = math.cos(theta)
                        b = math.sin(theta)
                        x0 = a * rho
                        y0 = b * rho
                        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                        
                        if difference(actual_line, pt1, pt2) < error:
                            #print ("line : {0}, new one : {1}".format(actual_line, (pt1,pt2)))
                            new_line = (pt1,pt2)
                            error = difference(actual_line, pt1, pt2)
                    print(error)
                    #time.sleep(0.5)
                    if error > 20:
                        new_line = old_lines[j]
                        #print("new line not found !")
                    #else:
                        #print("new line found !")
                        
                    new_lines.append(new_line)
                    
                #print(new_lines)
                #time.sleep(1)
                for k in range(2):
                    cv2.line(line_image, new_lines[k][0], new_lines[k][1], (0,0,255), 3)
                    
                old_lines = new_lines
                                            
                
        cv2.imshow("RACETRACKER", threshold)
        cv2.imshow("TEST", line_image)

        #print("fps: {0}".format(1 / (time.time() - last_time)))

        '''if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break'''
        if cv2.waitKey(1) & 0xFF == ord("s"):
            initialised = True
            print("Initialisation Done")


            
