# import keyboard
import numpy as np
import cv2 as cv
import time

cap = cv.VideoCapture('basicvideo1.mp4')
# cap = cv.VideoCapture(1)

background_color = ((50, 50, 50), (180, 255, 255))

# samples = np.loadtxt('C:\\Users\\alesh\\Desktop\\prac\\generalsamples.data', np.float32)
# responses = np.loadtxt('C:\\Users\\alesh\\Desktop\\prac\\generalresponses.data', np.float32)
samples = np.loadtxt('samples_pool.data', np.float32)
responses = np.loadtxt('response_pool.data', np.float32)
responses = responses.reshape((responses.size, 1))

model = cv.ml.KNearest_create()
model.train(samples, cv.ml.ROW_SAMPLE, responses)


BLACK = ((0, 0, 0), (180, 255, 70)) #70 // 40
COLORS = ((0, 150, 15), (180, 255, 255)) # 145 // 

frames = []

def fig(num):
    if num == 1:
        return "Triangle"
    elif num == 2:
        return "Plant"
    elif num == 3:
        return "Circle"
    elif num == 4:
        return "Fish"
    return "Square"


def dnn(img, x_diff, y_diff):
    global frame, depth
    figures = []
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    colors_mask = cv.inRange(hsv, COLORS[0], COLORS[1])
    contours, h1 = cv.findContours(colors_mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE) 
    max1 = 0
    for cont in contours:
        a1 = cv.contourArea(cont)
        x, y, w, h = cv.boundingRect(cont)
        if x + w != 640 and 350<a1<4000 and abs(w - h) < 30 and w * h / a1 > 0.8:
            max1 = a1
            roi = colors_mask[y:y + h, x:x + w]
            roismall = cv.resize(roi, (10, 10))
            roismall = roismall.reshape((1, 100))
            roismall = np.float32(roismall)
            retval, results, neigh_resp, dists = model.findNearest(roismall, k=1)
            string = fig(int((results[0][0])))
            color = color_detect(hsv[y + h // 2: y + h // 2 + 1, x + w // 2: x + w // 2 + 1])
            cv.rectangle(frame, (x + x_diff, y + y_diff), (x + w + x_diff, y + h + y_diff), (255, 0, 0), 2)
            cv.putText(frame, string, (x + x_diff, y + y_diff + h), 0, 0.7, (255, 255, 255))
            figures.append((color, string, cont))
    cntf, clas = classifier(figures)
    x_cntf, y_cntf, w_cntf, h_cntf = cv.boundingRect(cntf)
    cv.rectangle(res, (x_cntf + x_diff, y_cntf + y_diff), (x_cntf + w_cntf + x_diff, y_cntf + h_cntf + y_diff), (255, 0, 255), 2)
    cv.putText(res, "Type: " + clas, (20, 40), 0, 1, (255, 255, 255))
    cv.putText(res, "Depth: " + str(depth), (20, 80), 0, 1, (255, 255, 255))
    # cv.imshow('dnn', colors_mask)

def color_detect(img):
    h, s, v = cv.split(img)
    # print(h[0][0])
    if h[0][0] < 20:
        return 5
    elif h[0][0] < 40:
        return 4
    elif h[0][0] < 90:
        return 3
    elif h[0][0] < 150:
        return 2
    return 0
    
def classifier(values):
    fig = [i[1] for i in values]
    color = [i[0] for i in values]
    level = 'ERROR'
    if fig.count('Fish') >= 3:
        level = 'Icefish'
    elif color.count(3) == 4:
        level = 'Phytoplankton'
    elif fig.count('Plant') >= 3:
        level = 'Plants'
    else:
        level = "Shellfish"
    for i in values:
        if level == 'Icefish' and i[1] != 'Fish':
            return i[2], level
        elif level == 'Phytoplankton' and i[0] != 3:
            return i[2], level
        elif level == 'Plants' and i[1] != 'Plant':
            return i[2], level
        elif level == 'Shellfish' and (i[1] != 'Circle' or (i[0] == 2 or i[0] == 3)):
            return i[2], level
    return 0, 0
    

def check_depth(img):
    global frame
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)[150:400, 120:420]
    _, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)
    contours, h1 = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE) 
    max1 = 0
    summ = 0
    for cont in contours:
        area = cv.contourArea(cont)
        summ += area
    # cv.imshow('11', thresh)
    return round(summ * 1.4 / 85000, 2)
    # cv.imshow('11', thresh)

def clamp(a,a1,a2):
    if a>=a2: return a2
    if a<=a1: return a1
    return a

def contrast(img):
    # lab = cv.cvtColor(frame, cv.COLOR_BGR2LAB)
    # l_channel, a, b = cv.split(lab)
    # # Applying CLAHE to L-channel
    # # feel free to try different values for the limit and grid size:
    # clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # cl = clahe.apply(l_channel)
    # # merge the CLAHE enhanced L-channel with the a and b channel
    # limg = cv.merge((cl,a,b))
    # # Converting image from LAB Color model to BGR color spcae
    # enhanced_img = cv.cvtColor(limg, cv.COLOR_LAB2BGR)
    # # Stacking the original image with the enhanced image
    # # result = np.hstack((img, enhanced_img))
    alpha = 1.5
    beta = 40
    return cv.convertScaleAbs(img, alpha=alpha, beta=beta)

offset = 10
state = 3
index = 1
time_start = time.time()
time_depth = time.time()
ideal_frame = np.zeros((640, 320))

while(cap.isOpened()):
    if state == 1:
        ret, frame = cap.read()
        frame=contrast(frame)
        img = frame
        res = img.copy()
        clear = frame.copy()
        img_grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        _, tresh = cv.threshold(img_grey, 40, 225, cv.THRESH_BINARY_INV)
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        black_mask = cv.inRange(hsv, BLACK[0], BLACK[1])
        contours, h1 = cv.findContours(black_mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE) 
        max1 = 0
        check_depth(img)
        for cont in contours:
            a1 = cv.contourArea(cont)
            x, y, w, h = cv.boundingRect(cont)
            if x + w != 640 and w / h > 4 and a1 > 300 and a1 / (w * h) > 0.2 and a1 > max1:
                max1 = a1
                y1, y2, x1, x2 = y - int(w * 0.95)  - offset, y + offset, x - offset, x + w + offset
                dnn_frame = frame[clamp(y1, 0, 480):clamp(y2, 0, 480), clamp(x1, 0, 640):clamp(x2, 0, 640)].copy()
                ideal_frame = dnn_frame
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                try:
                    dnn(dnn_frame, x1, y1)
                except cv.error:
                    pass


        cv.imshow('contours', frame)
        cv.imshow("alien", res)
        cv.imshow("clear", clear)
        # cv.imshow("mask", black_mask)
    elif state == 2:
        cv.putText(res, "Time: " + str(round(time.time() - time_start, 2)), (20, 120), 0, 1, (255, 255, 255))
        cv.imshow('Alien', res)
        cv.imwrite("C:\\Users\\alesh\\Desktop\\NTO\\" + str(index) + '.jpg', res)

        index += 1
        state = 3
        time_depth = time.time()
    elif state == 3:
        ret, frame = cap.read()
        frame = contrast(frame)
        depth = check_depth(frame)
        cv.imshow('contours', frame)
        if time.time() - time_depth > 10:
            state = 1
            print("state = 1")


    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    # while cv.waitKey(1) & 0xFF != ord('a'):
    #     pass
    if cv.waitKey(1) & 0xFF == ord('1'):
        state = 2

    # time.sleep(0.005)

cap.release()
cv.destroyAllWindows()