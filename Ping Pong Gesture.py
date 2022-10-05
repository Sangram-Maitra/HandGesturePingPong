import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

#importing all images
imgBackground = cv2.imread("Resources/Background.png")
imgGameover = cv2.imread("Resources/gameOver.png")
imgBall = cv2.imread("Resources/Ball.png", cv2.IMREAD_UNCHANGED)
imgBat1 = cv2.imread("Resources/bat1.png", cv2.IMREAD_UNCHANGED)
imgBat2 = cv2.imread("Resources/bat2.png", cv2.IMREAD_UNCHANGED)

#Hand Detector
detector = HandDetector(detectionCon=0.8, maxHands=2)

# Variables
ballPos = [100, 100]
speedx = 10
speedy = 10
gameover = False
score = [0, 0]


while True:
    _, img = cap.read()
    img = cv2.flip(img, 1) # 1 means horizontal flip and 3 for vertical flip
    imgRaw = img.copy()
    #if we write draw = false inside findHands() it will remove the marks in hands also need to remove the img
    # Find the hand and its landmarks
    hands, img = detector.findHands(img, flipType=False)  # with draw

    #overlaying the background images
    img = cv2.addWeighted(img, 0, imgBackground, 1, 0.0)

    # Check for the hands
    if hands:
        for hand in hands:
            x,y,w,h = hand['bbox']
            h1,w1, _ = imgBat1.shape
            y1 = y - h1//2
            y1 = np.clip(y1, 20, 415)
            # w1 = 26
            # h1 = 129
            if hand['type'] == "Left":
                img = cvzone.overlayPNG(img, imgBat1, (59, y1))
                if 59 < ballPos[0] <59+w1 and y1<ballPos[1]<y1+h1:
                    # speedx += 2
                    # speedy += 2
                    speedx = -speedx
                    ballPos[0] += 30
                    score[0]+=1

            if hand['type'] == "Right":
                img = cvzone.overlayPNG(img, imgBat2, (1195, y1))
                if 1145 < ballPos[0] <1165 and y1<ballPos[1]<y1+h1:
                    # speedx += 2
                    # speedy += 2
                    speedx = -speedx
                    ballPos[0] -= 30
                    score[1]+=1

    # Game Over
    if ballPos[0]<40 or ballPos[0]>1200:
        gameover = True

    if gameover:
        img = imgGameover
        cv2.putText(img, str(score[1]+score[0]).zfill(2),(585,360),cv2.FONT_HERSHEY_COMPLEX, 2.5, (200,0,20), 5)


    else:
        # Move The ball
        if ballPos[1] >= 500 or ballPos[1] <= 10:
            speedy = -speedy

        ballPos[0] += speedx
        ballPos[1] += speedy

        # Draw the ball
        img = cvzone.overlayPNG(img, imgBall, ballPos)

        cv2.putText(img, str(score[0]),(300,650),cv2.FONT_HERSHEY_COMPLEX, 3, (255,255,255), 5)
        cv2.putText(img, str(score[1]),(900,650),cv2.FONT_HERSHEY_COMPLEX, 3, (255,255,255), 5)

    img[580:700, 20:233] = cv2.resize(imgRaw, (213,120))

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)               #for waiting 1 mili sec
    if key == ord("r"):
        ballPos = [100, 100]
        speedx = 15
        speedy = 15
        gameover = False
        score = [0, 0]
        imgGameover = cv2.imread("Resources/gameOver.png")