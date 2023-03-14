import cv2

#Function to create box in face
def faceBox(faceNet, frame):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (227, 227), [104, 117, 123], swapRB = False)
    faceNet.setInput(blob)
    detection = faceNet.forward()
    boxs = [] 
    for i in range(detection.shape[2]):
        confidence = detection[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detection[0,0,i,3]*frameWidth)
            y1 = int(detection[0,0,i,4]*frameHeight)
            x2 = int(detection[0,0,i,5]*frameWidth)
            y2 = int(detection[0,0,i,6]*frameHeight)
            boxs.append([x1,y1,x2,y2])
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0, 255, 0), 1)
    return frame, boxs
    
    
#Face Detect File    
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
#Age Detect File
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
#Gender Detect File
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)


faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Pria', 'Wanita']


video = cv2.VideoCapture(0)
padding = 20
#Looping Capture Video With Camera
while True:
    ret,frame = video.read()
    #Make Camera Mirror
    frameFlip = cv2.flip(frame, 1)
    frame,boxs = faceBox(faceNet, frameFlip)
    
    #Looping Result In faceBox
    for box in boxs:
        face = frame[max(0,box[1]-padding):min(box[3]+padding,frame.shape[0]-1),max(0,box[0]-padding):min(box[2]+padding, frame.shape[1]-1)]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB = False)
        
        genderNet.setInput(blob)
        genderPredict = genderNet.forward()
        gender = genderList[genderPredict[0].argmax()]
        
        ageNet.setInput(blob)
        agePredict = ageNet.forward()
        age = ageList[agePredict[0].argmax()]
        
        label = "{}, {}".format(gender, age)
        cv2.rectangle(frame, (box[0], box[1]-30), (box[2], box[1]), (0, 255, 0), -1)
        cv2.putText(frame, label, (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA, False)
        
    #Showing Camera In Window
    cv2.imshow("Age Gender Detection", frameFlip)
    #close when clicking "esc"
    close = cv2.waitKey(1)
    if close == 27:
        break
video.release()
cv2.destroyAllWindows()