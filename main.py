import pyttsx3
import cv2
import numpy as np

# TTS
engine = pyttsx3.init()

# TTS Functionality
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Load your model here, u can use yolov3 v9 v8 v7 doesn't matter, in my case, I used v3!
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()  # Get the layer names
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]  # output layers

# Classnames for our model ( in this case YOLOv3 ofc )
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize the webcam, in my opinion my webcam device is 1
# you can check it out in device manager to see how many webcams you've got.
# example : VideoCapture(0) - VideoCapture(3) - etc...
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    height, width, channels = frame.shape

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Confidence threshold (You can increase it if you want), by confidence you can tell how much u sure about ur object detection guess, something like that...
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))

            # Drawing frames around detected objects
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label + " " + confidence, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


            # Calling tts to tell me what Computer watching (like before, u can delete this if u want)
            speak(f"You are showing me {label}")

    # Window name
    cv2.imshow("Object Detection by AlirezaPlus", frame)

    # Break on 'q' key ( to quit the instance just press Q (WAITKEY 1 stands for 1 ms thats mean u have to hold ur Q for 1ms which is instant is this case))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
