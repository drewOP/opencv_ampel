import cv2
from matplotlib import pyplot as plt

frozen_model = "frozen_inference_graph.pb"
config_file = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
model = cv2.dnn_DetectionModel(frozen_model, config_file)
classLables = []
file_name = 'coco.names'
with open(file_name, 'rt') as fpt:
    classLables = fpt.read().rstrip('\n').split('\n')

model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

# read image
# img = cv2.imread('cat.png')


# classIndex, confidence, bbox = model.detect(img, confThreshold=0.6)
# print(classIndex)

# font_scale = 3
# font = cv2.FONT_HERSHEY_PLAIN
# for classInd, conf, boxes in zip(classIndex.flatten(), confidence.flatten(), bbox):
#    cv2.rectangle(img, boxes, (255, 0, 0), 2)
#    cv2.putText(img, classLables[classInd-1], (boxes[0]+10, boxes[1]+40), font, fontScale=font_scale, color=(0, 255, 0), thickness=3)

# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.show()


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open Video")

font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN

while True:
    hsv, frame = cap.read()

    classIndex, confidence, bbox = model.detect(frame, confThreshold=0.55)

    print(classIndex)
    if (len(classIndex) != 0):
        for classInd, conf, boxes in zip(classIndex.flatten(), confidence.flatten(), bbox):
            if classInd == 10:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                cv2.rectangle(frame, boxes, (255, 0, 255), 2)
                cv2.putText(frame, classLables[classInd - 1], (boxes[0] + 10, boxes[1] + 40), font,
                            fontScale=font_scale, color=(0, 255, 0), thickness=2)
    cv2.imshow('Title', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
