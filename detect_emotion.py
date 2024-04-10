import cv2, cv2.data
from deepface import DeepFace

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
video_capture = cv2.VideoCapture(0)


def detect_bounding_box(vid):
    gray_img = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_img, 1.1, 5, minSize=(40, 40))
    for x, y, w, h in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return faces


while True:
    result, video_frame = video_capture.read()
    if result is False:
        break
    faces = detect_bounding_box(video_frame)
    cv2.imshow("Video", video_frame)
    cv2.imwrite("face.jpg", video_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()

objs = DeepFace.analyze(
    img_path="face.jpg", actions=["emotion", "gender"], enforce_detection=False
)
print(objs)
