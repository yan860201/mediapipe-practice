import cv2
import mediapipe as mp

# Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    # Image
    ret, image = cap.read()

    height, width, _ = image.shape
    # cv2讀進來的圖預設是BGR，要轉成RGB使用
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    result = mp_face_mesh.FaceMesh(refine_landmarks=True).process(rgb_image)
    if result.multi_face_landmarks:
        for facial_landmarks in result.multi_face_landmarks:
            #畫虹膜
            mp_drawing.draw_landmarks(
                image = image,
                landmark_list = facial_landmarks,
                connections = mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec = None, #可改mp_drawing.DrawingSpec(color = (0, 0, 255))
                connection_drawing_spec = mp_drawing_styles.get_default_face_mesh_iris_connections_style()
            )

            #畫臉部輪廓
            mp_drawing.draw_landmarks(
                image = image,
                landmark_list = facial_landmarks,
                connections = mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec = None, #可改mp_drawing.DrawingSpec(color = (0, 0, 255))
                connection_drawing_spec = mp_drawing_styles.get_default_face_mesh_contours_style()
            )

            #畫網格
            mp_drawing.draw_landmarks(
                image = image,
                landmark_list = facial_landmarks,
                connections = mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec = None, #可改mp_drawing.DrawingSpec(color = (0, 0, 255))
                connection_drawing_spec = mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )

            #用for迴圈一個一個點畫出來
            # for i in range(0, 468):
            #     point1 = facial_landmarks.landmark[i]  # 第一點座標
            #     x = int(point1.x * width)  # 座標不能是浮點數，所以轉成int
            #     y = int(point1.y * height)
            #     cv2.circle(image, (x, y), 2, (255, 0, 0), -1)

    cv2.imshow("圖", image)
    if cv2.waitKey(20) and 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
