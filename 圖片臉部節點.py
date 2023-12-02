import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

# Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def plot_face_blendshapes_bar_graph(face_blendshapes):
  # Extract the face blendshapes category names and scores.
  face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
  face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
  # The blendshapes are ordered in decreasing score value.
  face_blendshapes_ranks = range(len(face_blendshapes_names))

  fig, ax = plt.subplots(figsize=(12, 12))
  bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
  ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
  ax.invert_yaxis()

  # Label each bar with values
  for score, patch in zip(face_blendshapes_scores, bar.patches):
    plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

  ax.set_xlabel('Score')
  ax.set_title("Face Blendshapes")
  plt.tight_layout()
  plt.show()

# Image
image = cv2.imread("QIUZE3969.JPG")
height, width, _ = image.shape
height //= 2
width //= 2
image = cv2.resize(image, (width, height))
# cv2讀進來的圖預設是BGR，要轉成RGB使用
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

畫虹膜 = False
畫臉部輪廓 = False
畫網格 = False

result = mp_face_mesh.FaceMesh(refine_landmarks=True).process(rgb_image)
if result.multi_face_landmarks:
    for facial_landmarks in result.multi_face_landmarks:
        if 畫網格:
            mp_drawing.draw_landmarks(
                image = image,
                landmark_list = facial_landmarks,
                connections = mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec = None, #可改mp_drawing.DrawingSpec(color = (0, 0, 255))
                connection_drawing_spec = mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )

        if 畫虹膜:
            mp_drawing.draw_landmarks(
                image = image,
                landmark_list = facial_landmarks,
                connections = mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec = None, #可改mp_drawing.DrawingSpec(color = (0, 0, 255))
                connection_drawing_spec = mp_drawing_styles.get_default_face_mesh_iris_connections_style()
            )

        if 畫臉部輪廓:
            mp_drawing.draw_landmarks(
                image = image,
                landmark_list = facial_landmarks,
                connections = mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec = None, #可改mp_drawing.DrawingSpec(color = (0, 0, 255))
                connection_drawing_spec = mp_drawing_styles.get_default_face_mesh_contours_style()
            )



        #用for迴圈一個一個點畫出來
        # for i in range(0, 468):
        #     point1 = facial_landmarks.landmark[i]  # 第一點座標
        #     x = int(point1.x * width)  # 座標不能是浮點數，所以轉成int
        #     y = int(point1.y * height)
        #     cv2.circle(image, (x, y), 2, (255, 0, 0), -1)

        left_eye_point = facial_landmarks.landmark[473]
        right_eye_point = facial_landmarks.landmark[468]
        print("left_eye_point:", left_eye_point, "right_eye_point:", right_eye_point)

plot_face_blendshapes_bar_graph(result.multi_face_blendshapes[0])

cv2.imshow("圖", image)
if cv2.waitKey(0) and 0xFF == ord("q"):
    image.release()
    cv2.destroyAllWindows()


