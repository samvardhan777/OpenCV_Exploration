import cv2
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import os
from torch.nn.functional import cosine_similarity
import faiss

mtcnn = MTCNN()
dimension = 512 
index = faiss.IndexFlatL2(dimension)
model = InceptionResnetV1(pretrained='vggface2').eval()
image_folder = 'test_sample'
image_files = os.listdir(image_folder)

image_embeddings = {}

for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    image = cv2.imread(image_path)
    boxes, _ = mtcnn.detect(image)
    if boxes is not None:
        for box in boxes:
            x, y, w, h = [int(i) for i in box]
            face = image[y:y+h, x:x+w]
            face = cv2.resize(face, (160, 160))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = np.transpose(face, (2, 0, 1))
            face = (face - 127.5) / 128.0 
            face_tensor = torch.tensor(face, dtype=torch.float32).unsqueeze(0)
            embedding = model(face_tensor)
            image_embeddings[image_file] = embedding
            index.add(embedding.detach().numpy())
            
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Detect faces in the frame
    boxes, _ = mtcnn.detect(frame)

    if boxes is not None:
        for box in boxes:
            x, y, w, h = [int(i) for i in box]
            face = frame[y:y+h, x:x+w]
            cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2) 
            if face.size == 0:  # Check if face is empty
                continue
            face = cv2.resize(face, (160, 160))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = np.transpose(face, (2, 0, 1))
            face = (face - 127.5) / 128.0 
            face_tensor = torch.tensor(face, dtype=torch.float32).unsqueeze(0)
            embedding = model(face_tensor)
    
    max_similarity = -1
    max_image_file = None
    similarity_threshold = 0.3  # Increase the similarity threshold to optimize the model

    for image_file, image_embedding in image_embeddings.items():
        similarity = cosine_similarity(embedding, image_embedding)
        if similarity.item() > similarity_threshold and similarity.item() > max_similarity:
            max_similarity = similarity.item()
            max_image_file = image_file
            print(max_image_file)

    if max_image_file is not None:
        max_image_path = os.path.join(image_folder, max_image_file)
        max_image = cv2.imread(max_image_path)
        max_image = cv2.resize(max_image, (100, 100))
    else:
        max_image = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.putText(max_image, 'No Match Found', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    frame[0:100, frame.shape[1]-100:frame.shape[1]] = max_image


    cv2.imshow('Real-Time Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()


