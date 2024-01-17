import os
import cv2
import random
import matplotlib.pyplot as plt

plt.figure(figsize=(50, 50))

classes_folder_path = r'C:\Users\eTech\Desktop\ML Project\Data_Extracted\Datasets\Peliculas'


# This is Just an Function to SHow random Selected coloured frame from videos with labeling.
all_classes_names = os.listdir(classes_folder_path)
random_range = random.sample(range(len(all_classes_names)), len(all_classes_names))

for counter, random_index in enumerate(random_range, 1):

    selected_class_name = all_classes_names[random_index]

    video_files_names_list = os.listdir(os.path.join(classes_folder_path, selected_class_name))

    selected_video_file_name = random.choice(video_files_names_list)

    video_reader = cv2.VideoCapture(os.path.join(classes_folder_path, selected_class_name, selected_video_file_name))

    video_reader.set(1, 25)

    _, bgr_frame = video_reader.read()

    bgr_frame = cv2.resize(bgr_frame, (224, 224))

    video_reader.release()

    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

    cv2.putText(rgb_frame, selected_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

    plt.subplot(5, 4, counter)
    plt.imshow(rgb_frame)
    plt.axis('off')

plt.show()