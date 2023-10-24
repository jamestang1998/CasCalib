import cv2
import os
import tqdm

video_path = 'CalibSingleFromP2D/videos'
frames_path = 'CalibSingleFromP2D/Frames'
frames_enum = ['campus4-c0_avi','campus4-c1_avi','campus4-c2_avi']
video_enum = ['4p-c0.avi','4p-c1.avi','4p-c2.avi']

for i in range(3):
    print(video_path + '/' + video_enum[i])
    print(frames_path + '/' + frames_enum[i])
    cap = cv2.VideoCapture(video_path + '/' + video_enum[i])
    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()
    output_directory = frames_path + '/' + frames_enum[i]
    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Define the filename for the JPEG image
        frame_filename = os.path.join(output_directory, f'{frame_number:08d}.jpg')
        print(frame_filename)
        # Save the frame as a JPEG image
        if not (cv2.imwrite(frame_filename, frame)):
            print("error")
        frame_number += 1
    cap.release()
cv2.destroyAllWindows()


