import cv2  # Computer vision library
import numpy as np  # Scientific computing library
import object_detection  # Custom object detection program
from tensorflow import keras  # Library for neural networks
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from datetime import datetime

# Make sure the video file is in the same directory as your code
filename = 'las_vegas_Trim.mp4'
file_size = (1920, 1080)  # Assumes 1920x1080 mp4
scale_ratio = 1  # Option to scale to fraction of original size.

# We want to save the output to a video file
output_filename = 'las_vegas_annotated.mp4'
output_frames_per_second = 30.0

# Load the SSD neural network that is trained on the COCO data set
model_ssd = object_detection.load_ssd_coco()

# Load the trained neural network
model_traffic_lights_nn = keras.models.load_model("traffic.h5")


def main():
    # Load a video
    cap = cv2.VideoCapture(filename)

    # Create a VideoWriter object so we can save the video output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    result = cv2.VideoWriter(output_filename,
                             fourcc,
                             output_frames_per_second,
                             file_size)

    frame_number = 0
    # Process the video
    while cap.isOpened():

        # Capture one frame at a time
        success, frame = cap.read()

        # Do we have a video frame? If true, proceed.
        if success:

            # Resize the frame
            width = int(frame.shape[1] * scale_ratio)
            height = int(frame.shape[0] * scale_ratio)
            frame = cv2.resize(frame, (width, height))

            # Store the original frame
            original_frame = frame.copy()

            output_frame = object_detection.perform_object_detection_video(
                model_ssd, frame, model_traffic_lights=model_traffic_lights_nn)

            # Write the frame to the output video file
            result.write(output_frame)

            frame_number += 1
            print('{0}: Frame number: {1}'.format(datetime.now(), frame_number))

        # No more video frames left
        else:
            print('No frames left')
            break

    # Stop when the video is finished
    cap.release()

    # Release the video recording
    result.release()

    # Close all windows
    cv2.destroyAllWindows()


main()