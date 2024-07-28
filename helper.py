from ultralytics import YOLO
import time
import streamlit as st
import cv2
from pytube import YouTube
import settings


def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model


def display_tracker_options():
    """
    Displays options for enabling object tracking in the Streamlit app.

    Returns:
        Tuple (bool, str): A tuple containing a boolean flag for displaying the tracker and the selected tracker type.
    """
    display_tracker = st.radio("Display Tracker", ("Yes", "No"))
    is_display_tracker = True if display_tracker == "Yes" else False
    if display_tracker:
        tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
        return is_display_tracker, tracker_type
    return is_display_tracker, None


def display_frames(
    model, acc, st_frame, image, is_display_tracker=None, tracker_type=None
):
    """
    Displays detected objects from a video stream and overlays the count of each detected object.

    Parameters:
        model (YOLO): A YOLO object detection model.
        acc (float): The model's confidence threshold.
        st_frame (streamlit.Streamlit): A Streamlit frame object.
        image (PIL.Image.Image): A frame from a video stream.
        is_display_tracker (bool): Whether or not to display a tracker.
        tracker_type (str): The type of tracker to display.

    Returns:
        tuple: (results, detected_objects_summary_list)
    """
    image = cv2.resize(image, (720, int(720 * (9 / 16))))
    if is_display_tracker:
        res = model.track(image, conf=acc, persist=True, tracker=tracker_type)
    else:
        res = model.predict(image, conf=acc)

    detected_objects_summary_list = res[0].boxes.cls

    # Convert results to plot with counts overlay
    res_plot = res[0].plot()

    # Count detected objects
    detected_objects_count = {}
    for obj in detected_objects_summary_list:
        obj_name = model.names[int(obj)]
        if obj_name in detected_objects_count:
            detected_objects_count[obj_name] += 1
        else:
            detected_objects_count[obj_name] = 1

    # Create a summary string
    summary = ", ".join([f"{name}: {count}" for name, count in detected_objects_count.items()])

    # Draw the summary text on the image
    font_scale = 1.0  # Increase font scale to make text larger
    thickness = 2     # Increase thickness for better visibility
    text_size, _ = cv2.getTextSize(summary, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    text_width = text_size[0]
    text_height = text_size[1]
    image_height, image_width, _ = res_plot.shape
    text_x = (image_width - text_width) // 2
    text_y = text_height + 10  # Position the text near the top

    cv2.putText(res_plot, summary, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

    st_frame.image(
        res_plot,
        caption="Detected Video",
        channels="BGR",
        use_column_width=True,
    )

    return res, detected_objects_summary_list




def sum_detections(detected_objects_summary_list, model):
    """
    Summarizes detected objects from a list and returns the summary with counts.

    Parameters:
        detected_objects_summary_list (list): List of detected object indices.

    Returns:
        str: A string summarizing the detected objects and their counts.
    """
    detected_objects_count = {}
    for obj in detected_objects_summary_list:
        obj_name = model.names[int(obj)]
        if obj_name in detected_objects_count:
            detected_objects_count[obj_name] += 1
        else:
            detected_objects_count[obj_name] = 1

    summary = ", ".join([f"{name}: {count}" for name, count in detected_objects_count.items()])
    return summary
