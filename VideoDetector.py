import PIL
import settings
import streamlit as st
import os
import cv2
from pathlib import Path
import shutil
import helper


class VideoDetector:
    def __init__(self, model, accuracy):
        self.model = model
        self.accuracy = accuracy

    def detect(self):
        video_path = None
        source_vid = None
        with st.sidebar:
            source_vid = st.file_uploader("Upload a video", type=["mp4"])
        is_display_tracker, tracker = helper.display_tracker_options()
        try:
            if source_vid is not None:
                video_path = os.path.join("videos", source_vid.name)
                with open(video_path, "wb") as video_file:
                    video_file.write(source_vid.read())
            else:
                video_path = os.path.join("videos", "default.mp4")
            st.video(video_path)
        except Exception as ex:
            st.error(f"Error loading video")
            st.error(ex)

        detected_objects_summary_list = []

        if st.sidebar.button("Detect Objects"):
            vid_cap = cv2.VideoCapture(video_path)
            output_path = "processed_video.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = vid_cap.get(cv2.CAP_PROP_FPS)
            width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            st_frame = st.empty()
            while vid_cap.isOpened():
                success, image = vid_cap.read()
                if success:
                    res, detected_objects = helper.display_frames(
                        self.model,
                        self.accuracy,
                        st_frame,
                        image,
                        is_display_tracker,
                        tracker,
                    )
                    detected_objects_summary_list.extend(detected_objects)
                    out.write(res[0].plot())
                else:
                    vid_cap.release()
                    out.release()
                    if Path(video_path).name != "default.mp4":
                        os.remove(video_path)
                    break

            # Display the processed video within the app
            st.video(output_path)
            
            # Optionally remove the file after display (or keep for user convenience)
            os.remove(output_path)

