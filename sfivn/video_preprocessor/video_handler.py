import os
import math

import cv2
from pytube import YouTube
from loguru import logger


def download_video(video_id: str, output_dir: str = "../data/videos"):
    try:
        yt = YouTube("http://youtube.com/watch?v={}".format(video_id))
        stream = (
            yt.streams.filter(progressive=True, file_extension="mp4")
            .order_by("resolution")
            .desc()
            .first()
        )

        stream.download(output_path=output_dir)

        os.rename(
            "{}/{}".format(output_dir, stream.default_filename),
            "{}/{}.mp4".format(output_dir, video_id),
        )
    except Exception as e:
        logger.debug(e)
        pass


def split_video_to_images(
    video_path: str, output_images_directory: str, num_sec_per_image: int = 1
):
    if os.path.exists(output_images_directory) == False:
        os.mkdir(output_images_directory)

    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Get the frames per second (fps) of the video
    fps = video.get(cv2.CAP_PROP_FPS)

    # Set the desired interval in seconds
    interval = num_sec_per_image

    # Calculate the frame interval based on the fps
    frame_interval = math.ceil(fps * interval)

    # Read and save frames at the specified interval
    frame_count = 0
    while True:
        # Read the next frame
        success, frame = video.read()

        # Check if the frame was read successfully
        if not success:
            break

        # Save the frame as an image
        if frame_count % frame_interval == 0:
            image_path = f"{output_images_directory}/frame_{frame_count}.jpg"
            cv2.imwrite(image_path, frame)

        frame_count += 1

    # Release the video capture object
    video.release()
