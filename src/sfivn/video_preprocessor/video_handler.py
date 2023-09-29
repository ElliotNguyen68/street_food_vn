import os
import math
import shutil

import cv2
from pytube import YouTube
from loguru import logger


def download_video(video_id: str, output_dir: str = "../data/videos")-> str:
    """

    Args:
        video_id (str): _description_
        output_dir (str, optional): _description_. Defaults to "../data/videos".

    Returns:
        str: name of file 
    """
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
        return stream.default_filename 
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
    logger.info('Done open video')

    # Get the frames per second (fps) of the video
    fps = video.get(cv2.CAP_PROP_FPS)

    # Set the desired interval in seconds
    interval = num_sec_per_image

    # Calculate the frame interval based on the fps
    frame_interval = math.ceil(fps * interval)

    # Read and save frames at the specified interval
    frame_count = 0
    logger.info('start extract frames')
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


def framing_video_base_on_video_id(
    id:str,
    frames_output_dir:str,
    num_sec_per_frame:int=1,
    remove_video_after_framings:bool=True
):
    name_video=download_video(
        video_id=id,
        output_dir=frames_output_dir+'_video', 
    )

    split_video_to_images(
        video_path=frames_output_dir+'_video'+'/{}'.format(name_video),
        num_sec_per_image=num_sec_per_frame,
        output_images_directory=frames_output_dir+'_frames'
    )
    
    if remove_video_after_framings:
        shutil.rmtree(frames_output_dir+'_video')
        
    
    
    
    
    
     