import os
import math
import shutil
from typing import Any,Callable


import cv2
from pytube import YouTube
from loguru import logger
import pandas as pd


def download_video(video_id: str, output_dir: str = "../data/videos") -> str:
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
            "{}/{}".format(output_dir, video_id),
        )
        return video_id
    except Exception as e:
        logger.debug(e)
        pass


def split_video_to_images(
    video_path: str,
    output_images_directory: str,
    num_sec_per_image: int = 1,
    separate_by_video_name: bool = True,
) -> int:
    """_summary_

    Args:
        video_path (str): _description_
        output_images_directory (str): _description_
        num_sec_per_image (int, optional): _description_. Defaults to 1.
        separate_by_video_name (bool, optional): _description_. Defaults to True.

    Returns:
        int: number of frames
    """
    video_id = video_path.split("/")[-1]
    if os.path.exists(output_images_directory) == False:
        os.mkdir(output_images_directory)
    if separate_by_video_name:
        try:
            # remove if exist folder for this video
            shutil.rmtree(output_images_directory + "/{}".format(video_id))
        except Exception as e:
            logger.debug(e)
        if os.path.exists(output_images_directory + "/{}".format(video_id)) == False:
            logger.info("mkdir sep")
            os.mkdir(output_images_directory + "/{}".format(video_id))

    # Open the video file
    video = cv2.VideoCapture(video_path)
    logger.info("Done open video")

    # Get the frames per second (fps) of the video
    fps = video.get(cv2.CAP_PROP_FPS)

    # Set the desired interval in seconds
    interval = num_sec_per_image

    # Calculate the frame interval based on the fps
    frame_interval = math.ceil(fps * interval)

    # Read and save frames at the specified interval
    frame_count = 0
    logger.info("start extract frames")
    image_base_path = "{}".format(output_images_directory)
    if separate_by_video_name:
        image_base_path += "/{}".format(video_id)
    no_actual_frames = 0
    while True:
        # Read the next frame
        success, frame = video.read()

        # Check if the frame was read successfully
        if not success:
            break

        # Save the frame as an image
        if frame_count % frame_interval == 0:
            image_path = image_base_path + f"/frame_{no_actual_frames}.jpg"
            cv2.imwrite(image_path, frame)
            no_actual_frames += 1
        frame_count += 1
    logger.info("Num frames in {} video = {}".format(video_path, no_actual_frames))

    # Release the video capture object
    video.release()

    return no_actual_frames


def framing_video_base_on_video_id(
    id: str,
    frames_output_dir: str,
    sec_per_frame: int = 1,
    remove_video_after_framings: bool = True,
)->int:
    name_video = download_video(
        video_id=id,
        output_dir=frames_output_dir + "_video",
    )

    num_frames_extracted = split_video_to_images(
        video_path=frames_output_dir + "_video" + "/{}".format(name_video),
        num_sec_per_image=sec_per_frame,
        output_images_directory=frames_output_dir + "_frames",
    )
 
    if remove_video_after_framings:
        shutil.rmtree(frames_output_dir + "_video")
     
    return num_frames_extracted


def video_extract_base_on_id(
    video_id: str,
    frames_output_dir: str,
    extract_function: Callable,
    sec_per_frames: 1,
    *args
)->pd.DataFrame:
    num_frames_in_video = framing_video_base_on_video_id(
        id=video_id, sec_per_frames=sec_per_frames
    )

    base_dir = frames_output_dir + "_frames"
    list_features=[]
    list_frames=[]
    for frame_no in num_frames_in_video:
        image_path = base_dir + "/frames_{}".format(frame_no)
        
        value = extract_function(image_path)
        list_features.append(value)
        list_frames.append('{}_{}'.format(video_id,frame_no))
        
    return pd.DataFrame(
        {
            'frame_no':list_frames,
            'features':list_features
        }
    )
        
    
        
