"""
Extracts images from (compressed) videos, used for the FaceForensics++ dataset

Usage: see -h or https://github.com/ondyari/FaceForensics

Author: Andreas Roessler
Date: 25.01.2019
"""
import os
from os.path import join
import argparse
import subprocess
import cv2
from tqdm import tqdm


DATASET_PATHS = {
    'original': 'original_sequences',
    'Deepfakes': 'manipulated_sequences/Deepfakes',
    'Face2Face': 'manipulated_sequences/Face2Face',
    'FaceSwap': 'manipulated_sequences/FaceSwap',
    'DeepFakeDetection': 'manipulated_sequences/DeepFakeDetection',
    'FaceShifter': 'manipulated_sequences/FaceShifter',
    'NeuralTextures': 'manipulated_sequences/NeuralTextures'
}
COMPRESSION = ['c0', 'c23', 'c40']


def extract_frames(data_path, output_path, method='cv2', frame_skip=1):
    """Method to extract frames, either with ffmpeg or opencv. FFmpeg won't
    start from 0 so we would have to rename if we want to keep the filenames
    coherent.
    
    Parameters:
        data_path: str, path to the video file
        output_path: str, path to save the extracted frames
        method: str, method to use for extraction ('ffmpeg' or 'cv2')
        frame_skip: int, number of frames to skip between each extraction
    """
    os.makedirs(output_path, exist_ok=True)
    if method == 'ffmpeg':
        # Use ffmpeg to extract frames, with skipping
        subprocess.check_output(
            'ffmpeg -i {} -vf "select=not(mod(n\,{}))" -vsync vfr {}'.format(
                data_path, frame_skip, join(output_path, '%04d.png')),
            shell=True, stderr=subprocess.STDOUT)
    elif method == 'cv2':
        reader = cv2.VideoCapture(data_path)
        frame_num = 0
        saved_frame_num = 0
        while reader.isOpened():
            success, image = reader.read()
            if not success:
                break
            if frame_num % frame_skip == 0:
                cv2.imwrite(join(output_path, '{:04d}.png'.format(saved_frame_num)),
                            image)
                saved_frame_num += 1
            frame_num += 1
        reader.release()
    else:
        raise Exception('Wrong extract frames method: {}'.format(method))


def extract_method_videos(data_path, dataset, compression, frame_skip):
    """Extracts all videos of a specified method and compression in the
    FaceForensics++ file structure"""
    videos_path = join(data_path, DATASET_PATHS[dataset],  'videos')#compression,
    images_path = join(data_path, DATASET_PATHS[dataset],  'images')#compression,
    for video in tqdm(os.listdir(videos_path)):
        image_folder = video.split('.')[0]
        extract_frames(join(videos_path, video),
                       join(images_path, image_folder),
                       frame_skip=frame_skip)
                       


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument('--data_path', type=str)
    p.add_argument('--dataset', '-d', type=str,
                   choices=list(DATASET_PATHS.keys()) + ['all'],
                   default='all')
    p.add_argument('--compression', '-c', type=str, choices=COMPRESSION,
                   default='c23')
    p.add_argument('--frame_skip', type=int, default=40, 
                   help="Number of frames to skip between each extraction.")
    args = p.parse_args()

    if args.dataset == 'all':
        for dataset in DATASET_PATHS.keys():
            args.dataset = dataset
            extract_method_videos(**vars(args))
    else:
        extract_method_videos(**vars(args))
