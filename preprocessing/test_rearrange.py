# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-29
# description: Data pre-processing script for deepfake dataset.


"""
After running this code, it will generates a json file looks like the below structure for re-arrange data.

{
    "FaceForensics++": {
        "Deepfakes": {
            "video1": {
                "label": "fake",
                "frames": [
                    "/path/to/frames/video1/frame1.png",
                    "/path/to/frames/video1/frame2.png",
                    ...
                ]
            },
            "video2": {
                "label": "fake",
                "frames": [
                    "/path/to/frames/video2/frame1.png",
                    "/path/to/frames/video2/frame2.png",
                    ...
                ]
            },
            ...
        },
        "original_sequences": {
            "youtube": {
                "video1": {
                    "label": "real",
                    "frames": [
                        "/path/to/frames/video1/frame1.png",
                        "/path/to/frames/video1/frame2.png",
                        ...
                    ]
                },
                "video2": {
                    "label": "real",
                    "frames": [
                        "/path/to/frames/video2/frame1.png",
                        "/path/to/frames/video2/frame2.png",
                        ...
                    ]
                },
                ...
            }
        }
    }
}
"""


import os
import glob
import re
import cv2
import json
import yaml
import pandas as pd
from pathlib import Path


def generate_dataset_file(dataset_name, dataset_root_path, output_file_path, compression_level='c23', perturbation = 'end_to_end'):
    """
    Description:
        - Generate a JSON file containing information about the specified datasets' videos and frames.
    Args:
        - dataset: The name of the dataset.
        - dataset_path: The path to the dataset.
        - output_file_path: The path to the output JSON file.
        - compression_level: The compression level of the dataset.
    """

    # Initialize an empty dictionary to store dataset information.
    dataset_dict = {}
    if dataset_name == 'testdata':
        dataset_path = os.path.join(dataset_root_path, dataset_name) # ~/testdata
        dataset_dict[dataset_name] = {f'{dataset_name}_Real': {'test': {}},
                                f'{dataset_name}_Fake': {'test': {}}}
        for index, entry in enumerate(os.scandir(dataset_path)):
            if index < 3000:
                label = f'{dataset_name}_Fake'
                dataset_dict[dataset_name][label]['test'][index] = {'label': label, 'frames': [entry.path]}
            else:
                label = f'{dataset_name}_Real'
                dataset_dict[dataset_name][label]['test'][index] = {'label': label, 'frames': [entry.path]}

    # Convert the dataset dictionary to JSON format and save to file
    output_file_path = os.path.join(output_file_path, dataset_name + '.json')
    with open(output_file_path, 'w') as f:
        json.dump(dataset_dict, f)
    # print the successfully generated dataset dictionary
    print(f"{dataset_name}.json generated successfully.")

if __name__ == '__main__':
    # from config.yaml load parameters
    yaml_path = './preprocessing/config.yaml'
    # open the yaml file
    try:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.parser.ParserError as e:
        print("YAML file parsing error:", e)
    config['rearrange']['dataset_name']['default'] = 'testdata'
    config['rearrange']['dataset_root_path']['default'] = '/'
    dataset_name = config['rearrange']['dataset_name']['default']
    dataset_root_path = config['rearrange']['dataset_root_path']['default']
    output_file_path = config['rearrange']['output_file_path']['default']
    comp = config['rearrange']['comp']['default']
    perturbation = config['rearrange']['perturbation']['default']
    # Call the generate_dataset_file function
    generate_dataset_file(dataset_name, dataset_root_path, output_file_path, comp, perturbation)
