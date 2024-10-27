## ⏳ Quick Start

### 1. Installation
1. (option 1) You can run the following script to configure the necessary environment:

```
conda create -n DeepfakeBench python=3.11
conda activate DeepfakeBench
sh environment.sh
```
2. Download the [shape_predictor_81_face_landmarks.dat](https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.0/shape_predictor_81_face_landmarks.dat) file. Then, copy the downloaded shape_predictor_81_face_landmarks.dat file into the `./preprocessing/dlib_tools folder`.

### 2. Download Data

<a href="#top">[Back to top]</a>

All datasets used in our model can be downloaded here   // TODO

| Types                     | Links| Notes|       
| -------------------------|------- | ----------------------- |
| **Rgb-format Datasets**  | [Password: ogjn](https://pan.baidu.com/s/1NAMUHcZvsIm7l6hMHeEQjQ?pwd=ogjn)| Preprocessed data|       
| **Lmdb-format Datasets** | [Password: g3gj](https://pan.baidu.com/s/1riMCN5iXTJ2g9fQjtlZswg?pwd=g3gj)| LMDB database for each dataset|       
| **Json Configurations**  | [Password: dcwv](https://pan.baidu.com/s/1d7PTV2GK-fpGibcbtnQDqQ?pwd=dcwv)| Data arrangement|       


The provided datasets are:

| Dataset Name                    | Notes                   |
| ------------------------------- | ----------------------- |
| A_data                          | -                       |
| starganv2                       | -                       |
| web,web2                        |  From Website           |
| whichfaceisreal                 | -                       |
| B_data                          |  Only Test Data         |


Upon downloading the datasets, please ensure to store them in the [`./datasets`](./datasets/) folder, arranging them in accordance with the directory structure outlined below:

```
datasets
├── A_data
│   ├── fake
│   │   └── frame
│   |           ├── 0/*.png
│   |           ├── 1/*.png
│   |           ├── ...
│   |           └── xxxx/*.png
│   └── real
│         └── frame
│               ├── 0/*.png
│               ├── 1/*.png
│               ├── ...
│               └── xxxx/*.png
└── web
    ├── fake
    │   └── frame
    |           ├── 0/*.png
    |           ├── 1/*.png
    |           ├── ...
    |           └── xxxx/*.png
    └── real
          └── frame
                ├── 0/*.png
                ├── 1/*.png
                ├── ...
                └── xxxx/*.png

Other datasets are similar to the above structure
```

The downloaded json configurations should be arranged as:
```
preprocessing
└── dataset_json
    ├── A_data.json
    ├── B_data.json
    ├── starganv2.json
    └── web.json
```

### 3. Rearrangement

<a href="#top">[Back to top]</a>

To start preprocessing your dataset, please follow these steps:

1. Open the [`./preprocessing/config.yaml`](./preprocessing/config.yaml) and locate the line `default: DATASET_YOU_SPECIFY`. Replace `DATASET_YOU_SPECIFY` with the name of the dataset you want to preprocess, such as `A_data`.

7. Specify the `dataset_root_path` in the config.yaml file. Search for the line that mentions dataset_root_path. By default, it looks like this: ``dataset_root_path: ./datasets``.
Replace `./datasets` with the actual path to the folder where your dataset is arranged. 

Once you have completed these steps, you can proceed with running the following line to do the preprocessing:
```
cd preprocessing

python rearrange.py
```
After running the above line, you will obtain the JSON files for each dataset in the `./preprocessing/dataset_json` folder. The rearranged structure organizes the data in a hierarchical manner, grouping videos based on their labels and data splits (*i.e.,* train, test, validation). Each video is represented as a dictionary entry containing relevant metadata, including file paths, labels, compression levels (if applicable), *etc*. 


### 5. Training

<a href="#top">[Back to top]</a>

To run the training code, you should first download the pretrained weights for the corresponding **backbones** (These pre-trained weights are from ImageNet). You can download them from [Link](https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.0/pretrained.zip). After downloading, you need to put all the weights files into the folder `./pretrained`.

Then, you should go to the `./training/config/detector/` folder and then Choose our detector [`ylw_detector.py`](training/detectors/ylw_detector.py) to be trained. You can adjust the parameters in [`ylw.yaml`](./training/config/detector/ylw.yaml) to specify the parameters, *e.g.,* training and testing datasets, epoch, frame_num, *etc*.

After setting the parameters, you can run with the following to train the Xception detector:

```
python training/train.py --detector_path ./training/config/detector/ylw.yaml
```

You can also adjust the training and testing datasets using the command line, for example:

```
python training/train.py \
--detector_path ./training/config/detector/ylw.yaml  \
--train_dataset "A_data" "web" \
--test_dataset "B_data"
```

By default, the checkpoints and features will be saved during the training process. If you do not want to save them, run with the following:

```
python training/train.py \
--detector_path ./training/config/detector/ylw.yaml  \
--train_dataset "A_data" "web" \
--test_dataset "B_data" \
--no-save_ckpt \
--no-save_feat
```

### 6. Evaluation
If you only want to evaluate the detectors to produce the results of the cross-dataset evaluation, you can use the the [`test.py`](./training/test.py) code for evaluation. Here is an example:

```
python3 training/test.py \
--detector_path ./training/config/detector/ylw.yaml \
--test_dataset "B_data" \
--weights_path ./logs/training/ylw_99_B_data/test/B_data/ckpt_best.pth
```

## 🏆 Results
