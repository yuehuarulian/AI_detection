import subprocess
import os
import zipfile
import requests

def download_pretrained_weights(url, dest_folder, zip_name='pretrained.zip'):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # 下载文件
    zip_path = os.path.join(dest_folder, zip_name)
    print(f"Downloading pretrained weights from {url}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(zip_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded to {zip_path}")
    else:
        print(f"Failed to download file: {response.status_code}")
        return

    # 解压文件
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_folder)
    print(f"Extracted files to {dest_folder}")

    os.remove(zip_path)
    print(f"Removed zip file {zip_path}")



def main():
    # 获取当前 Python 文件所在的路径
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    print('main.py所在位置:',current_file_dir)
    os.chdir(current_file_dir)
    
    # pretrained
    url = "https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.0/pretrained.zip"
    dest_folder =  "./"
    download_pretrained_weights(url, dest_folder)

    # test_rearrange
    command = [
        'python', os.path.join('preprocessing/test_rearrange.py')
    ]

    try:
        result = subprocess.run(command, check=True,capture_output=True,text=True)
        print("Command Output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Command failed with error:", e.stderr)
        print("预处理失败")
    
    # test
    command = [
        "python", "training/ylw_test.py",
        "--detector_path", "./training/config/detector/ylw.yaml",
        "--test_dataset", "testdata",
        "--weights_path", "./ylw_model/ckpt_best.pth",
        '--csv_path', os.path.join(current_file_dir,'../')
    ]
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("Command Output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Command failed with error:", e.stderr)
        print("测试失败")

if __name__ == "__main__":
    main()
