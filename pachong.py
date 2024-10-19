import os
import requests
from bs4 import BeautifulSoup
import urllib.parse
import base64
import time

# 设置目标网页
url = "https://www.bing.com/images/search?q=%E6%98%8E%E6%98%9F%E5%90%8D%E4%BA%BA%E5%B8%A6%E8%83%8C%E6%99%AF%E7%85%A7%E7%89%87&qs=n&form=QBIR&sp=-1&lq=0&pq=%E6%98%8E%E6%98%9F%E5%90%8D%E4%BA%BA%E5%B8%A6%E8%83%8C%E6%99%AFzhao%27p&sc=0-13&cvid=86F566E541574C2F876119A544D05F9F&ghsh=0&ghacc=0&first=1&cw=1827&ch=1050"
headers = {"User-Agent": "Mozilla/5.0"}

# 创建存储图片的文件夹
output_dir = "datasets/downloaded_real"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 记录已下载图片
log_file = "downloaded_log.txt"
downloaded_images = set()

# 读取已经下载的图片日志
if os.path.exists(log_file):
    with open(log_file, "r") as f:
        downloaded_images = set(f.read().splitlines())

# 发送请求并解析页面
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, "html.parser")

# 找到所有图片标签并下载图片
img_tags = soup.find_all("img")
for img_tag in img_tags:
    img_url = img_tag.get("src")

    if img_url and img_url.startswith("http"):  # 确保是有效的 URL
        # 解析URL，移除不合法字符
        parsed_url = urllib.parse.urlparse(img_url)
        img_name = os.path.basename(parsed_url.path)

        if img_name in downloaded_images:
            print(f"Skipping {img_name}, already downloaded.")
            continue

        # 检查文件名是否合法且非空
        if img_name:
            img_name = img_name.split("?")[0]  # 移除?之后的部分
            img_path = os.path.join(output_dir, img_name)

            # 下载并保存图片，增加重试机制
            retries = 3
            for attempt in range(retries):
                try:
                    img_data = requests.get(img_url, timeout=10).content
                    with open(img_path, "wb") as img_file:
                        img_file.write(img_data)
                    print(f"Downloaded {img_path}")

                    # 记录已下载的图片
                    with open(log_file, "a") as f:
                        f.write(img_name + "\n")

                    break  # 下载成功后跳出重试循环
                except Exception as e:
                    print(f"Error downloading {img_url}: {e}")
                    if attempt < retries - 1:
                        print(f"Retrying ({attempt + 1}/{retries})...")
                        time.sleep(2)  # 等待2秒再重试
                    else:
                        print(f"Failed to download {img_url} after {retries} attempts.")

    elif img_url and img_url.startswith("data:image"):  # 处理 base64 编码的图像
        try:
            # 提取图像类型和base64编码数据
            header, encoded = img_url.split(",", 1)
            file_ext = header.split(";")[0].split("/")[1]  # 提取文件扩展名

            # 生成文件名
            img_name = f"embedded_image_{len(os.listdir(output_dir))}.{file_ext}"
            img_path = os.path.join(output_dir, img_name)

            # 解码 base64 并保存
            img_data = base64.b64decode(encoded)
            with open(img_path, "wb") as img_file:
                img_file.write(img_data)
            print(f"Downloaded embedded image {img_path}")

            # 记录已下载的图片
            with open(log_file, "a") as f:
                f.write(img_name + "\n")

        except Exception as e:
            print(f"Error decoding base64 image: {e}")

    else:
        print(f"Skipping invalid image URL: {img_url}")
