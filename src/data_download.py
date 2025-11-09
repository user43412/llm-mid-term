# simple_download.py
import os
import urllib.request
import zipfile


def simple_download():
    """简单版本的数据下载"""

    # 创建目录
    os.makedirs("../data", exist_ok=True)

    # 下载URL
    url = "https://huggingface.co/datasets/iwslt2017/resolve/main/data/2017-01-trnted/texts/en/de/en-de.zip"
    zip_path = "data/en-de.zip"

    print("正在下载 IWSLT2017 en-de 数据集...")

    # 下载
    urllib.request.urlretrieve(url, zip_path)
    print("下载完成!")

    # 解压
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("data")
    print("解压完成!")

    # 删除zip文件
    os.remove(zip_path)
    print("清理完成!")

    print("数据集已保存到 data/en-de/ 目录")


if __name__ == "__main__":
    simple_download()