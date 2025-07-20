# 图片下载脚本使用指南

## 快速开始

1. 双击运行 `run_download.ps1` (PowerShell脚本) 或 `run_download.bat` (批处理文件)
2. 程序将安装必要的依赖并开始下载图片

## 命令行参数

下载脚本支持以下命令行参数:

```
--tsv FILE      指定TSV文件路径 (默认: open-images-dataset-train0.tsv)
--output DIR    指定输出目录 (默认: 当前目录)
--batch-size N  每个文件夹的图片数量 (默认: 1000)
--workers N     并行下载的线程数 (默认: 10)
```

## 示例用法

使用自定义参数:

```powershell
# PowerShell
python download_images.py --tsv "my_data.tsv" --output "D:\downloaded_images" --batch-size 500 --workers 20
```

```batch
# 命令提示符
python download_images.py --tsv "my_data.tsv" --output "D:\downloaded_images" --batch-size 500 --workers 20
```

## 文件夹结构

图片将按照以下结构保存:

```
输出目录/
  1/             # 第1批1000张图片
    0000_image1.jpg
    0001_image2.jpg
    ...
  2/             # 第2批1000张图片
    0000_image1001.jpg
    0001_image1002.jpg
    ...
  download_progress.json  # 下载进度记录文件
```

## 进度保存

程序会自动保存下载进度。如果下载过程中断，再次运行程序将从上次中断的位置继续下载。
