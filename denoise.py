import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import metrics
import bm3d
import csv
from tqdm import tqdm  # 进度条工具


# 配置参数
class Config:
    DATASET_PATH = "/project/datasets/Set14"  # 数据集路径
    OUTPUT_DIR = "/project/week1/results"  # 输出目录
    # 噪声参数
    GAUSSIAN_SIGMA = 25  # 高斯噪声标准差（0-255范围）
    SALT_PEPPER_PROB = 0.05  # 椒盐噪声概率（5%）
    BM3D_SIGMA = 30  # BM3D参数
    PLOT_WIDTH = 15  # 图像显示宽度
    PLOT_HEIGHT = 5  # 图像显示高度
    FONT_SIZE = 12  # 标注文字大小
    SAVE_IMAGES = True  # 保存图像结果
    SAVE_CSV = True  # 保存CSV指标


# 添加噪声函数
# 添加高斯噪声到图像（使用正态分布生成噪声矩阵，clip确保像素值不溢出）
def add_gaussian_noise(image, sigma):  # image: 输入图像(0-255) sigma: 噪声标准差
    image_float = image.astype(np.float32) / 255.0  # 将图像转换为[0,1]范围
    noise = np.random.normal(0, sigma / 255, image_float.shape)  # 生成高斯噪声 (均值为0, 标准差为sigma/255)
    noisy_image_float = image_float + noise  # 添加噪声
    noisy_image_float = np.clip(noisy_image_float, 0, 1)  # 裁剪到[0,1]范围
    return (noisy_image_float * 255).astype(np.uint8)  # 转换回[0,255]范围


# 添加椒盐噪声到图像（各2.5%概率生成黑白噪声点，共5%受影响像素）
def add_salt_pepper_noise(image, prob):  # prob: 噪声点出现的总概率  noisy_image: 加噪后的图像
    noisy_image = image.copy()  # 深拷贝原图像
    mask = np.random.random(image.shape[:2])  # 创建噪声掩码（生成二维随机矩阵）
    noisy_image[mask < prob / 2] = 0  # 添加胡椒噪声 (黑色像素)
    noisy_image[mask > 1 - prob / 2] = 255  # 添加盐噪声 (白色像素)
    return noisy_image


# 图像处理与评估函数
# 对加噪图像进行去噪并计算指标
# 参数 original: 原始图像  noisy: 加噪图像 noise_type: 噪声类型 ('gaussian' 或 'salt_pepper') sigma_psd: BM3D的噪声标准差估计
def denoise_and_evaluate(original, noisy, noise_type, sigma_psd):
    # 计算加噪图像的指标(data_range=255，因为图像是8位)
    psnr_n = metrics.peak_signal_noise_ratio(original, noisy)  # psnr_n: 加噪图像的PSNR
    ssim_n = metrics.structural_similarity(original, noisy, data_range=255)  # ssim_n: 加噪图像的SSIM

    # 使用BM3D进行去噪
    if noise_type == 'gaussian':
        # 对于高斯噪声直接使用BM3D
        denoised = bm3d.bm3d(noisy, sigma_psd=sigma_psd)
    else:
        # 对于椒盐噪声，先应用3x3中值滤波去除脉冲噪声，再用BM3D处理残留噪声
        preprocessed = cv2.medianBlur(noisy, 3)  # 中值滤波预处理
        denoised = bm3d.bm3d(preprocessed, sigma_psd=sigma_psd)

    # 计算去噪后的指标
    denoised = denoised.astype(np.uint8)  # 确保去噪图像的数据类型
    psnr_d = metrics.peak_signal_noise_ratio(original, denoised)  # psnr_d: 去噪图像的PSNR
    ssim_d = metrics.structural_similarity(original, denoised, data_range=255)  # ssim_d: 去噪图像的SSIM

    return denoised, psnr_n, ssim_n, psnr_d, ssim_d


# 可视化并保存结果
# image_name: 图像名称 (用于保存)
def plot_results(original, noisy, denoised, psnr_n, ssim_n, psnr_d, ssim_d, noise_type, image_name):
    plt.figure(figsize=(Config.PLOT_WIDTH, Config.PLOT_HEIGHT))  # 创建画布

    # 原始图像
    plt.subplot(1, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image', fontsize=Config.FONT_SIZE)
    plt.axis('off')

    # 加噪图像
    plt.subplot(1, 3, 2)
    plt.imshow(noisy, cmap='gray')
    plt.title(f'Noisy ({noise_type})\nPSNR: {psnr_n:.2f}dB, SSIM: {ssim_n:.4f}',
              fontsize=Config.FONT_SIZE)
    plt.axis('off')

    # 去噪图像
    plt.subplot(1, 3, 3)
    plt.imshow(denoised, cmap='gray')
    plt.title(f'BM3D Denoised\nPSNR: {psnr_d:.2f}dB, SSIM: {ssim_d:.4f}',
              fontsize=Config.FONT_SIZE)
    plt.axis('off')

    # 调整布局并保存
    plt.tight_layout()

    if Config.SAVE_IMAGES:
        output_path = os.path.join(
            Config.OUTPUT_DIR,
            f"{image_name}_{noise_type}_results.png"
        )
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

    plt.close()


# 主处理流程
def main():
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    results = []  # 初始化结果存储
    # 获取Set14图像列表
    image_files = [f for f in os.listdir(Config.DATASET_PATH)
                   if f.endswith(('.png', '.jpg', '.bmp'))]

    print(f"Processing {len(image_files)} images from Set14")

    # 处理每张图像，用OpenCV以灰度模式读取
    for img_file in tqdm(image_files):  # 使用tqdm显示进度条
        try:
            # 读取图像
            img_path = os.path.join(Config.DATASET_PATH, img_file)
            original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if original is None:
                continue

            image_name = os.path.splitext(img_file)[0]  # 提取不带扩展名的图像名称（用于后续结果保存）

            # 处理高斯噪声
            noisy_g = add_gaussian_noise(original, Config.GAUSSIAN_SIGMA)  # 添加高斯噪声
            denoised_g, psnr_n_g, ssim_n_g, psnr_d_g, ssim_d_g = denoise_and_evaluate(
                original, noisy_g, 'gaussian', Config.BM3D_SIGMA  # 去噪并计算指标
            )
            # 保存结果
            plot_results(original, noisy_g, denoised_g, psnr_n_g, ssim_n_g, psnr_d_g, ssim_d_g,
                         'gaussian', image_name)

            # 处理椒盐噪声
            noisy_sp = add_salt_pepper_noise(original, Config.SALT_PEPPER_PROB)
            denoised_sp, psnr_n_sp, ssim_n_sp, psnr_d_sp, ssim_d_sp = denoise_and_evaluate(
                original, noisy_sp, 'salt_pepper', Config.BM3D_SIGMA
            )
            # 保存结果
            plot_results(original, noisy_sp, denoised_sp, psnr_n_sp, ssim_n_sp, psnr_d_sp, ssim_d_sp,
                         'salt_pepper', image_name)

            # 存储结果，将两种噪声的处理结果分别存入结果列表
            results.append({  # 高斯
                'image': image_name,
                'noise_type': 'gaussian',
                'psnr_noisy': psnr_n_g,
                'ssim_noisy': ssim_n_g,
                'psnr_denoised': psnr_d_g,
                'ssim_denoised': ssim_d_g
            })

            results.append({  # 椒盐
                'image': image_name,
                'noise_type': 'salt_pepper',
                'psnr_noisy': psnr_n_sp,
                'ssim_noisy': ssim_n_sp,
                'psnr_denoised': psnr_d_sp,
                'ssim_denoised': ssim_d_sp
            })

        # 异常处理（捕获单个图像处理中的异常，避免中断整个批处理）
        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")

    # 保存CSV结果
    if Config.SAVE_CSV and results:
        csv_path = os.path.join(Config.OUTPUT_DIR, "denoising_results.csv")  # 创建CSV文件路径
        with open(csv_path, 'w', newline='') as csvfile:  # 写入表头（自动获取第一条结果的键）
            fieldnames = results[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow(result)
        print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()
