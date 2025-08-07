import torch
import numpy as np
import cv2
from PIL import Image, ImageTk
from cmwrn_config import Config
from cmwrn_model import WeatherResNet
from sky_model import ResNetSkyClassifier
from cmwrn_dataset import transform_image  # 假设你有一个函数来预处理图像
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import time

def predict_single_image(image_path, label_sunny, label_rain, progress_bar, label_time):
    progress_bar['value'] = 0
    root.update_idletasks()

    # ==================== 1. 加载模型 ====================
    progress_bar['value'] = 20
    root.update_idletasks()

    # 加载天空检测模型
    sky_model = ResNetSkyClassifier().to(Config.DEVICE)
    sky_model.load_state_dict(torch.load('sky_best_model.pth'))
    sky_model.eval()

    # 加载天气模型（修改后的版本）
    weather_model = WeatherResNet(pretrained=False).to(Config.DEVICE)
    checkpoint = torch.load(Config.MODEL_SAVE_PATH)
    weather_model.load_state_dict(checkpoint['model_state_dict'])
    weather_model.eval()

    # ==================== 2. 预处理图像 ====================
    progress_bar['value'] = 40
    start_time = time.time()
    root.update_idletasks()
    image = cv2.imread(image_path)
    if image is None:
        messagebox.showerror("错误", "无法读取图片")
        progress_bar['value'] = 0
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image)

    # 统一预处理
    image_tensor = transform_image(image_pil)
    image_tensor = image_tensor.unsqueeze(0).to(Config.DEVICE)

    # ==================== 3. 天空检测 ====================
    progress_bar['value'] = 60
    root.update_idletasks()

    with torch.no_grad():
        sky_output = sky_model(image_tensor)
        sky_pred = torch.argmax(sky_output, dim=1).cpu().numpy()[0]

    # ==================== 4. 条件任务执行 ====================
    progress_bar['value'] = 80
    root.update_idletasks()

    # 根据天空检测结果执行不同任务
    if sky_pred == 0:  # 无天空
        # 仅执行降雨分类
        with torch.no_grad():
            _, rain_output = weather_model(image_tensor, run_sunny=False)
        sunny_pred = None
    else:  # 有天空
        # 执行全部任务
        with torch.no_grad():
            sunny_output, rain_output = weather_model(image_tensor)
        sunny_pred = sunny_output.cpu().numpy().flatten()[0]

    # ==================== 5. 结果解析 ====================
    rain_pred = torch.argmax(rain_output, dim=1).cpu().numpy()[0]
    classes = ['no-rain condition', 'light-moderate rain', 'heavy rain']

    # 更新GUI显示
    if sky_pred == 0:
        label_sunny.config(text='sky state bias index: No sky, no recognition')
        label_rain.config(text=f'rainfall intensity: {classes[rain_pred]}')
    else:
        label_sunny.config(text=f'sky state bias index: {sunny_pred:.4f}')
        label_rain.config(text=f'rainfall intensity: {classes[rain_pred]}')

    progress_bar['value'] = 100
    end_time = time.time()
    user_time = end_time - start_time
    label_time.config(text=f'time: {user_time:.4f} s')  # 更新用时显示
    print(f'用时{user_time} s')
    root.update_idletasks()

def select_image(image_label, label_sunny, label_rain, label_time, image_path_var):
    file_path = filedialog.askopenfilename()
    if file_path:
        # 显示图片
        image = cv2.imread(file_path)
        if image is None:
            messagebox.showerror("错误", "无法读取图片，请检查路径是否正确")
            return
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image)
        image_pil = image_pil.resize((250, 250), Image.LANCZOS)  # 固定图片显示尺寸
        image_tk = ImageTk.PhotoImage(image_pil)
        image_label.config(image=image_tk)
        image_label.image = image_tk  # 保持对 PhotoImage 的引用

        # 清空预测结果
        label_sunny.config(text='sky state bias index: None')
        label_rain.config(text='rainfall intensity: None')
        label_time.config(text='time: None')

        # 保存图片路径
        image_path_var.set(file_path)

def main():
    global root
    root = tk.Tk()
    root.title("天气识别")
    root.geometry("700x400")  # 设置窗口尺寸

    # 左侧图片显示区域
    left_frame = tk.Frame(root)
    left_frame.pack(side=tk.LEFT, padx=20, pady=20, fill=tk.BOTH, expand=True)

    image_label = tk.Label(left_frame)
    image_label.pack(side=tk.TOP, padx=10, pady=10)

    # 右侧控制区域
    right_frame = tk.Frame(root)
    right_frame.pack(side=tk.RIGHT, padx=20, pady=20, fill=tk.BOTH, expand=True)

    # 用于存储图片路径的变量
    image_path_var = tk.StringVar()

    # 按钮区域
    frame_buttons = tk.Frame(right_frame)
    frame_buttons.pack(side=tk.TOP, pady=20)

    btn_select_image = tk.Button(frame_buttons, text="Select Picture", command=lambda: select_image(image_label, label_sunny, label_rain,label_time, image_path_var))
    btn_select_image.pack(side=tk.LEFT, padx=10)

    btn_predict = tk.Button(frame_buttons, text="Recognize", command=lambda: predict_single_image(image_path_var.get(), label_sunny, label_rain, progress_bar, label_time))
    btn_predict.pack(side=tk.LEFT, padx=10)

    # 进度条
    progress_bar = ttk.Progressbar(right_frame, orient="horizontal", length=200, mode="determinate")
    progress_bar.pack(side=tk.TOP, pady=20)

    # 预测结果显示区域
    frame_results = tk.Frame(right_frame)
    frame_results.pack(side=tk.TOP, pady=20)

    label_sunny = tk.Label(frame_results, text="sky state bias index: None", font=("Helvetica", 12))
    label_sunny.pack(side=tk.TOP)

    label_rain = tk.Label(frame_results, text="rainfall intensity: None", font=("Helvetica", 12))
    label_rain.pack(side=tk.TOP)

    # 用时显示区域
    label_time = tk.Label(frame_results, text="time: None", font=("Helvetica", 12))
    label_time.pack(side=tk.TOP)

    root.mainloop()

if __name__ == '__main__':
    main()