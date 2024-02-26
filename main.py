import cv2
import numpy as np
import matplotlib.pyplot as plt


def frame_diff(video):

    ret, prev_frame = video.read()
    out_target = cv2.VideoWriter('output/fd_target.mp4', cv2.VideoWriter_fourcc(
        *'mp4v'), 30, video.read()[1].shape[:2][::-1])
    while True:
        ret, current_frame = video.read()
        if not ret:
            break  # 视频结束

        # 计算帧间差
        frame_diff = cv2.absdiff(prev_frame, current_frame)

        # 应用阈值
        threshold = 30
        _, thresholded_frame = cv2.threshold(
            frame_diff, threshold, 255, cv2.THRESH_BINARY)

        # 显示帧间差结果
        cv2.imshow('src', current_frame)
        cv2.imshow('Frame Difference', thresholded_frame)
        out_target.write(thresholded_frame)
        prev_frame = current_frame

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


def get_ave_var(video):
    sum1 = np.zeros(video.read()[1].shape)
    sum2 = np.zeros(video.read()[1].shape)
    # 采样12帧初始化均值和方差
    n = 12
    for i in range(n):
        ret, frame = video.read()

        if ret == False:
            break
        sum1 += frame
    ave = sum1 / n
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for i in range(n):
        ret, frame = video.read()

        if ret == False:
            break
        sum2 += (frame - ave)**2
    var = sum2 / n
    return ave, var


def singleGaussian(video, ave, var):
    # ave = video.read()[1].astype(np.float32)
    # var = 20*20*np.ones(video.read()[1].shape)
    learning_rate = 0.2
    lambda_ = 3
    background = video.read()[1].astype(np.float32)

    out_target = cv2.VideoWriter('output/sg_target.mp4', cv2.VideoWriter_fourcc(
        *'mp4v'), 30, video.read()[1].shape[:2][::-1])
    out_background = cv2.VideoWriter('output/sg_background.mp4', cv2.VideoWriter_fourcc(
        *'mp4v'), 30, video.read()[1].shape[:2][::-1])

    while True:

        ret, frame = video.read()
        if ret == False:
            break

        bg_points = np.where(
            (np.abs(frame.astype(np.float32) - ave) < lambda_ * np.sqrt(var)).sum(axis=2) == 3)

        ave[bg_points] = (1-learning_rate)*ave[bg_points] + \
            learning_rate*frame[bg_points]
        var[bg_points] = (1-learning_rate)*var[bg_points] + learning_rate * \
            (frame[bg_points] - ave[bg_points])**2

        background[bg_points] = (1-learning_rate)*background[bg_points] + \
            learning_rate*frame[bg_points]
        # 原图减去背景

        obj = cv2.absdiff(frame, background.astype(np.uint8))
        # 二值化
        _, obj = cv2.threshold(obj, 30, 255, cv2.THRESH_BINARY)
        # 形态学处理
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

        obj = cv2.morphologyEx(obj, cv2.MORPH_OPEN, kernel)
        obj = cv2.morphologyEx(obj, cv2.MORPH_CLOSE, kernel)
        # 显示
        out_target.write(obj.astype(np.uint8))
        out_background.write(background.astype(np.uint8))
        cv2.imshow('obj', obj)
        cv2.imshow('background', background.astype(np.uint8))
        cv2.imshow('src', frame)
        if cv2.waitKey(1) == 27 or cv2.waitKey(1) == ord('q'):
            break

    out_target.release()
    out_background.release()
    cv2.destroyAllWindows()


def GMM(video):
    # 创建背景建模器
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    out_target = cv2.VideoWriter('output/gmm_target.mp4', cv2.VideoWriter_fourcc(
        *'mp4v'), 30, video.read()[1].shape[:2][::-1])
    out_background = cv2.VideoWriter('output/gmm_background.mp4', cv2.VideoWriter_fourcc(
        *'mp4v'), 30, video.read()[1].shape[:2][::-1])
    while True:
        ret, frame = video.read()
        if not ret:
            break

        # 应用背景建模器
        foreground_mask = bg_subtractor.apply(frame)

        # 获取背景
        background = bg_subtractor.getBackgroundImage()
        # foreground_mask = cv2.absdiff(frame, background)
        # # 二值化
        # _, foreground_mask = cv2.threshold(
        #     foreground_mask, 30, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        foreground_mask = cv2.morphologyEx(
            foreground_mask, cv2.MORPH_OPEN, kernel)
        foreground_mask = cv2.morphologyEx(
            foreground_mask, cv2.MORPH_CLOSE, kernel)
        # 显示原始帧、前景和背景
        cv2.imshow('Original Frame', frame)
        cv2.imshow('Foreground', foreground_mask)
        cv2.imshow('Background', background)
        out_target.write(cv2.cvtColor(foreground_mask, cv2.COLOR_GRAY2BGR))
        out_background.write(background)
        if cv2.waitKey(30) & 0xFF == 27 or cv2.waitKey(30) == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    video = cv2.VideoCapture('1.avi')

    # 帧间差法
    # frame_diff(video)

    # 单高斯模型
    # ave, var = get_ave_var(video)
    # video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    # singleGaussian(video, ave, var)

    # 高斯混合模型
    GMM(video)
