# coding=utf-8

import os
import cv2
import argparse
from multiprocessing.pool import ThreadPool
from itertools import repeat
from pathlib import Path
import time


def filter_format(x, all_formats):
    if x[-4:] in all_formats:
        return True
    else:
        return False


def video2frame(video_path, frame_save_path, interval):
    """
    将视频按固定间隔读取写入图片
    :param video_path: 视频存放路径
    :param frame_save_path:　保存路径
    :param interval:　保存帧间隔
    :return:　Nothing but saved images directly
    """

    video_name = video_path.split('/')[-1].split('.')[0]
    image_saved_path = '/'.join(video_path.split('/')[1:-3] + ['images'] + [video_path.split('/')[-2]])
    frame_saved_full_path = os.path.join('/', image_saved_path, frame_save_path)
    if not os.path.exists(frame_saved_full_path):
        os.makedirs(frame_saved_full_path)

    cap = cv2.VideoCapture(video_path)
    # 分辨率-宽度
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # 分辨率-高度
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_index = 0
    frame_count = 0
    if cap.isOpened():
        success = True
    else:
        success = False
        print("读取失败!")

    while success:
        success, frame = cap.read()

        if frame_index % interval == 0 and frame is not None:
            resize_frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            # cv2.imwrite(each_video_save_full_path + each_video_name + "_%d.jpg" % frame_index, resize_frame)
            status = cv2.imwrite(os.path.join(frame_saved_full_path, "%s_%d.jpg" % (video_name, frame_count)),
                                 resize_frame)

            print("---> 正在保存第%d帧:" % frame_index, success)
            frame_count += 1

        frame_index += 1

    cap.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-dir', type=str, help='Directory containing videos')
    parser.add_argument('--video-formats', nargs='+', type=str, default=".mp4", help='Accepted video formats')
    parser.add_argument('--time-interval', type=int, default="60", help='Save frame with 60 time interval')
    # 添加需要处理的视频
    # video_src_path = "/home/yousixia/data/video/20210520"
    # video_formats = [".mp4"]
    opt = parser.parse_args()
    opt.video_dir = Path(opt.video_dir).as_posix()
    frames_save_path = opt.video_dir.split('/')[-1] + '-' + time.strftime("%Y%m%d-%H:%M:%S")
    videos = os.listdir(opt.video_dir)

    videos = list(filter(lambda x: filter_format(x, opt.video_formats), videos))
    videos = [os.path.join(opt.video_dir, video) for video in videos]
    n = len(videos)
    results = ThreadPool(n if n < os.cpu_count()
                         else os.cpu_count()).imap(
        lambda x: video2frame(*x),
        zip(videos, repeat(frames_save_path), repeat(opt.time_interval)))
    for result in results:
        if result is not None:
            result.wait()
            result.close()

