# coding=utf-8

import os
import cv2
from multiprocessing.pool import ThreadPool
from itertools import repeat

video_src_path = "/home/yousixia/data/video/20210520"
video_formats = [".mp4"]
frames_save_path = "/home/yousixia/data/labelme_20210520"
# width = 320
# height = 240
time_interval = 60


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
    each_video_save_full_path = os.path.join(frame_save_path, video_name)
    if not os.path.exists(each_video_save_full_path):
        os.makedirs(each_video_save_full_path)

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
        print("---> 正在读取第%d帧:" % frame_index, success)

        if frame_index % interval == 0 and frame is not None:
            resize_frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            # cv2.imwrite(each_video_save_full_path + each_video_name + "_%d.jpg" % frame_index, resize_frame)
            cv2.imwrite(each_video_save_full_path + "/%s_%d.jpg" % (video_name, frame_count), resize_frame)
            frame_count += 1

        frame_index += 1

    cap.release()


if __name__ == '__main__':
    videos = os.listdir(video_src_path)

    videos = list(filter(lambda x: filter_format(x, video_formats), videos))
    videos = [os.path.join(video_src_path, video) for video in videos]
    n = len(videos)
    results = ThreadPool(n if n < os.cpu_count()
                            else os.cpu_count()).imap(lambda x: video2frame(*x),
                                                      zip(videos, repeat(frames_save_path), repeat(time_interval)))
    for result in results:
        if result is not None:
            result.wait()
            result.close()

