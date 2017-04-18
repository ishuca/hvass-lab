########################################################################
#
# 인터넷으로부터 CIFAR-10 데이터셋을 다운 받고 메모리로 불러오는 함수
#
# Implemented in Python 3.5
#
# 사용법:
# 1) 원하는 저장 경로를 갖는 data_path 변수를 설정
# 2) data_path 위치에 존재하지 않는다면 maybe_download_and_extract() 함수 호출
# 3) 클래스 이름의 배열을 얻는 load_class_names() 호출
# 4) 학습 데이터셋과 테스트셋에 대해 이미지와 클래스 숫자 그리고 one-hot 인코딩된 클래스 라벨을
#    얻는 load_training_data() 와 load_test_data() 호출
# 5) 프로그램에 반환 데이터 사용
#
# Format:
# 학습 데이터셋과 테스트 데이터셋의 이미지는 4차원 넘파이 배열 각각의 모양은:
# [image_number, height, width, channel]이고
# 각 픽셀은 0.0 과 1.0 사이의 float 이다
#
########################################################################
#
# This file is part of the TensorFlow Tutorials available at:
#
# https://github.com/Hvass-Labs/TensorFlow-Tutorials
#
# Published under the MIT License. See the file LICENSE for details.
#
# Copyright 2016 by Magnus Erik Hvass Pedersen
#
########################################################################

import numpy as np
import pickle
import os
import download
from dataset import one_hot_encoded

########################################################################

# 데이터셋을 다운받고 저장하기를 원하는 디렉토리
# 아래 함수들을 호출하기 전에 설정해야함
data_path = "data/CIFAR-10/"

# 데이터셋의 URL
data_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

########################################################################
# 이미지의 크기에 대한 다양한 상수
# 프로그램에서 이 상수들을 사용한다

# 각 이미지의 높이와 너비
img_size = 32

# 각 이미지 안에 채널 수, 3채널 : Red, Green, Blue
num_channels = 3

# 1차원으로 바꿨을 때 이미지의 길이
img_size_flat = img_size * img_size * num_channels

# 클래스의 수
num_classes = 10

########################################################################
# 올바른 크기의 배열을 할당하기 위해 사용되는 다양한 상수

# 학습 데이터셋의 파일의 수
_num_files_train = 5

# 학습 데이터셋에 각 배치 파일에 대한 이미지의 수
_images_per_file = 10000

# 학습 데이터셋에 이미지의 총 수
# 효율적으로 미리 배열을 할당하기 위해 사용한다
_num_images_train = _num_files_train * _images_per_file

########################################################################
# 다운로드, 압축풀기, 데이터 불러오기를 위한 private 함수


def _get_file_path(filename=""):
    """
    Return the full path of a data-file for the data-set.
    데이터셋에 대한 데이터 파일의 완전한 경로를 반환

    만약 filename이 공백이면 이 파일들의 디렉토리를 반환

    """

    return os.path.join(data_path, "cifar-10-batches-py/", filename)


def _unpickle(filename):
    """
    주어진 파일을 언피클 하고 데이터를 반환

    적절한 디렉토리 이름이 파일명 앞에 추가된다
    """

    # 이 파일에 대한 완전한 경로를 생성
    file_path = _get_file_path(filename)

    print("Loading data: " + file_path)

    with open(file_path, mode='rb') as file:
        # 파이썬 3.X 에서 encoding을 설정하는 것은 중요하다
        # 그렇지 않으면 에러가 발생한다
        data = pickle.load(file, encoding='bytes')

    return data


def _convert_images(raw):
    """
    Convert images from the CIFAR-10 format and
    CIFAR-10 형식으로부터 이미지를 변환하고
    픽셀은 0.0과 1.0 사이의 float 형식인 4차원 배열 [image_number, height, width, channel]을 반환
    """

    # 데이터파일로부터의 raw 이미지를 float로 바꾼다
    raw_float = np.array(raw, dtype=float) / 255.0

    # 배열을 4차원으로 재형성한다
    images = raw_float.reshape([-1, num_channels, img_size, img_size])

    # 배열의 인덱스를 순서를 조정한다
    # channel, height, width 였던 것을
    # height, width, channel로 변경
    images = images.transpose([0, 2, 3, 1])

    return images


def _load_data(filename):
    """
    픽클된 데이터파일을 불러오고 변환된 이미지와 클래스 숫자를 반환
    """

    # 피클된 데이터파일을 불러온다
    data = _unpickle(filename)

    # raw 이미지를 얻는다
    raw_images = data[b'data']

    # 각 이미지에 대한 클래스 숫자를 얻는다. 넘파이배열로 바꾼다
    cls = np.array(data[b'labels'])

    # 이미지를 변환한다
    images = _convert_images(raw_images)

    return images, cls


########################################################################
# 인터넷으로부터 데이터셋을 다운로드 받고 메모리에 불러오는 Public 함수


def maybe_download_and_extract():
    """
    data_path(원하는 경로를 처음에 설정해라) 에 존재하지 않는다면 CIFAR-10 데이터셋을
    다운로드 하고 추출한다
    """

    download.maybe_download_and_extract(url=data_url, download_dir=data_path)


def load_class_names():
    """
    Load the names for the classes in the CIFAR-10 data-set.
    CIFAR-10 데이터셋에서 클래스에 대한 이름을 불러온다.

    이름을 갖는 리스트틀 반환한다. 예: names[3]은 클래스 숫자 3에 해당하는 이름이다

    """

    # 피클된 파일로부터 클래스 이름을 불러온다
    raw = _unpickle(filename="batches.meta")[b'label_names']

    # 바이너리 스트링으로부터 변환한다
    names = [x.decode('utf-8') for x in raw]

    return names


def load_training_data():
    """
    Load all the training-data for the CIFAR-10 data-set.
    CIFAR-10 데이터셋에 대한 학습데이터셋을 불러온다.

    5개의 파일 속에 나뉘어진 이 데이터셋은 여기서 병합된다

    이미지, 클래스 숫자, one-hot 인코딩된 클래스 라벨을 반환한다.

    """

    # 효율성을 위해 이미지와 클래스 숫자에 대한 배열을 미리 할당한다
    images = np.zeros(shape=[_num_images_train, img_size, img_size, num_channels], dtype=float)
    cls = np.zeros(shape=[_num_images_train], dtype=int)

    # 현재 배치에 대한 시작 인덱스
    begin = 0

    # 각 데이터 파일에 대해 반복
    for i in range(_num_files_train):
        # 데이터파일로부터 클래스 숫자와 이미지를 불러온다
        images_batch, cls_batch = _load_data(filename="data_batch_" + str(i + 1))

        # 이 배치 안에 이미지의 수
        num_images = len(images_batch)

        # 현재 배치에 대한 끝 인덱스
        end = begin + num_images

        # 배열속에 이미지를 저장한다
        images[begin:end, :] = images_batch

        # 배열속에 클래스 숫자를 저장한다
        cls[begin:end] = cls_batch

        # 다음 배치를 위해 시작인덱스는 현재의 끝 인덱스이다.
        begin = end

    return images, cls, one_hot_encoded(class_numbers=cls, num_classes=num_classes)


def load_test_data():
    """
    CIFAR-10 데이터셋에 대한 테스트셋을 불러온다.

    이미지, 클래스 숫자, one-hot 인코딩된 클래스 라벨을 반환한다.
    """

    images, cls = _load_data(filename="test_batch")

    return images, cls, one_hot_encoded(class_numbers=cls, num_classes=num_classes)

########################################################################
