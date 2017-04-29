########################################################################
#
# The pre-trained VGG16 Model for TensorFlow.
#
# 이 모델은 DeepDream 에서 잘 작동하는 Inception 5h 모델보다 Style Transfer 에서
# 더 좋게 보이는 이미지를 만들어내는 것 같다
#
# See the Python Notebook for Tutorial #15 for an example usage.
#
# Implemented in Python 3.5 with TensorFlow v0.11.0rc0
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
import tensorflow as tf
import download
import os

########################################################################
# 다양한 디렉토리와 파일명들

# 미리 학습된 VGG16 모델은 이 튜토리얼에서 얻어진다.
# https://github.com/pkmital/CADL/blob/master/session-4/libs/vgg16.py

# 클래스 이름은 다음 URL에서 가능하다
# https://s3.amazonaws.com/cadl/models/synset.txt

# VGG16 모델 파일에 대한 인터넷 URL
# 이건 미래에 바뀔 수 있고 업데이트할 필요가 있을 수 있다
data_url = "https://s3.amazonaws.com/cadl/models/vgg16.tfmodel"

# 다운로드된 데이터가 저장될 디렉토리
data_dir = "vgg16/"

# 텐서플로 그래프 정의를 포함하는 파일
path_graph_def = "vgg16.tfmodel"

########################################################################


def maybe_download():
    """
    data_dir 안에 이미 존재하지 않으면, 인터넷으로부터 VGG16 모델을
    다운로드 받는다. 경고! 이 파일은 약 550 MB 이다.
    """

    print("Downloading VGG16 Model ...")

    # 인터넷에서 이 파일은 압축된 포맷으로 저장되어 있지 않다.
    # 이 함수는 파일 확장자가 .zip 이나 tar.gz 이 아닌 경우에 추출해서는 안 된다.
    download.maybe_download_and_extract(url=data_url, download_dir=data_dir)


########################################################################


class VGG16:
    """
    VGG16 모델은 1000개의 다른 카테고리 속에 이미지를 분류하기 위해
    이미 학습된 심층 신경망이다.

    이 클래스의 새로운 인스턴스를 만들 때, VGG16 모델은 불러와지고,
    학습 없이 즉시 사용할 수 있다.
    """

    # 입력 이미지 보내기 위한 텐서의 이름
    tensor_name_input_image = "images:0"

    # dropout 된 무작위 값에 대한 텐서의 이름
    tensor_name_dropout = 'dropout/random_uniform:0'
    tensor_name_dropout1 = 'dropout_1/random_uniform:0'

    # Style Transfer에서 사용되기 위한 모델 안에 콘볼루션 레이어의 이름들
    layer_names = ['conv1_1/conv1_1', 'conv1_2/conv1_2',
                   'conv2_1/conv2_1', 'conv2_2/conv2_2',
                   'conv3_1/conv3_1', 'conv3_2/conv3_2', 'conv3_3/conv3_3',
                   'conv4_1/conv4_1', 'conv4_2/conv4_2', 'conv4_3/conv4_3',
                   'conv5_1/conv5_1', 'conv5_2/conv5_2', 'conv5_3/conv5_3']

    def __init__(self):
        # 파일로부터 모델을 불러온다. 이 방법은 약간 혼란스럽고
        # 몇가지 단계가 필요하다

        # 새로운 텐서플로 계산 그래프를 만든다
        self.graph = tf.Graph()

        # 새로운 그래프를 디폴트로 설정한다
        with self.graph.as_default():

            # 텐서플로 그래프는 다양한 플랫폼에서 동작하는 파일 형식인
            # Protocol Buffers 라 불리는 것이 디스크에 저장되어있다.
            # 이것은 바이너리 파일로 저장되어 있다.

            # 바이너리 읽기로 그래프 정의 파일을 연다
            path = os.path.join(data_dir, path_graph_def)
            with tf.gfile.FastGFile(path, 'rb') as file:
                # 그래프 정의는 텐서플로 그래프의 저장된 복사본이다.
                # 먼저 우리는 빈 그래프정의를 만들어야한다
                graph_def = tf.GraphDef()

                # 그래프정의 안에 proto-buf 파일을 불러온다
                graph_def.ParseFromString(file.read())

                # 마지막으로 기본 텐서플로 그래프에 그래프 정의를 불러온다
                tf.import_graph_def(graph_def, name='')

                # 이제 그래프는 proto-buf 파일로부터 VGG16 모델을 가지고 있다

            # 그래프에 입력되는 이미지를 갖는 텐서에 대한 참조점을 얻는다
            self.input = self.graph.get_tensor_by_name(self.tensor_name_input_image)

            # 공통적으로 사용되는 레이어에 대한 텐서들의 참조점을 얻는다
            self.layer_tensors = [self.graph.get_tensor_by_name(name + ":0") for name in self.layer_names]

    def get_layer_tensors(self, layer_ids):
        """
        주어진 id 를 갖는 레이어에 대한 텐서의 참조점 리스트를 반환
        """

        return [self.layer_tensors[idx] for idx in layer_ids]

    def get_layer_names(self, layer_ids):
        """
        주어진 id를 갖는 레이어의 이름 리스트를 반환
        """

        return [self.layer_names[idx] for idx in layer_ids]

    def get_all_layer_names(self, startswith=None):
        """
        그래프에서 모든 레이어의 리스트 반환
        이 리스트는 주어진 문자로 시작하는 이름들에 필터링될 수 있다
        """

        # 이 그래프에 모든 레이어에 대한 이름 리스트를 얻는다
        names = [op.name for op in self.graph.get_operations()]

        # 주어진 문자열로 시작하는 것만을 필터링한다
        if startswith is not None:
            names = [name for name in names if name.startswith(startswith)]

        return names

    def create_feed_dict(self, image):
        """
        # 이미지를 갖는 feed-dict을 만들고 반환한다

        :param image:
            이 입력 이미지는 이미 디코딩된 3차원 배열이다
            이 픽셀들은 0과 255 사이여야만 한다(float or int)

        :return:
            텐서플로 안에 인셉션 그래프에 보내기 위한 사전
        """

        # '빈' 차원을 추가해 3차원 배열을 4차원으로 확장한다.
        # 왜냐하면 하나의 이미지를 보낼 것이기 때문이지만,
        # 인셉션 모델은 여러개의 이미지들을 입력으로 갖도록 만들어졌다.
        image = np.expand_dims(image, axis=0)

        if False:
            # VGG16 모델을 사용하는 원본 코드에서, dropout에 대한
            # 무작위 값은 1.0으로 설정되어있다.
            # 실험은 Style Transfer에선 문제가 없는 것처럼 보이고,
            # GPU 에서는 오류를 일으킨다.
            dropout_fix = 1.0

            # 텐서플로에 입력 데이터를 위한 feed_dict을 만든다
            feed_dict = {self.tensor_name_input_image: image,
                         self.tensor_name_dropout: [[dropout_fix]],
                         self.tensor_name_dropout1: [[dropout_fix]]}
        else:
            # 텐서플로에 입력 데이터를 위한 feed_dict을 만든다
            feed_dict = {self.tensor_name_input_image: image}

        return feed_dict

########################################################################
