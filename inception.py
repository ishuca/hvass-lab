########################################################################
#
# The Inception Model v3 for TensorFlow.
#
# input-image is of each class.
# 이미지를 분류하기 위해 미리 학습된 심층신경망이다.
# 이미지나 jpeg 파일명을 주면, 읽어서 인셉션 모델의 입력되고,
# 이미지가 어떤 클래스인지 나타내는 배열을 출력한다
#
# 맨 밑에 있는 예제나 노트북 파일을 보라
#
# 튜토리얼 7은 인셉션 모델을 사용하는 법을 보여준다
# 튜토리얼 8은 전이 학습을 위해 인셉션 모델을 사용하는 법을 보여준다
#
# 전이학습(Transfer Learning)이란 무엇인가?
#
# 전이학습은 학습된 것보다 다른 데이터셋의 이미즈를 분류하기 위해 신경망을 사용하는 것이다.
# 예를 들어, 인셉션 모델은 매우 강력하고 비싼 컴퓨터를 사용해 이미지넷 데이터셋에 학습되어있다.
# 하지만 인셉션 모델은 심지어 두개의 데이터셋에 클래스의 수가 다른 경우에도 
# 전체 모델을 다시 학습하는 것 없이 학습되지 않은 데이터셋에 재 사용할 수 있다.
# 이것은 학습하기 위해 강력하고 비싼 컴퓨터 필요없이 
# 자신의 데이터셋에 대해 인셉션 모델을 사용할 수 있게 한다.
#
# 소프트맥스 분류기 전에 인셉션 모델의 마지막 레이어는 전이 레이어(Transfer Layer)라 불린다.
# 왜냐하면 이 레이어의 출력이 자신의 데이터셋에 
# 학습 될 새로운 소프트맥스 분류기에 입력으로 사용될 것이기 때문이다
#
# 전이 레이어의 출력 값은 전이 값(Transfer Values)라고 불린다.
# 이들은 새롭게 만들 또다른 신경망이나 소프트맥스 분류기의 입력이 될 실제 값이다.
#
# 'bottleneck'은 때대로 전이 레이어나 전이 값을 가르키는 말로 사용된다 하지만
# 여기서는 혼란스러워 사용하지 않는다.
#
# Implemented in Python 3.5 with TensorFlow v0.10.0rc0
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
from cache import cache
import os
import sys

########################################################################
# 바꿀 수 있는 디렉토리와 파일명들

# 인셉션 모델을 갖는 tar-file에 대한 인터넷 URL
# 미래에 바뀔 수 있고 업데이트될 필요가 있다
data_url = "http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz"

# 다운로드한 자료를 저장할 디렉토리
data_dir = "inception/"

# 다운로드된 클래스 숫자와 uid 사이의 맵핑을 포함한 파일
path_uid_to_cls = "imagenet_2012_challenge_label_map_proto.pbtxt"

# uid와 string 사이의 매핑을 포함한 파일
path_uid_to_name = "imagenet_synset_to_human_label_map.txt"

# 텐서플로 그래프 정의를 포함한 파일
path_graph_def = "classify_image_graph_def.pb"

########################################################################


def maybe_download():
    """
    data_dir에 존재하지 않으면, 인터넷으로부터 인셉션 모델을 다운로드한다
    이 파일은 약 85 MB 다.
    """

    print("Downloading Inception v3 Model ...")
    download.maybe_download_and_extract(url=data_url, download_dir=data_dir)


########################################################################


class NameLookup:
    """
    클래스 숫자를 갖는 이름을 보기 위해 사용된다.
    이것은 이들 숫자 대신에 클래스의 이름을 출력하는데 사용된다.
    e.g. "plant" or "horse".
    
    Maps between:
    - cls 는 1부터 1000의 정수로 된 클래스 숫자
    - uid 는 이미지 데이터셋의 문자로된 클래스 아이디. e.g. "n00017222"
    - 이름은 문자로 된 클래스 이름, e.g. "plant, flora, plant life"

    인셉션 모델는 실제로 1008개의 클래스이지만,
    매핑 파일에 1000개의 이름 밖에 없다.
    남아있는 8개의 클래스는 사용되지 않는다.
    """

    def __init__(self):
        # uid, cls , name 사이의 매핑은 평균적으로 O(1)의 시간 사용을 갖는 사전이다.
        # 하지만 가장 나쁜 경우 O(n)
        self._uid_to_cls = {}   # uid 로부터 cls 매핑
        self._uid_to_name = {}  # uid 로부터 name 매핑
        self._cls_to_uid = {}   # clf 로부터 uid 매핑

        # 파일로부터 uid - name 매핑을 읽어온다
        path = os.path.join(data_dir, path_uid_to_name)
        with open(file=path, mode='r') as file:
            # 파일로부터 모든 줄을 읽는다
            lines = file.readlines()

            for line in lines:
                # 새줄 단어 삭제
                line = line.replace("\n", "")

                # 탭으로 나눈다
                elements = line.split("\t")

                # uid를 꺼낸다
                uid = elements[0]

                # 클래스 이름을 꺼낸다
                name = elements[1]

                # 룩업 사전에 삽입
                self._uid_to_name[uid] = name

        # 파일로부터 uid - cls 매핑을 읽어온다
        path = os.path.join(data_dir, path_uid_to_cls)
        with open(file=path, mode='r') as file:
            # 파일로부터 모든 줄을 읽는다
            lines = file.readlines()

            for line in lines:
                # 파일은 적절한 포맷이라고 가정한다
                # 그러므로 다음 줄들은 쌍으로 나오고, 다른 줄들은 무시된다

                if line.startswith("  target_class: "):
                    # 이 줄은 정수로된 클래스 숫자여야한다

                    # 줄을 나눈다
                    elements = line.split(": ")

                    # 정수로된 클래스 숫자를 꺼낸다
                    cls = int(elements[1])

                elif line.startswith("  target_class_string: "):
                    # 이 줄은 문자로 된 uid 여야 한다

                    # 줄을 나눈다
                    elements = line.split(": ")

                    # 문자로 된 uid를 꺼낸다 e.g. "n01494475"
                    uid = elements[1]

                    # 쌍따옴표를 지운다
                    uid = uid[1:-2]

                    # uid와 cls 사이를 룩업 테이블에 삽입한다
                    self._uid_to_cls[uid] = cls
                    self._cls_to_uid[cls] = uid

    def uid_to_cls(self, uid):
        """
        주어진 uid-String의 정수로 된 클래스 숫자를 반환
        """

        return self._uid_to_cls[uid]

    def uid_to_name(self, uid, only_first_name=False):
        """
        주어진 uid 에 대해 클래스 이름 반환

        몇몇 클래스 이름은 이름의 리스트이다. 만약 첫번째 이름만 원한다면 only_first_name=True 설정하라
        """

        # uid로부터 이름을 꺼낸다
        name = self._uid_to_name[uid]

        # 리스트에서 첫번째만을 사용할 것인가?
        if only_first_name:
            name = name.split(",")[0]

        return name

    def cls_to_name(self, cls, only_first_name=False):
        """
        정수 클래스 숫자로부터 클래스 이름 반환

        몇몇 클래스 이름은 이름의 리스트이다. 만약 첫번째 이름만 원한다면 only_first_name=True 설정하라
        """

        # cls 로부터 uid를 꺼낸다
        uid = self._cls_to_uid[cls]

        # uid로부터 이름을 꺼낸다
        name = self.uid_to_name(uid=uid, only_first_name=only_first_name)

        return name


########################################################################


class Inception:
    """
    인셉션 모델은 1000개의 다른 카테고리 속에 이미지를 분류하기 위해 이미 학습된 심층 신경망이다.

    이 클래스의 새로운 인스턴스를 만들 때, 인셉션 모델은 불러오고 학습 없이 즉시 사용가능하다.

    인셉션 모델은 전이 학습을 위해 사용될 수 있다
    """

    # jpeg 입력 이미지에 대한 텐서의 이름
    tensor_name_input_jpeg = "DecodeJpeg/contents:0"

    # 디코드된 입력 이미지에 대한 텐서의 이름
    # jpeg 가 아닌 다른 형식의 이미지를 보내기 위해 사용
    tensor_name_input_image = "DecodeJpeg:0"

    # 리사이즈된 입력 이미지에 대한 텐서의 이름
    # 리사이즈된 후에 이미지를 가져오기 위해 사용
    tensor_name_resized_image = "ResizeBilinear:0"

    # 소프트맥스 분류기의 출력에 대한 텐서의 이름
    # 인셉션 모델을 가지고 이미지를 분류하기 위해 사용
    tensor_name_softmax = "softmax:0"

    # 소프트맥스 분류기의 정규화되지 않은 출력에 대한 텐서의 이름 (aka. logits)
    tensor_name_softmax_logits = "softmax/logits:0"

    # 인셉션 모델의 출력에 대한 텐서의 이름
    # 전이 학습을 위해 사용된다
    tensor_name_transfer_layer = "pool_3:0"

    def __init__(self):
        # 클래스 숫자와 클래스 이름 사이의 매핑
        # 클래스 이름은 문자로 출력된다 e.g. "horse" or "plant"
        self.name_lookup = NameLookup()

        # 파일로부터 인셉션 모델을 불러오자. 
        # 텐서플로로 이걸 하는 방법은 조금 혼란스럽고
        # 몇단계를 거쳐야한다

        # 새로운 텐서플로 계산 그래프를 만든다.
        self.graph = tf.Graph()

        # 새로운 그래프를 default로 설정한다
        with self.graph.as_default():

            # 텐서플로 그래프는 Protocol Buffers
            # proto-bufs 라 불리는 다양한 플랫폼에서 작동하는 파일 형식으로 디스크에 저장되어 있다
            # 이경우 바이너리 파일로 저장되어 있다

            # 바이너리로 그래프 정의 파일을 연다
            path = os.path.join(data_dir, path_graph_def)
            with tf.gfile.FastGFile(path, 'rb') as file:
                # 이 그래프 정의(graph_def)를 텐서플로 그래프에 복사한다.
                # 첫째로 우린 빈 그래프 정의가 필요하다
                graph_def = tf.GraphDef()

                # 그래프 정의 속에 proto-buf 파일을 불러온다
                graph_def.ParseFromString(file.read())

                # 마지막으로 이 graph-def를 default 텐서플로 그래프로 불러온다
                tf.import_graph_def(graph_def, name='')

                # 이제 self.graph는 proto-buf 파일로부터 인셉션 모델을 가지고 있다

        # 소프트맥스 분류기의 출력에 대한 적절한 이름을 갖는
        # 텐서를 불러옴으로써 인셉션 모델의 출력을 얻자
        self.y_pred = self.graph.get_tensor_by_name(self.tensor_name_softmax)

        # 인셉션 모델의 정규화되지 않은 출력을 얻자 (aka. softmax-logits)
        self.y_logits = self.graph.get_tensor_by_name(self.tensor_name_softmax_logits)

        # 신경망에 입력될 리사이즈된 입력에 대한 텐서를 얻자
        self.resized_image = self.graph.get_tensor_by_name(self.tensor_name_resized_image)

        # 그래프의 마지막 레이어에 대한 텐서를 얻자. aka. 전이 레이어
        self.transfer_layer = self.graph.get_tensor_by_name(self.tensor_name_transfer_layer)

        # 전이 레이어 안에 요소의 수를 얻자
        self.transfer_len = self.transfer_layer.get_shape()[3]

        # 그래를 실행하기 위한 텐서플로 세션을 만든다
        self.session = tf.Session(graph=self.graph)

    def close(self):
        """
        인셉션 모델을 사용하는 것이 끝났을 때 이 함수를 호출한다
        이것은 텐서플로 세션을 닫고 자원을 풀어준다
        """

        self.session.close()

    def _write_summary(self, logdir='summary/'):
        """
        summary-file에 그래프를 쓰므로 텐서보드에서 볼 수 있다

        이 함수는 디버깅을 위해 사용되고 미래에 제거되거나 바뀔 수도 있다

        :param logdir:
            summary-file들을 쓰기 위한 디렉토리

        :return:
            Nothing.
        """

        writer = tf.train.SummaryWriter(logdir=logdir, graph=self.graph)
        writer.close()

    def _create_feed_dict(self, image_path=None, image=None):
        """
        이미지를 갖는 feed-dict을 만들고 반환한다

        :param image_path:
            입력 이미지는 이 파일 경로를 갖는 jpeg 파일이다

        :param image:
            입력 이미지는 이미 디코딩된 3차원 배열이다
            이 픽셀은 0에서 255 사이의 값어야만 한다

        :return:
            텐서플로에 인셉션 그래프에 보내기 위한 사전
        """

        if image is not None:
            # 이미지는 이미 디코딩된 3차원 배열로 보내진다
            feed_dict = {self.tensor_name_input_image: image}

        elif image_path is not None:
            # 바이트의 배열로 jpeg 이미지는 읽어진다
            image_data = tf.gfile.FastGFile(image_path, 'rb').read()

            # 이미지는 jpeg로 인코딩된 이미지로 전해진다
            feed_dict = {self.tensor_name_input_jpeg: image_data}

        else:
            raise ValueError("Either image or image_path must be set.")

        return feed_dict

    def classify(self, image_path=None, image=None):
        """
        한장의 이미지를 분류하기 위해 인셉션 모델을 사용하기

        이 이미지는 자동적으로 299 x 299 픽셀로 리사이즈될 것이고,
        튜토리얼 7의 노트북에서 볼 수 있다

        :param image_path:
            입력 이미지는 이 파일 경로를 갖는 jpeg 파일이다

        :param image:
            입력 이미지는 이미 디코딩된 3차원 배열이다
            이 픽셀은 0에서 255 사이의 값어야만 한다

        :return:
            인셉션 모델이 이미지가 각 클래스일 거라고 생각하는 가능성을 나타내는 float의 배열 (aka. softmax-array)
        """

        # 입력 이미지와 가지고 텐서플로 그래프를 위한 feed_dict을 만든다
        feed_dict = self._create_feed_dict(image_path=image_path, image=image)

        # 예측 라벨을 얻기 위해 텐서플로 세션을 실행한다
        pred = self.session.run(self.y_pred, feed_dict=feed_dict)

        # 1차원 배열로 줄인다
        pred = np.squeeze(pred)

        return pred

    def get_resized_image(self, image_path=None, image=None):
        """
        인셉션 모델에 이미지를 넣고 리사이즈된 이미지를 반환
        이 리사이즈된 이미지는 신경망이 이들의 입력으로 무엇을 보는지 볼 수 있다

        :param image_path:
            입력 이미지는 이 파일 경로를 갖는 jpeg 파일이다

        :param image:
            입력 이미지는 이미 디코딩된 3차원 배열이다
            이 픽셀은 0에서 255 사이의 값어야만 한다

        :return:
            이미지를 갖는 3 차원 배열
        """

        # 입력 이미지와 가지고 텐서플로 그래프를 위한 feed_dict을 만든다
        feed_dict = self._create_feed_dict(image_path=image_path, image=image)

        # 예측 라벨을 얻기 위해 텐서플로 그래프를 실행
        resized_image = self.session.run(self.resized_image, feed_dict=feed_dict)

        # 4차원 텐서의 1차원 제거
        resized_image = resized_image.squeeze(axis=0)

        # 픽셀들은 0.0 과 1.0 사이로 조정
        resized_image = resized_image.astype(float) / 255.0

        return resized_image

    def print_scores(self, pred, k=10, only_first_name=True):
        """
        상위 k 개 예측 클래스에 대한 점수( or 확률) 출력

        :param pred:
            예측 클래스 라벨은 predict() 함수로부터 출력된다

        :param k:
            몇개 클래스까지 출력할 것인가?

        :param only_first_name:
            몇몇 클래스 이름은 이름의 리스트이다. 만약 첫번째 이름만 원한다면 only_first_name=True 설정하라

        :return:
            Nothing.
        """

        # 예측 배열에 대한 정렬된 인덱스를 얻는다
        idx = pred.argsort()

        # 가장 작은 것부터 높은 순으로 정렬되어있다. 뒤에서 k개를 선택한다
        top_k = idx[-k:]

        # 역순으로 상위 k 클래스를 반복한다 (i.e. 높은게 우선)
        for cls in reversed(top_k):
            # 클래스 이름 검색
            name = self.name_lookup.cls_to_name(cls=cls, only_first_name=only_first_name)

            # 이 클래스에 대한 예측 점수 (or 확률)
            score = pred[cls]

            # 점수와 클래스 이름을 출력한다
            print("{0:>6.2%} : {1}".format(score, name))

    def transfer_values(self, image_path=None, image=None):
        """
        주어진 이미지에 대한 전이 값을 계산한다.
        이들은 소프트맥스 레이어 전에 인셉션 모델의 마지막 레이어의 값들이다.

        전이 값은 다른 데이터셋과 다른 분류에 대한 전이 학습 에서
        인셉션 모델을 사용할 수 있게 한다

        데이터셋에 모든 이미지에 대한 전이값을 계산하는 것은 몇시간 이상이 걸릴 수도 있다.
        그러므로 transfer_values_cache() 함수를 사용해 결과를 캐쉬하는 것이 유용하다.

        :param image_path:
            입력 이미지는 이 파일 경로를 갖는 jpeg 파일

        :param image:
            입력 이미지는 이미 디코딩된 3차원 배열
            픽셀들은 0에서 255 사이여야만 한다

        :return:
            이들 이미지에 대한 전이 값들
        """

        # 입력 이미지를 갖는 텐서플로 그래프에 대한 feed_dict을 만든다
        feed_dict = self._create_feed_dict(image_path=image_path, image=image)

        # 인셉션 모델을 위한 그래프를 실행시키기 위해 텐서플로를 사용한다
        # 이것은 전이값들이라 불리는 소프트맥스 분류 전에 인셉션 모델의 마지막 레이어에 대한 값을 계산한다
        transfer_values = self.session.run(self.transfer_layer, feed_dict=feed_dict)

        # 1차원 배열로 줄인다
        transfer_values = np.squeeze(transfer_values)

        return transfer_values


########################################################################
# Batch-processing.


def process_images(fn, images=None, image_paths=None):
    """
    각 이미지에 대해 fn() 함수를 호출한다. e.g. 위의 인셉션 모델로부터의 transfer_values()
    모든 결과는 결합되고 반환된다.

    :param fn:
        각 이미지에 대해 호출할 함수

    :param images:
        처리될 이미지의 리스트

    :param image_paths:
        처리할 이미지에 대한 파일 경로의 리스트

    :return:
        결과를 갖는 넘파이 배열
    """

    # images를 사용할 것인가? image_paths를 사용할 것인가?
    using_images = images is not None

    # 이미지의 수
    if using_images:
        num_images = len(images)
    else:
        num_images = len(image_paths)

    # 이 결과에 대한 미리 할당된 리스트
    # 이것은 다른 배열에 대한 참조를 갖는다. 참조는 None으로 초기화된다
    result = [None] * num_images

    # 각 입력 이미지에 대해 반복
    for i in range(num_images):
        # 상태 메시지. \r은 현재 줄을 덮어씀을 의미한다
        msg = "\r- Processing image: {0:>6} / {1}".format(i+1, num_images)

        # 상태 메시지 출력
        sys.stdout.write(msg)
        sys.stdout.flush()

        # 나중에 사용하기 위해 이미지를 처리하고 결과를 저장하라
        if using_images:
            result[i] = fn(image=images[i])
        else:
            result[i] = fn(image_path=image_paths[i])

    print()

    # 결과를 넘파이 배열로 바꾼다
    result = np.array(result)

    return result


########################################################################


def transfer_values_cache(cache_path, model, images=None, image_paths=None):
    """
    이 함수는 만약 이미 게산되었다면 전이값을 불러오고, 그렇지 않다면
    값을 계산하고 나중에 다시 불러올 수 있는 파일에 쓴다

    왜냐하면 전이 값은 계산하기에 매우 비쌀 수 있다. 인셉션 모델에서
    직접적으로 transfer_values()를 호출하는 것 대신에 이 함수를 통해 값을
    캐쉬하는 것이 유용할 수 있다

    튜토리얼 8에서 이 함수를 사용하는 예제를 볼 수 있다

    :param cache_path:
        이미지에 대한 캐쉬된 전이값을 갖고 있는 파일

    :param model:
        인셉션 모델의 인스턴스

    :param images:
        이미지를 갖는 4차원 배열 [image_number, height, width, colour_channel]

    :param image_paths:
        이미지들에 대한 파일 경로의 배열 (jpeg 포맷이어야만 한다)

    :return:
        이들 이미지에 대한 인셉션 모델로부터 전이값
    """

    # 만약 캐쉬 파일이 존재하지 않는다면 이미지를 처리하기 위한 도움 함수
    # 이것은 필요하다 왜냐하면 cache() 함수에 fn=model.transfer_values와 fn=process_images를 둘다 제공하지 않기 때문이다
    def fn():
        return process_images(fn=model.transfer_values, images=images, image_paths=image_paths)

    # 캐쉬파일로부터 전이값을 읽어오거나 파일이 존재하지 않는다면 그들을 계산하라
    transfer_values = cache(cache_path=cache_path, fn=fn)

    return transfer_values


########################################################################
# 사용 예제

if __name__ == '__main__':
    print(tf.__version__)

    # 존재하지 않는다면 인셉션 모델을 다운로드한다
    maybe_download()

    # 인셉션 모델을 불러오므로써 이미지를 분류할 준비 되었다
    model = Inception()

    # 다운로드한 파일을 포함한 jpeg_image에 대한 경로
    image_path = os.path.join(data_dir, 'cropped_panda.jpg')

    # 이 이미지를 분류하기 위해 인셉션 모델을 사용한다
    pred = model.classify(image_path=image_path)

    # 상위 10개 예측에 대한 점수와 이름을 출력한다
    model.print_scores(pred=pred, k=10)

    # 텐서플로 세션을 닫는다
    model.close()

    # 전이 학습은 튜토리얼 8에서 시연한다

########################################################################
