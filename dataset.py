########################################################################
#
# 폴더에 모든 파일로 구성되는 데이터셋을 만들기 위한  클래스
#
# 예제 사용은 knifey.py 와 튜토리얼 #09 에서 보인다
#
# Implemented in Python 3.5
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
import os
from cache import cache

########################################################################


def one_hot_encoded(class_numbers, num_classes=None):
    """
    정수의 배열로부터 One-Hot 인코딩된 클래스 라벨을 만든다

    예를 들어, 만약 class_number=2 와 num_classes=4 라면
    one-hot 인코딩된 라벨은 floast 배열: [0., 0., 1., 0.]

    :param class_numbers:
        클래스 숫자를 갖는 정수의 배열
        이 정수들은 0부터 num_classes-1을 포함한다고 가정한다

    :param num_classes:
        Number of classes. If None then use max(cls)-1.
        클래스의 수. 만약 None 이라면 max(cls)-1을 사용한다

    :return:
        2차원 배열: [len(cls), num_classes]
    """

    # None이 주어지면, 클래스의 수를 찾는다
    if num_classes is None:
        num_classes = np.max(class_numbers) - 1

    return np.eye(num_classes, dtype=float)[class_numbers]


########################################################################


class DataSet:
    def __init__(self, in_dir, exts='.jpg'):
        """
        Create a data-set consisting of the filenames in the given directory
        and sub-dirs that match the given filename-extensions.
        주어진 확장자가 맞는 주어진 디렉토리와 하위 디렉토리의 파일명으로 구성된 데이터셋을 만든다

        예를 들어, knifey-spoony 데이터셋(knifey.py를 보라)은 아래의 폴더 구조를 갖는다:

        knifey-spoony/forky/
        knifey-spoony/knifey/
        knifey-spoony/spoony/
        knifey-spoony/forky/test/
        knifey-spoony/knifey/test/
        knifey-spoony/spoony/test/

        이것은 세개의 클래스를 의미한다 : forky, knifey, spoony

        만약 우리가 in_dif = "knifey-spoony/"와 새로운 데이터셋 객체를 만들면,
        이것은 이들 디렉토리를 통해 스캔하고 이들 클래스 각각에 대한 학습 데이터셋과 테스트 셋을 만든다

        학습 데이터셋은 아래 디렉토리의 모든 *.jpg 파일명의 리스트를 갖는다

        knifey-spoony/forky/
        knifey-spoony/knifey/
        knifey-spoony/spoony/

        테스트셋은 아래 디렉토리들의 모든 *.jpg 파일명의 리스트를 갖는다.

        knifey-spoony/forky/test/
        knifey-spoony/knifey/test/
        knifey-spoony/spoony/test/

        사용 예제는 튜토리얼 09를 보라

        :param in_dir:
            데이터 셋에 파일들에 대한 최상위 폴더
            위의 예제에서는 'knifey-spoony/'

        :param exts:
            문자열이거나 유효한 파일명과 확장자를 문자열의 튜플.
            대소문자를 구분하지 않는다

        :return:
            객체 인스턴스
        """

        # 완전한 경로로 입력 디렉토리를 확장한다(절대 주소)
        in_dir = os.path.abspath(in_dir)

        # 입력 디렉토리
        self.in_dir = in_dir

        # 모든 파일 확장자를 소문자로 바꾼다
        self.exts = tuple(ext.lower() for ext in exts)

        # 클래스 이름
        self.class_names = []

        # 학습데이터셋에 모든 파일들에 대한 파일명
        self.filenames = []

        # 테스트셋에 대한 모든 파일들에 대한 파일명
        self.filenames_test = []

        # 학습데이터셋에 각 파일에 대한 클래스 숫자
        self.class_numbers = []

        # 테스트셋에 각 파일에 대한 클래스 숫자
        self.class_numbers_test = []

        # 데이터셋에서 클래스의 총 수
        self.num_classes = 0

        # 입력 디렉토리안에 모든 파일과 디렉토리에 대해 반복
        for name in os.listdir(in_dir):
            # 이 파일과 디렉토리에 대한 완전한 경로
            current_dir = os.path.join(in_dir, name)

            # 만약 이게 디렉토리라면
            if os.path.isdir(current_dir):
                # 클래스 리스트에 디렉토리 이름을 추가
                self.class_names.append(name)

                # 학습 데이터셋

                # 이 디렉토리에서 모든 유효한 파일명을 얻는다
                filenames = self._get_filenames(current_dir)

                # 학습 데이터셋에 대한 모든 파일 리스트에 그것들을 더한다
                self.filenames.extend(filenames)

                # 클래스에 대한 클래스 숫자
                class_number = self.num_classes

                # 클래스 숫자의 배열을 만든다
                class_numbers = [class_number] * len(filenames)

                # 학습 데이터셋에 대한 모든 클래스 숫자 리스트에 그것들을 더한다
                self.class_numbers.extend(class_numbers)

                # Test-set.

                # 하위 폴더명 'test'안에 모든 유효한 파일명을 얻는다
                filenames_test = self._get_filenames(os.path.join(current_dir, 'test'))

                # 테스트셋에 모든 클래스 수샂 리스에 그것들을 더한다
                self.filenames_test.extend(filenames_test)

                # 클래스 숫자의 배열을 만든다
                class_numbers = [class_number] * len(filenames_test)

                # 테스트셋에 대한 모든 클래스 숫자 리스트에 그것들을 더한다
                self.class_numbers_test.extend(class_numbers)

                # 데이터셋의 총 클래스수를 증가시킨다
                self.num_classes += 1

    def _get_filenames(self, dir):
        """
        주어진 디렉토리 내에 맞는 확장자를 갖는 파일명의 배열을 만들고 반환한다

        :param dir:
            파일들에 대해 검색할 디렉토리. 하위 폴더는 안한다

        :return:
            파일들의 리스트. 단지 파일명, 디렉토리는 포함하지 않는다.
        """

        # 빈리스트를 초기화
        filenames = []

        # 만약 디렉토리가 존재하면
        if os.path.exists(dir):
            # 맞는 확장자를 갖는 모든 파일명을 얻는다
            for filename in os.listdir(dir):
                if filename.lower().endswith(self.exts):
                    filenames.append(filename)

        return filenames

    def get_paths(self, test=False):
        """
        데이터셋에 파일들에 대한 완전한 경로를 얻는다.

        :param test:
            Boolean. Return the paths for the test-set (True) or training-set (False).
            진리값, True면 테스트셋에 대한 경로를 반환하거나 False면 학습 데이터셋을 반환

        :return:
            경로 이름에 대한 문자열들을 갖는 이터레이터
        """

        if test:
            # 테스트셋에 대한 파일명들과 클래스 숫자들을 사용
            filenames = self.filenames_test
            class_numbers = self.class_numbers_test

            # 테스트셋에 대한 하위 폴더명
            test_dir = "test/"
        else:
            # 학습 데이터셋에 대한 파일명들과 클래스 숫자들을 사용
            filenames = self.filenames
            class_numbers = self.class_numbers

            # 테스트셋에 대한 하위 폴더명을 사용하지 않는다
            test_dir = ""

        for filename, cls in zip(filenames, class_numbers):
            # 이 파일에 대한 완전한 경로명
            path = os.path.join(self.in_dir, self.class_names[cls], test_dir, filename)

            yield path

    def get_training_set(self):
        """
        학습 데이터셋 안에 파일들에 대한 경로의 리스트와 정수로된 클래스의 숫자의 리스트와
        one-hot 인코딩된 클래스 숫자의 배열를 반환
        """

        return list(self.get_paths()), \
               np.asarray(self.class_numbers), \
               one_hot_encoded(class_numbers=self.class_numbers,
                               num_classes=self.num_classes)

    def get_test_set(self):
        """
        테스트셋 안에 파일들에 대한 경로의 리스트와 정수로된 클래스의 숫자의 리스트와
        one-hot 인코딩된 클래스 숫자의 배열를 반환
        """

        return list(self.get_paths(test=True)), \
               np.asarray(self.class_numbers_test), \
               one_hot_encoded(class_numbers=self.class_numbers_test,
                               num_classes=self.num_classes)


########################################################################


def load_cached(cache_path, in_dir):
    """
    이미 존재하면 캐쉬 파일을 불러오고, 그렇지 않으면 새로운 객체를 만들고 캐쉬파일에 저장하는
    데이터셋 객체를 만들기 위한 Wrapper 함수

    만약 파일명의 이름이 매번 데이터셋을 불러올 때 일치하는 것을 보장하기를 원한다면 유용하다
    예를 들면, 다른 캐쉬파일에 저장된 변환 값을 갖는 결합 데이터셋 객체를 사용하는 경우,
    e.g. 튜토리얼 #09 에서 이 예제를 볼 수 있다.

    :param cache_path:
        캐쉬파일에 대한 경로

    :param in_dir:
        데이터셋 안에 파일들에 대한 최상위 경로.
        데이터셋 init 메소드에 대한 인자

    :return:
        데이터셋 객체
    """

    print("Creating dataset from the files in: " + in_dir)

    # 만약 DataSet(in_dir=data_dir)의 객체 인스턴스가 이미 캐쉬파일안에 존재한다면 불러오고,
    # 그렇지 않다면 객체 인스턴스를 만들고 다음번을 위해 캐쉬파일에 저장한다
    dataset = cache(cache_path=cache_path,
                    fn=DataSet, in_dir=in_dir)

    return dataset


########################################################################
