########################################################################
#
# 함수나 클래스의 Cache-wrapper
#
# 하드 디스크에 객체 인스턴트를 만들거나 함수 호출 결과를 저장.
# 데이터를 지속적으로 쓴다면 매우 빠르고 쉽게 불러올 수 있다.
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

import os
import pickle
import numpy as np

########################################################################


def cache(cache_path, fn, *args, **kwargs):
    """
    함수나 클래스에 대한 Cache-wrapper. 만약 캐쉬 파일이 존재한다면
    자료는 불러와지고 반환되고, 그렇지 않다면 함수가 호출되고, 결과는 캐쉬를 위해 저장된다.
    fn 인자는 클래스가 될 수도 있다. 이 경우 객체 인스턴트가 만들어지고, 캐쉬파일에 저장된다

    :param cache_path:
        캐쉬 파일에 대한 파일 경로

    :param fn:
        호출할 함수나 클래스

    :param args:
        함수나 클래스 init 메소드에 대한 인자

    :param kwargs:
        함수나 클래스 init 메소드에 대한 키워드 인자

    :return:
        함수가 호출된 결과나 객체 인스턴트가 만들어진 것의 결과
    """

    # 캐쉬 파일이 존재한다면
    if os.path.exists(cache_path):
        # 파일로부터 캐쉬 데이터를 불러온다
        with open(cache_path, mode='rb') as file:
            obj = pickle.load(file)

        print("- Data loaded from cache-file: " + cache_path)
    else:
        # 캐쉬 파일이 존재하지 않는다면

        # 함수나 인자에 갖는 class 인스턴트를 호출
        obj = fn(*args, **kwargs)

        # 캐쉬 파일에 자료를 저장
        with open(cache_path, mode='wb') as file:
            pickle.dump(obj, file)

        print("- Data saved to cache-file: " + cache_path)

    return obj


########################################################################


def convert_numpy2pickle(in_path, out_path):
    """
    넘파이 파일을 피클로 바꿈

    자료를 저장하기 위해 넘파이를 사용한 캐쉬 함수의 첫번째 버젼
    모든 자료는 재계산되는 대신에, 이 함수를 사용해 캐쉬 파일로 바꿀 수 있다

    :param in_path:
        numpy.save()를 사용해 쓰여진 넘파일 포맷의 입력 파일

    :param out_path:
        피클 파일로 쓰여진 출력 파일

    :return:
        없음
    """

    # 넘파이를 사용해 데이터를 불러온다
    data = np.load(in_path)

    # 피클을 사용해 저장한다
    with open(out_path, mode='wb') as file:
        pickle.dump(data, file)


########################################################################

if __name__ == '__main__':
    # 캐쉬 파일을 사용한 짧은 예제

    # This is the function that will only get called if the result
    # is not already saved in the cache-file. This would normally
    # be a function that takes a long time to compute, or if you
    # need persistent data for some other reason.
    # 이 함수는 결과가 캐쉬파일에 이미 저장되지 않았다면 단순히 호출만을 한다
    # 이것을 일반적으로 계산하는데 오랜 시간이 걸리거나 몇몇 이유로
    # 지속적으로 사용할 필요가 있는 데이터에 사용된다.
    def expensive_function(a, b):
        return a * b

    print('Computing expensive_function() ...')

    # 이미 존재한다면 캐쉬파일로부터 결과를 불러온다,
    # 그렇지 않다면 expensive_function(a=123, b=456)을 계산하고
    # 다음 번을 위해 캐쉬 파일에 결과를 저장한다
    result = cache(cache_path='cache_expensive_function.pkl',
                   fn=expensive_function, a=123, b=456)

    print('result =', result)

    # Newline.
    print()

    # 캐쉬 파일에 객체를 저장하는 또다른 예제

    # 이 클래스의 객체 인스턴트를 캐쉬하고 싶다
    # 동기는 한번 계산하는데 비싼 계산이 수행되거나, 지속적으로 필요한 데이터라서 하는 것
    class ExpensiveClass:
        def __init__(self, c, d):
            self.c = c
            self.d = d
            self.result = c * d

        def print_result(self):
            print('c =', self.c)
            print('d =', self.d)
            print('result = c * d =', self.result)

    print('Creating object from ExpensiveClass() ...')

    # Either load the object from a cache-file if it already exists,
    # otherwise make an object-instance ExpensiveClass(c=123, d=456)
    # and save the object to the cache-file for the next time.
    # 이미 존재한다면 캐쉬파일로부터 결과를 불러온다,
    # 그렇지 않다면 객체 인스턴스 ExpensiveClass(a=123, b=456)를 만들고
    # 다음 번을 위해 캐쉬 파일에 객체를 저장한다
    obj = cache(cache_path='cache_ExpensiveClass.pkl',
                fn=ExpensiveClass, c=123, d=456)

    obj.print_result()

########################################################################
