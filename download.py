########################################################################
#
# 인터넷으로부터 데이터파일을 다운로드받고 추출하는 함수
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

import sys
import os
import urllib.request
import tarfile
import zipfile

########################################################################


def _print_download_progress(count, block_size, total_size):
    """
    다운로드 과정을 출력하기 위한 함수.
    maybe_download_and_extract() 함수 안에 뒤에 호출되는 함수로써 사용
    """

    # 퍼센트 계산
    pct_complete = float(count * block_size) / total_size

    # 상태 메시지. \r은 줄을 덮어 쓰라는 의미
    msg = "\r- Download progress: {0:.1%}".format(pct_complete)

    # 출력해라
    sys.stdout.write(msg)
    sys.stdout.flush()


########################################################################


def maybe_download_and_extract(url, download_dir):
    """
    만약 존재하지 않는다면 파일을 다운받고 추출한다
    url은 tar-ball 파일을 가정한다

    :param url:
        다운받기 위한 tar-file에 대한 인터넷 URL
        Example: "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

    :param download_dir:
        다운로드된 파일이 저장된 디렉토리
        Example: "data/CIFAR-10/"

    :return:
        없음
    """

    # 인터넷으로부터 다운로드된 파일을 저장하기 위한 파일명
    # 이 URL 로붵의 파일명을 사용하고, 다운로드 디렉토리에 그것을 더한다
    filename = url.split('/')[-1]
    file_path = os.path.join(download_dir, filename)

    # 만약 이미 파일이 존재하는지 확인
    # 만약 존재한다면 이것은 추출(압축해제)된다
    # 그렇지 않으면 다운로드하고 추출된다
    if not os.path.exists(file_path):
        # 다운로드 폴더가 존재하는지 확인, 없으면 만든다
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)

        # 인터넷으로부터 다운로드 받는다
        file_path, _ = urllib.request.urlretrieve(url=url,
                                                  filename=file_path,
                                                  reporthook=_print_download_progress)

        print()
        print("Download finished. Extracting files.")

        if file_path.endswith(".zip"):
            # zip_file을 푼다
            zipfile.ZipFile(file=file_path, mode="r").extractall(download_dir)
        elif file_path.endswith((".tar.gz", ".tgz")):
            # tar-ball 을 푼다
            tarfile.open(name=file_path, mode="r:gz").extractall(download_dir)

        print("Done.")
    else:
        print("Data has apparently already been downloaded and unpacked.")


########################################################################
