# -*- coding: utf-8 -*-
"""
입력 파일들의 형태로를 분리해서 input.txt파일에 저장합니다.

한글 한 글자가 분리되는 형식은 다음과 같습니다.

[초성][중성][종성][조합문자]

"""

from Hangulpy.Hangulpy import * # https://github.com/rhobot/Hangulpy
import codecs
import glob

def dump_file(filename):
    result=u""

    with codecs.open(filename,"r", encoding="UTF8") as f:
        for line in f.readlines():
            line = tuple(line)
            result = result + decompose_text(line)

    return result


files=glob.glob("data/conv/*.txt") + glob.glob("data/conv/*.TXT")

for f in files:
    print dump_file(f).encode("utf8")