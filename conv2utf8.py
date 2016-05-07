# -*- coding: utf-8 -*-
"""
data/origin/*.txt 파일을 읽어서 인코딩을 감지한 후 UTF-8로 바꿉니다.

"""

import chardet # https://github.com/chardet/chardet
import glob
import codecs

def conv_file(fromfile, tofile):
    with open(fromfile, "r") as f:
        sample_text=f.read(10240)

    pred = chardet.detect(sample_text)

    if not pred['encoding'] in ('EUC-KR', 'UTF-8', 'CP949', 'UTF-16LE'):
        print "WARNING! Unknown encoding! : %s = %s" % (fromfile, pred['encoding'])
        pred['encoding'] = "CP949" # 못찾으면 기본이 CP949

    elif pred['confidence'] < 0.9:
        print "WARNING! Unsured encofing! : %s = %s / %s" % (fromfile, pred['confidence'], pred['encoding'])

    with codecs.open(fromfile, "r", encoding=pred['encoding'], errors="ignore") as f:
        with codecs.open(tofile, "w+", encoding="utf8") as t:
            all_text = f.read()
            t.write(all_text)


files=glob.glob("data/origin/*.txt") + glob.glob("data/origin/*.TXT")
for idx, fromfile in enumerate(files):

    tofile = fromfile.replace("data/origin/","data/conv/")

    print "[%d/%d] converting %s " % (idx+1 ,len(files), fromfile)
    conv_file(fromfile, tofile)

print "Convert Complete!"
