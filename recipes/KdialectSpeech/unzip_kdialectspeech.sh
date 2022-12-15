#!/bin/bash

if [ $# -ne 2 ]
then
    echo "Usage: unzip_kdialect.sh <KdialectSpeech dir> <dest dir>"
fi

# KSPONPATH=$1
KDIALECTPATH=$1
DESTPATH=$2

mkdir -p $DESTPATH
# mkdir -p $DESTPATH/test

echo "expanding train data"
unzip "$KDIALECTPATH/중노년층방언/KsponSpeech_01.zip" -d $DESTPATH
unzip "$KDIALECTPATH/중노년층방언/KsponSpeech_02.zip" -d $DESTPATH
unzip "$KDIALECTPATH/중노년층방언/KsponSpeech_03.zip" -d $DESTPATH
unzip "$KDIALECTPATH/중노년층방언/KsponSpeech_04.zip" -d $DESTPATH
unzip "$KDIALECTPATH/중노년층방언/KsponSpeech_05.zip" -d $DESTPATH

# echo "expanding eval data"
# unzip "$KDIALECTPATH/평가용_데이터/KsponSpeech_eval.zip" -d $DESTPATH/test