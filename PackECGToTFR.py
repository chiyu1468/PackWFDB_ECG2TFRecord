

import wfdb
import os, re, sys
import numpy as np
import scipy as sp
import tensorflow as tf
import ECGPreProcess as epp

'''
這是處理ECG用的檔案
目前主要兩個功能
1. 把wfdb訊號整段轉成 tfrecord
2. 把wfdb訊號切成每一個心跳一筆 再轉成 tfrecord
3. 
'''

class TFRDataProcessor():
    def __init__(self, recordFile, writeMode = False, WantSampleFreq = 125, WantChannel = [1]):
        self.WantSampleFreq = WantSampleFreq
        self.WantChannel = WantChannel
        self.WantECGsliceLen = 125
        self.writeMode = writeMode
        self.recordFile = [recordFile]
        if self.writeMode:
            self.TFWriter = tf.python_io.TFRecordWriter(path=recordFile)

    # ============================= external method ==============================

    def writePackage(self, rootPath):
        if self.writeMode:
            a = analysisDataPath(rootPath)
            for b in a:
                personName, dataName, signal = parseWFDBData(b, self.WantSampleFreq, self.WantChannel)
                self.writeSingleExample(personName, dataName, signal)

    def writeSlicePackage(self, rootPath):

        if self.writeMode:
            a = analysisDataPath(rootPath)
            for singlePath in a:
                personName, dataName, signal, fs = parseWFDBData(singlePath, self.WantChannel)
                print("Now Doing : ", singlePath) # debug
                # signals = epp.sliceECGsignal_1(signal) # 第一版的切開
                signals, fsList = epp.sliceECGsignal_2(signal, fs, self.WantECGsliceLen) # 第二版的切開
                if fsList == None: continue # 這個 ecg 找不到 R-peak

                #TODO 把訊號塞進tfrecord
                for i in range(len(signals)):
                    # self.writeSingleExample(personName, dataName, signals[i]) # 第一版的切開
                    self.writeSingleExample(personName, dataName, signals[i], fsList[i]) # 第二版的切開

    def decodeRecordFile(self, num_epochs = None):

        TFReader = tf.TFRecordReader()

        queueList = tf.train.string_input_producer(string_tensor=self.recordFile,num_epochs=num_epochs)

        a,b = TFReader.read(queue=queueList)

        features = tf.parse_single_example(
            b,
            features={
                'personName': tf.VarLenFeature( tf.string),
                'dataName': tf.VarLenFeature( tf.string),
                'data': tf.VarLenFeature( tf.float32),
                'fs': tf.VarLenFeature( tf.float32),
            })

        rpersonName = tf.cast(x=features['personName'], dtype=tf.string)
        rdataName = tf.cast(x=features['dataName'], dtype=tf.string)
        # rdata = tf.cast(x=features['data'], dtype=tf.float32)
        rdata = tf.sparse_tensor_to_dense(features['data']) # 喔喔喔 終於看到曙光
        # rdata = tf.decode_raw(features['data'], tf.float32) # 不能用
        rfs = tf.sparse_tensor_to_dense(features['fs'])

        return rpersonName,rdataName,rdata, rfs

    # ============================= internal method ==============================

    def writeSingleExample(self, personName, dataName, signal, signalFS):

        TFeatures = tf.train.Features(feature={
            'personName': self._bytes_feature(bytes(personName, 'UTF8')),
            'dataName' : self._bytes_feature(bytes(dataName, 'UTF8')),
            'data' : self._float32_feature(signal.tolist()),
            'fs' : self._float32_feature([signalFS])
        })

        example_W = tf.train.Example(features=TFeatures)

        self.TFWriter.write(example_W.SerializeToString())


    def __del__(self):
        if self.writeMode:
            self.TFWriter.close()

    # 二進位資料
    @classmethod
    def _bytes_feature(cls, value):
        # ? 需要測試 [value] 不確定是否正確
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    # 整數資料
    @classmethod
    def _int64_feature(cls, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    # 浮點數資料
    @classmethod
    def _float32_feature(cls, value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))



# Copy from PackECGToH5
def analysisDataPath(ecg_path):
    '''
        if database file structure
        - root_folder
            - person1
                - ecg_singla file (.hea/.dat/.atr)
            - person2
            ... and so on

        this function will search all the ecg_single file under root_folder & person
        then return an iteration object that can generate ecg_single file's path.

    Usage Notice: ecg_path must end up with "/" mark, 
                  for example: ecg_path = "/a/b/c/" <--- GOOD!!!
                               ecg_path = "/a/b/c"  <--- FAIL!!!

    :param ecg_path: path of root_folder
    :return: iteration for each person's ECG data path

    '''
    # search root_folder
    dirs = os.listdir(ecg_path)
    # print(dirs) #debug
    for x in dirs:
        # for each person search its folder
        # 由於每一筆 WFDB data 是由三個檔案構成
        # 所以為了重複不同副檔名的檔案 使用set
        files = set()
        # print(ecg_path + x) #debug
        if os.path.isdir(ecg_path + x):
            # print(os.listdir(ecg_path + x)) # debug
            for y in os.listdir(ecg_path + x):
                files.add(y.split(".")[0])
        for z in files:
            yield ecg_path + x + "/" + z
            #print(x + "/" + z) # debug

# Copy from PackECGToH5
def parseWFDBData(path, channels, unitRatio = 1):
    '''
    Update:2019.3.7
    change 
        1. input param "path, sampleRate, channels" --> "path, channels"
        2. output "filename, signal" --> "filename, signal, sampleRate"

    Since the WFDB_data from web has variety of sample rate & channels
    This function just read out wanted "ONLY ONE" channel from WFDB file,
    And also split name and path for the file.

    IMPOTANT: read more than one channel will generate error data.

    :param path: WFDB file without extension name
    :param channels: which one channel you want? use "readWFBDBasicInfo" to check out.
    :param unitRatio: Set to 1 mean no change, otherwise you can change uV to mV...etc
    :return: SingleECG object
    '''

    # Part0. read WFDB file
    SR = wfdb.rdsamp(path, channels=channels)

    # Part1. read ECG signal
    fileSampleRate = SR[1]["fs"]
    flattenSignal = SR[0].flatten()
    flattenSignal = np.array(flattenSignal, dtype=np.float16)

    # Part2. Moving ADC zero & Adjust amplitude by muli unitRatio
    # this mean moving is just for change amplitude, 
    # when slicing signal may will moving baseline window by window
    fixedSignal = flattenSignal - np.mean(flattenSignal)
    if unitRatio != 1:
        fixedSignal = flattenSignal * unitRatio

    # 原來調整 sample rate 的方式, 但是只能調整倍頻, 所以改寫另外的程式用內差方式改變
    # # http://python.usyiyi.cn/documents/NumPy_v111/reference/generated/numpy.ndarray.strides.html
    # # 搭配 ndarray 方法的 strides 參數調整sample rate
    # x = int(SR[1]["fs"] // sampleRate)
    # y = flattenSignal.strides[0]
    # z = flattenSignal.size//x
    # np.ndarray(shape=(z,), buffer=flattenSignal, strides=(x*y)）

    # Part3. create name from path
    a = path.split("/")
    name = a[-3] + "." + a[-2] # + "." + a[-1]

    return name , a[-1] , fixedSignal, fileSampleRate

def readWFBDBasicInfo(path):
    '''
    just input a path and showing the infomation of the file,
    nothing else
    '''
    HDR = wfdb.rdheader(path)
    print("channel \t sig_name \t Unit \t baseline \t adc_zero")
    print("===============================================")
    for i in range(HDR.n_sig):
        print(" {} \t {} \t {} \t {} \t {} ".format(
                i, HDR.sig_name[i], HDR.units[i], HDR.baseline[i], HDR.adc_zero[i]  ))

if __name__ == '__main__':
    
    print("Making File...")
    object = TFRDataProcessor("/where/you/want/to/save/signal/file.trf",True, WantChannel = [1])
    object.writeSlicePackage("/where/you/wfdb/folder/are/")
    print("Finish!")




