import unittest
import PackECGToTFR as P
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import ECGPreProcess as epp

ecgPaths = [
    '/some/database/folder1',
    '/some/database/folder2',]


class testPack(unittest.TestCase):

    def tearDown(self):
        # 寫入要一點時間 直接接著讀 會出問題
        time.sleep(0.5)

    def testsliceECGsignal_2(self):
        import numpy as np

        path1 = ecgPaths[0]
        _,_,signal,fs = P.parseWFDBData(path1, [1])
        print(fs)

        slices,fsList = epp.sliceECGsignal_2(signal,fs,125)
        if fsList == None: raise Exception("No Data!")

        plt.figure(1)
        plt.clf()
        plt.subplot(4,1,1)
        plt.plot(signal)
        for i in range(len(slices)):
            plt.subplot(4,3,i+4)
            plt.plot(slices[i])
            plt.text(0,0,fsList[i])
            if i >= 8: break
        plt.show()


    def testanalysisDataPath(self):
        print(" \n >>> test analysisDataPath <<<")
        a = P.analysisDataPath(self.testPath)
        for b in a:
            print(b)

    def testwriteSingleExample(self):
        print(" \n >>> test writeSingleExample <<<")
        # object = P.TFRDataProcessor("test.trf",True)

        # for path in ecgPaths:
        #     object.writeSingleExample(path)

    def testwritePackage(self):
        print(" \n >>> test writePackage <<<")
        object = P.TFRDataProcessor("test.trf",True)
        object.writePackage(self.testPath)

    def anothertestData(self):
        TFGenerator = tf.python_io.tf_record_iterator(path="test.trf")

        for i in TFGenerator:
            # 建立 Example
            example_R = tf.train.Example()

            # 解析來自於 TFRecords 檔案的資料
            example_R.ParseFromString(i)

            # 取出 height 這個 Feature
            str1 = example_R.features.feature['personName']

            str2 = example_R.features.feature['dataName']

            datastr = example_R.features.feature['data']
            print(str1)
            print(str2)
            print(type(datastr))

        TFGenerator.close()

    def testdecodeRecordFile(self):

        # object = P.TFRDataProcessor("test.trf")
        object = P.TFRDataProcessor("test.trf")
        mypn, mydn, mydata, fs = object.decodeRecordFile(1)

        init_op = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())

        with tf.Session()  as sess:
            # 初始化
            sess.run(init_op)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            for i in range(3):
                b1,b2,b3,b4 = sess.run([mypn, mydn, mydata, fs])
                print(" --- ")
                print(b1)
                print(b2)
                print(len(b3))
                print(b4)

    # 已經分離出去了
    # def testpeakdet(self):
    #     series = [0,0,0,2,0,0,0,-2,0,0,0,2,0,0,0,-2,0]
    #     maxtab, mintab = P.peakdet(series,.3)
    #     print("max tab : \n", maxtab)
    #     print("min tab : \n", mintab)

    def testwriteSlicePackage(self):
        object = P.TFRDataProcessor("test.trf",True)

        # for path in ecgPaths:

        object.writeSlicePackage(self.testPath)

    def testparseWFDBData(self):
        path1 = 'singe/wfdb/file/set'
        _,_,signal,fs = P.parseWFDBData(path1, [1])
        print(fs)

    def testreadWFBDBasicInfo(self):
        P.readWFBDBasicInfo(ecgPaths[0])

if __name__ == "__main__":
    suit = unittest.TestSuite()

    test = [
        # testPack("testpeakdet"), # 測試極大極小值演算
        # testPack("testanalysisDataPath"), # 資料檔路徑分析
        # testPack("testreadWFBDBasicInfo"), # 讀出單個檔案基本資料
        # testPack("testparseWFDBData"), # 讀出單個檔案 ecg signal
        # testPack("testsliceECGsignal_2"), # 測試 ECG 分段的功能

        # testPack("testwritePackage"), # 
        # testPack("testwriteSingleExample"), 
        # testPack("testwriteSlicePackage"), # 將 ECG 切斷後儲存成 tfrecord

        # testPack("anothertestData"),
        testPack("testdecodeRecordFile"),
        

            ]
    suit.addTests(test)

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suit)