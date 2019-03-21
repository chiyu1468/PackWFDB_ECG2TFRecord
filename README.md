# PackWFDB_ECG2TFRecord
read ecg signal from wfdb format, slice ecg become piece peak by peak, and pack them into tfrecord.

file descript:
    ECGPreProcess.py -> handle ecg slicing part
    PackECGToTFR.py -> a class deal with tfrecord package
    testPackECGToTFR.py -> unit test file
