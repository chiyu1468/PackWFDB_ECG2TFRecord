
import numpy as np
import scipy as sp

# =============== 第一種 ECG 切斷與偵測 ================

# Copy from https://gist.github.com/endolith/250860
def peakdet(v, delta, x = None):
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html
    
    Returns two arrays
    
    function [maxtab, mintab]=peakdet(v, delta, x)
    %PEAKDET Detect peaks in a vector
    %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    %        maxima and minima ("peaks") in the vector V.
    %        MAXTAB and MINTAB consists of two columns. Column 1
    %        contains indices in V, and column 2 the found values.
    %      
    %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    %        in MAXTAB and MINTAB are replaced with the corresponding
    %        X-values.
    %
    %        A point is considered a maximum peak if it has the maximal
    %        value, and was preceded (to the left) by a value lower by
    %        DELTA.
    
    % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.
    
    """
    maxtab = []
    mintab = []
       
    if x is None:
        x = np.arange(len(v))
    
    v = np.asarray(v)
    
    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')
    
    if not np.isscalar(delta):
        sys.exit('Input argument delta must be a scalar')
    
    if delta <= 0:
        sys.exit('Input argument delta must be positive')
    
    mn, mx = np.Inf, -np.Inf
    mnpos, mxpos = np.NaN, np.NaN
    
    lookformax = True
    
    for i in np.arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]
        
        if lookformax:
            if this < mx-delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return np.array(maxtab), np.array(mintab)

def createFilter():
    # 三角波 長度29 數值0~1
    a = np.linspace(start = 0.0, stop = 1.0, num = 14 , endpoint= False)
    b = np.linspace(start = 1.0, stop = 0.0, num = 15 , endpoint= True)
    ret = np.append(a,b)
    return ret

filter1 = createFilter()

def sliceECGsignal_1(input, filter = filter1):
    '''
    functions info>
        輸入一個ECG訊號 經過拆解以後 回傳多段的訊號
        拆解的概念： 卷積後找最小值 作為切割點
    generics>
        filter : ndarray
        input : ndarray
        return: list<ndarray>
    '''
    # 消除mean值
    fixInput = input - np.mean(input)
    # 一階微分
    fixInput -= sp.ndimage.interpolation.shift(input=fixInput, shift=1)
    fixInput = np.abs(fixInput)
    fixInput = np.convolve(fixInput, filter, 'SAME')
    maxtab, mintab = peakdet(fixInput, 0.9)

    ret = []
    print(np.shape(mintab)[0])
    # 為了讓第一個點跳過 所以先設定這樣
    x = -126
    for y in mintab[:,0]:
        # 若區間大於125個點 則略過該次區間
        if y-x > 125:
            x = y
            continue
        ecgSlice = np.zeros(shape = 125)
        ecgSlice[0:int(y-x)] += fixInput[int(x):int(y)]
        ret.append( ecgSlice )
        x = y
    
    return ret

# =============== 第二種 ECG 切斷與偵測 ================

# --------------- 偵測 -----------------
# Modify From https://github.com/KChen89/QRS-detection

# 標記出 rising 的地方
def lgth_transform(ecg, ws):
    lgth=ecg.shape[0]
    sqr_diff=np.zeros(lgth)
    diff=np.zeros(lgth)
    ecg=np.pad(ecg, ws, 'edge')
    for i in range(lgth):
        right=ecg[i+ws]-ecg[i+ws+ws]
        left=ecg[i+ws]-ecg[i]
        diff[i]=max(min(left, right), 0.0)
    return np.multiply(diff, diff)

# 區段積分 _ 就是把一段訊號加總
def integrate(ecg, ws):
    lgth=ecg.shape[0]
    integrate_ecg=np.zeros(lgth)
    ecg=np.pad(ecg, int(np.ceil(ws/2)), mode='symmetric')
    for i in range(lgth):
        integrate_ecg[i]=np.sum(ecg[i:i+ws])/ws
    return integrate_ecg

# convex detection _ may have better solution
def find_peak(data, fs, precision):
    ws = int(fs/precision)
    true_peaks=list()
    lgth=data.shape[0]
    predetectThr = 1 # 資料在進來前有做過預處理 所以不用這邊的閥值可以寫死
    # print("predetectThr : ", predetectThr) # debug
    index=int((ws-1)/2)
    lastP=-fs

    for i in range(lgth-ws+1):
        # 跳過太近的peak(四分之一秒內的) 
        if i - lastP < fs/4: continue
        temp=data[i:i+ws]
        # 預篩 變化量過小的也跳過
        if np.var(temp)<predetectThr: continue
        peak=True
        for j in range(index):
            if temp[index-j]<=temp[index-j-1] or temp[index+j]<=temp[index+j+1]:
                peak=False
                break

        if peak is True:
            lastP = i + index
            true_peaks.append(lastP)

    return np.asarray(true_peaks)

def find_R_peaks(ecg, peaks, ws):
    num_peak=peaks.shape[0]
    R_peaks=list()
    for j in range(num_peak):
        i=peaks[j]
        if i-2*ws>0 and i<ecg.shape[0]:
            temp_ecg=ecg[i-2*ws:i]
            R_peaks.append(int(np.argmax(temp_ecg)+i-2*ws))
    return np.asarray(R_peaks)

def find_S_point(ecg, R_peaks):
    num_peak=R_peaks.shape[0]
    S_point=list()
    for j in range(num_peak):
        i=R_peaks[j]
        cnt=i
        # 防止溢位
        if cnt+1>=ecg.shape[0]: break
        # 
        while ecg[cnt]>ecg[cnt+1]:
            cnt+=1
            if cnt>=ecg.shape[0]: break
        S_point.append(cnt)
    return np.asarray(S_point)

def find_Q_point(ecg, R_peaks):
    num_peak=R_peaks.shape[0]
    Q_point=list()
    for index in range(num_peak):
        i=R_peaks[index]
        cnt=i
        if cnt-1<0:
            break
        while ecg[cnt]>ecg[cnt-1]:
            cnt-=1
            if cnt<0:
                break
        Q_point.append(cnt)
    return np.asarray(Q_point)

def EKG_QRS_detect(ecg, fs):
    '''
        1. 先透過 lgth_transform 將訊號換算,變成 "時間-上升量" 的圖 -> ecg_lgth_transform
        2. 再連續做積分(區間加總) 算出整個 ＂pluse包＂-> ecg_integrate
        3. 用 find_peak 找出 "pluse包" 的最高點, 作為"pluse包"定位點 -> peaks
        4. 再用 find_R_peaks 從剛剛的定位點附近區間(fs/40 * 2),找出區間內的極大值,作為 R-peak
        5. (optional) 

        其他：
        當ecg的數值太小 不是正常的ecg 會抓不出R-peak
        抓不到的時候 plot 功能會爆掉
    '''
    sig_lgth=ecg.shape[0]
    ecg=ecg-np.mean(ecg)
    ecg_lgth_transform=lgth_transform(ecg, int(fs/20))

    ws=int(fs/8)
    ecg_integrate=integrate(ecg_lgth_transform, ws)/ws

    ws=int(fs/16)
    ecg_integrate=integrate(ecg_integrate, ws)

    ws=int(fs/36)
    ecg_integrate=integrate(ecg_integrate, ws)

    ws=int(fs/72)
    ecg_integrate=integrate(ecg_integrate, ws)

    peaks=find_peak(ecg_integrate, fs, 10)
    R_peaks=find_R_peaks(ecg, peaks, int(fs/40))
    # S_point=find_S_point(ecg, R_peaks)
    # Q_point=find_Q_point(ecg, R_peaks)
    return R_peaks

# --------------- 切斷 -----------------

def cutSignal(signal, trims):
    # 取兩個 R-peak 的中間值 作為切斷處
    # TODO 也許有更好的方式？
    # 回傳 list< ndarray<ecg片斷> >
    ret = []
    a = 0
    for i in range(np.shape(trims)[0]-1):
        b = int(np.ceil((trims[i] + trims[i+1])/2))
        ret.append(signal[a:b])
        a = b
    # 第一個不要
    del ret[0]
    return ret

from scipy.signal import resample_poly
def extendSignal(signal, fs, wantLen):
    ratio = np.ceil(wantLen / np.shape(signal)[0] * 100)
    fs_n = fs * ratio / 100
    signal_n = resample_poly(signal, ratio, 100)
    signal_n = signal_n[0:wantLen+1]
    return signal_n, fs_n

def sliceECGsignal_2(ecg, fs, sliceLen):
    # 把ecg訊號大小壓在 1~10 內,不然找 peak 演算法會出錯
    x = min(5000, np.shape(ecg)[0])
    tempEcg = ecg[0:x]
    tempEcg = tempEcg.astype(np.float64)
    sVar = np.sqrt(np.var(tempEcg))
    del tempEcg # 暫用數據 刪掉省記憶體 因為下面的演算很花時間 也花記憶體
    x = np.floor(np.log10(sVar))
    # print("{} : {}".format(sVar,x)) # debug
    ecg = ecg * np.power(10,-x+1)

    # 找 R_peak
    R_peak = EKG_QRS_detect(ecg, fs)
    # 如果找不到足夠的 R_peak 回傳 None
    if len(R_peak) < 2: return None, None
    # 根據 R_peak 切斷 ecg
    ecgPeaks = cutSignal(ecg, R_peak)
    # 調整長度
    fsList = []
    for i in range(len(ecgPeaks)):
        ecgPeaks[i], fs_n = extendSignal(ecgPeaks[i], fs, sliceLen)
        fsList.append(fs_n)
    return ecgPeaks, fsList

def test111(signal):
    # from scipy.signal import resample
    # from scipy.signal import resample_poly
    # f_fft = resample(temp, 100)
    # f_poly = resample_poly(temp, 100, 100)

    # 踩雷 - 如果用 float16 算var, 會因為數值爆掉變成 inf, 需要轉float64才夠
    signal = signal.astype(np.float64)
    print(signal.dtype)
    t1 = np.sum(np.abs(signal))
    t2 = np.var(signal)
    print("sum_abs : {}, var : {}".format(t1, t2))
    sVar = np.sqrt(t1)
    print(sVar)
    x = np.floor(np.log10(sVar))
    print(x)
    pass




