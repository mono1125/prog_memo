# 元ソース
# Pythonで高速フーリエ変換（FFT）の練習-1 簡単な信号でFFTを体験してみよう
# https://momonoki2017.blogspot.com/2018/03/pythonfft-1-fft.html
# %%
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from scipy import signal
# %%

# %%
x = np.arange(0, 2*np.pi, 0.1)
y = np.sin(x)
plt.plot(x, y)
# %%
# グラフのx軸をpiで表示
x = np.arange(0, 2*np.pi, 0.1)
y = np.sin(x)
plt.xticks(np.arange(0, 2.1*np.pi, 0.5*np.pi), ['0', '1/2$\pi$', u'$\pi$', '3/2$\pi$', u'2$\pi$'], fontsize=14)
plt.plot(x, y)
# %%

# %% [markdown]
## FFT
# - FFTで使うデータ数は$2^n$個にしなければならない
# %%
N = 32
n = np.arange(32)
signal = np.sin(2*np.pi*n / N)

plt.figure(figsize=(8,4))
plt.xlabel("n")
plt.ylabel("Signal")
plt.plot(signal)
# %%
N = 2 ** 6
n = np.arange(N)
freq = 10
f = np.sin(freq * 2 * np.pi * (n / N))

plt.figure(figsize=(8,4))
plt.xlabel("n")
plt.ylabel("Signal")
plt.plot(f)
# %%
F = np.fft.fft(f)
# %%
print(F)
# %%
# 複素数を絶対値変換
## 信号に入れた周期(x軸の値)のところでピークが現れる
## 周期を確認するときは後半側に出現するピークは無視する（鏡像）
## 周期はデータ数の半分までしか確認できない（サンプリング定理）
F_abs = np.abs(F)
plt.plot(F_abs)
# %%
# データ数の半分までをプロット
plt.plot(F_abs[:int(N/2)+1])
# %%
# y軸の値は意味のない値になっているため元の信号の振幅に合うように値を変換する
# 元の振幅に揃えるにはピーク強度の値をデータ数で割ればOK
# ただし、対となる鏡像の値を足し合わせる必要があるためデータ数で割ったあとに2倍する
F_abs_amp = (F_abs / N) *2
# 直流成分は2倍不要
F_abs_amp[0] = F_abs_amp[0] / 2
plt.plot(F_abs_amp[:int(N/2)+1])

# %%
# 信号の振幅を変更してFFTで正しい値が得られるかを確認する
N = 2 ** 5 # データ数
n = np.arange(N)
freq = 3 # 周期
amp = 4 # 振幅
f = amp * np.sin(freq *2 * np.pi * (n/N))

plt.figure(figsize=(8,4))
plt.xlabel('n')
plt.ylabel('Signal')
plt.plot(f)
# %%
# FFT
F = np.fft.fft(f)
# FFT結果を絶対値に変換
F_abs = np.abs(F)
# 振幅を元の信号に揃える
## 交流成分はデータ数で割って2倍する
F_abs_amp = (F_abs / N) * 2
## 直流成分は2倍不要
F_abs_amp[0] = F_abs_amp[0] / 2
## 振幅4, 周期3とわかる
plt.plot(F_abs_amp[:int(N/2)+1])
# %%

# %%
# 信号を複雑にする
# 信号は周期2と6のサインカーブを足し合わせた波形にする
N = 2 ** 5
n = np.arange(N)
## 周期
f1 = 2
f2 = 6
f = np.sin(f1 * 2 * np.pi*(n/N)) + np.sin(f2 * 2 * np.pi * (n/N))

plt.figure(figsize=(8,4))
plt.xlabel('n')
plt.ylabel('Signal')
plt.plot(f)
# %%
# データ数を増やしてなめらかにする
N = 2 ** 6
n = np.arange(N)
## 周期
f1 = 2
f2 = 6
f = np.sin(f1 * 2 * np.pi*(n/N)) + np.sin(f2 * 2 * np.pi * (n/N))

plt.figure(figsize=(8,4))
plt.xlabel('n')
plt.ylabel('Signal')
plt.plot(f)
# %%
# FFT
F = np.fft.fft(f)
# FFT結果を絶対値に変換
F_abs = np.abs(F)
# 振幅を元の信号に揃える
## 交流成分はデータ数で割って2倍する
F_abs_amp = (F_abs / N) * 2
## 直流成分は2倍不要
F_abs_amp[0] = F_abs_amp[0] / 2
## 周期2と6のところにピークが現れる
plt.plot(F_abs_amp[:int(N/2)+1])


# %%
# 周期と振幅を変えて試す
N = 2 ** 7
n = np.arange(N)
# 周期
f1 = 4
f2 = 10
# 振幅
a1 = 1.5
a2 = 3
f = a1 * np.sin(f1 * 2 * np.pi * (n/N)) + a2 * np.sin(f2 * 2 * np.pi * (n/N))

plt.figure(figsize=(8,4))
plt.xlabel('n')
plt.ylabel('Signal')
plt.plot(f)
# %%
# FFT
F = np.fft.fft(f)
# FFT結果を絶対値に変換
F_abs = np.abs(F)
# 振幅を元の信号に揃える
## 交流成分はデータ数で割って2倍する
F_abs_amp = (F_abs / N) * 2
F_abs_amp[0] = F_abs_amp[0] / 2

plt.plot(F_abs_amp[:int(N/2)+1])
# %%
