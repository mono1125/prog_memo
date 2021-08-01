# 元ソース
# Pythonで高速フーリエ変換（FFT）の練習-2 信号を時間軸と周波数軸で表現する
# https://momonoki2017.blogspot.com/2018/03/pythonfft-2.html
# %%
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# %%
# 時間を考慮した信号を作ってFFTを実践する。
# FFTは信号の時間軸を周波数軸に変換する意味合いがある

# %%
# データ数を128個、サンプリング周期は10msとする
N = 2 ** 7
# サンプリング周期100ms -> サンプリング周波数100Hz
dt = 0.01
# 周波数(10Hz) -> 正弦波の周期0.1sec
freq = 10
# 振幅1
amp = 1
# 時間軸
t = np.arange(0, N*dt, dt)
# 信号(周波数10, 振幅1の正弦波)
f = amp * np.sin(2 * np.pi * freq * t)

plt.xlabel('time(sec)', fontsize=14)
plt.ylabel('signal amplitude', fontsize=14)
plt.plot(t, f)
# %%
# FFT
F = np.fft.fft(f)
# FFTの結果を絶対値に変換
F_abs = np.abs(F)
# 振幅を元の信号に揃える
## 交流成分はデータ数で割って2倍する
F_abs_amp = (F_abs / N) * 2
## 直流成分は2倍不要
F_abs_amp[0] = F_abs_amp[0]

plt.plot(F_abs_amp)
# %%
# 周波数の検出点がズレてる
## x軸は周波数(Hz)に変換する必要がある
## 周波数(Hz)の取る範囲は1/サンプリング周期(sec)で求まる
## この場合1/0.01だから100Hzまでになる。
## 解析できる周波数はサンプリング周期に依存する
## サンプリング定理より解析できる周波数はサンプリング周波数の半分までとなる
# %%
# 周波数軸のデータ作成
## 周波数軸 linspace(開始, 終了, 分割数)
fq = np.linspace(0, 1.0/dt, N)
plt.xlabel('frequency [Hz]', fontsize=14)
plt.ylabel('signal amplitude', fontsize=14)
# 周波数軸に変えると10Hzのところにピークが出ている。
plt.plot(fq, F_abs_amp)
# %%
# 鏡像ピークを除くために表示範囲はナイキスト定数までにする
plt.xlabel('frequency [Hz]', fontsize=14)
plt.ylabel('signal amplitude', fontsize=14)
plt.plot(fq[:int(N/2)+1], F_abs_amp[:int(N/2)+1])
# %%
