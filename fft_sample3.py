# 元ソース
# Pythonで高速フーリエ変換（FFT）の練習-4 フィルタリングでノイズを除去する
# https://momonoki2017.blogspot.com/2018/03/pythonfft-4.html
#%%
from matplotlib import lines
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
np.random.seed(0)

# %%
# 信号の作成（正弦波・ノイズ）
N = 2 ** 7
# N = 2 ** 8
# サンプリング周期(sec)
dt = 0.01
# 周波数(Hz)
freq = 4
# 振幅
amp = 1

t = np.arange(0, N * dt, dt)
f = amp * np.sin(2 * np.pi*freq*t) + np.random.random(N) * 0.3
# f = amp * np.sin(2 * np.pi*freq*t) + np.random.random(N) * 0.5

plt.xlabel('time [sec]', fontsize=14)
plt.ylabel('signal', fontsize=14)
plt.plot(t, f)
# %%
# FFT
F = np.fft.fft(f)
# FFTの複素数結果を絶対値に変換
F_abs = np.abs(F)
# 振幅を元の信号に揃える
## 交流成分はデータ数で割って2倍
F_abs_amp = (F_abs / N) * 2
## 直流成分は2倍不要
F_abs_amp[0] = F_abs_amp[0] / 2

# 周波数軸のデータ作成
fq = np.linspace(0, 1.0/dt, N)

plt.xlabel('frequency [Hz]', fontsize=14)
plt.ylabel('amplitude', fontsize=14)
# プロットすると正弦波の周波数ピークのほか、ノイズに起因する細かい起伏がある
plt.plot(fq, F_abs_amp)

# %%
# IFFTをして元の信号を再現する
F_ifft = np.fft.ifft(F)
# 結果の実数部のみ取り出す
F_ifft_real = F_ifft.real

plt.plot(t, F_ifft_real, c='g')
# %%
plt.plot(t, f, label='original', alpha=0.5)
plt.plot(t, F_ifft_real, c='g', label='ifft', linestyle='--', alpha=0.5)
plt.legend()
# %%

# %%
# ノイズ除去する
# %%
# FFT結果をコピー
F2 = np.copy(F)
# %%
# ノイズを取り除くにはノイズに相当するデータ値を0にすればOKだが、
# ノイズとみなすしきい値は予め決めておく必要がある
# 今回は周波数10を超えるデータはノイズとみなしてフィルタリング処理でデータを0にする
# **フィルタリングに使う周波数をカットオフ**という

# Numpyでは配列の中に条件を指定してTrueならゼロを入れるような動き
# 今回の場合は変数fqに周波数軸の情報を入れているため、配列の中の条件は(fq > カットオフ値)とする
# %%
# 周波数でフィルタリング処理
## カットオフ周波数
fc = 10
## カットオフを超える周波数のデータを0にする(ノイズ除去)
F2[(fq > fc)] = 0
# %%
# フィルタリング処理したFFT結果の確認
## FFTの複素数結果を絶対値に変換
F2_abs = np.abs(F2)
# 振幅を元の信号に揃える
## 交流成分はデータ数で割って2倍
F2_abs_amp = (F2_abs / N) * 2
## 直流成分は2倍にしなくていい
F2_abs_amp[0] = F2_abs_amp[0] / 2

plt.xlabel('frequency [Hz]', fontsize=14)
plt.ylabel('amplitude', fontsize=14)
# 周波数10を超えるものは0になっている。
plt.plot(fq, F2_abs_amp, c='r')
# %%
# フィルタリング処理をしたFFT結果を逆FFT処理する
# %%
F2_ifft = np.fft.ifft(F2)
# %%
# IFFTの結果から実数部を取り出したあとの値は2倍にする。
# フィルタリングでナイキスト周波数以降もすべて0にしたため、振幅を揃えるために必要

## 実数部を取り出して振幅を元のスケールに戻す
F2_ifft_real = F2_ifft.real * 2
# %%
plt.plot(t, f, label='original')
plt.plot(t, F2_ifft_real, c='r', linewidth=4, alpha=0.7, label='filtered')
plt.legend(loc='best')
plt.xlabel('time [sec]', fontsize=14)
plt.ylabel('signal', fontsize=14)
# %%

# %%
# 振幅でフィルタリングをする
plt.xlabel('frequency [Hz]', fontsize=14)
plt.ylabel('amplitude', fontsize=14)
plt.hlines(y=[0.2], xmin=0, xmax=100, colors='r', linestyles='dashed')
plt.plot(fq, F_abs_amp)
# %%
# 上のグラフで線を引いた0.2より小さい振幅のデータは0にする
# %%
# FFT結果コピー
F3 = np.copy(F)
# 振幅強度のしきい値
ac = 0.2
# 振幅がしきい値未満は0にする(ノイズ除去)
F3[(F_abs_amp < ac)] = 0
# %%
# しきい値でフィルタリング処理した結果の確認
# FFTの複素数結果を絶対値に変換
F3_abs = np.abs(F3)
# 振幅を元の信号に揃える
F3_abs_amp = (F3_abs / N) * 2
F3_abs_amp[0] = F3_abs_amp[0] / 2

plt.xlabel('frequency [Hz]', fontsize=14)
plt.ylabel('amplitude', fontsize=14)
plt.plot(fq, F3_abs_amp, c='orange')
# %%
# IFFT
## 先例と違ってIFFTしたあとの実数部の2倍は不要(ナイキスト周波数が消えてない)
## 振幅強度でフィルタリングした結果をIFFT
F3_ifft = np.fft.ifft(F3)
F3_ifft_real = F3_ifft.real

# 乱数がカットされて周波数4の正弦波だけの信号になった
plt.plot(t, f, label='original')
plt.plot(t, F3_ifft_real, c='orange', linewidth=4, alpha=0.7, label='filtered')
plt.legend(loc='best')
plt.xlabel('time [sec]', fontsize=14)
plt.ylabel('signal', fontsize=14)

# %%


# %% [markdown]
# まとめコード

#%%
# 信号の作成(正弦波+ノイズ)
N = 2 ** 7 # サンプル数
dt = 0.01 # サンプリング周期(sec)
freq = 4 # 周波数(Hz)
amp = 1 # 振幅

# 時間軸
f = np.arange(0, N*dt, dt)
f = amp * np.sin(2*np.pi * freq * t) + np.random.random(N) * 0.3

# FFT
F = np.fft.fft(f)
## 複素数を絶対値に変換
F_abs = np.abs(F)
## 振幅を元の信号に揃える(交流成分2倍, 直流成分は非2倍する)
F_abs_amp = (F_abs / N) * 2
F_abs_amp[0] = F_abs_amp[0] / 2


# 周波数軸
## linspace(開始, 終了, 分割数)
fq = np.linspace(0, 1.0/dt, N)

# フィルタリング1 (周波数でカット)
F2 = np.copy(F)
## カットオフ周波数
fc = 10
## カットオフ周波数を超える周波数のデータを0にする(ノイズ除去)
F2[(fq > fc)] = 0
## FFTの複素数結果を絶対値に変換
F2_abs = np.abs(F2)
## 振幅を元の信号に揃える(交流成分2倍, 直流成分は非2倍する)
F2_abs_amp = (F2_abs / N) * 2
F2_abs_amp[0] = F2_abs_amp[0] / 2

## IFFT処理
F2_ifft = np.fft.ifft(F2)
## 実数部の取得して振幅を元のスケールに戻す
F2_ifft_real = F2_ifft.real * 2

# フィルタリング2 (振幅強度でカット)
F3 = np.copy(F)
## 振幅強度のしきい値
ac = 0.2
## 振幅がしきい値未満は0にする(ノイズ除去)
F3[(F_abs_amp < ac)] = 0
## 複素数を絶対値に変換する
F3_abs = np.abs(F3)
## 振幅を元の信号に揃える(交流成分2倍, 直流成分は非2倍する)
F3_abs_amp = (F3_abs_amp / N) * 2
F3_abs_amp[0] = F3_abs_amp[0] / 2

## IFFT処理
F3_ifft = np.fft.ifft(F3)
## 実数部の取得
F3_ifft_real = F3_ifft.real

# グラフプロット
fig = plt.figure(figsize=(12, 12))

## オリジナル信号
fig.add_subplot(321)
plt.xlabel('time [sec]', fontsize=14)
plt.ylabel('signal', fontsize=14)
plt.plot(t, f)

## オリジナル信号 -> FFT
fig.add_subplot(322)
plt.xlabel('frequency [Hz]', fontsize=14)
plt.ylabel('amplitude', fontsize=14)
plt.plot(fq, F_abs_amp)

## オリジナル信号 -> FFT -> 周波数フィルタ - > IFFT
fig.add_subplot(323)
plt.xlabel('time [sec]', fontsize=14)
plt.ylabel('signal(freq filter)', fontsize=14)
plt.plot(t, F2_ifft_real)

## オリジナル信号 -> FFT -> 周波数フィルタ
fig.add_subplot(324)
plt.xlabel('frequency [Hz]', fontsize=14)
plt.ylabel('amplitude(freq filter)', fontsize=14)
# plt.vlines(x=[10], ymin=0, ymax=1, colors='r', linestyles='dashed')
plt.fill_between([10, 100], [0,0], [1,1], color='g', alpha=0.2)
plt.plot(fq, F2_abs_amp)

## オリジナル信号 -> FFT -> 振幅強度フィルタ -> IFFT
fig.add_subplot(325)
plt.xlabel('time [sec]', fontsize=14)
plt.ylabel('signal(amp filter)', fontsize=14)
plt.plot(t, F3_ifft_real)

## オリジナル信号 -> FFT -> 振幅強度フィルタ
fig.add_subplot(326)
plt.xlabel('frequency [Hz]', fontsize=14)
plt.ylabel('amplitude(amp filter)', fontsize=14)
# plt.hlines(y=[0.2], xmin=0, xmax=100, colors='r', linestyles='dashed')
plt.fill_between([0, 100], [0, 0], [0.2, 0.2], color='g', alpha=0.2)
plt.plot(fq, F3_abs_amp)

plt.tight_layout()
# %%
