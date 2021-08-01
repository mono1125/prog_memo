# 元ソース
# Pythonで高速フーリエ変換（FFT）の練習-3 逆高速フーリエ変換（IFFT）の実践
# https://momonoki2017.blogspot.com/2018/03/pythonfft-3-ifft.html
# %%
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
#%%
# 逆FFTを行う
# FFTは信号の情報を時間軸から周波数軸に変換する操作
# IFFTは周波数軸を時間軸に変換する
# %%
N = 2 ** 7
# サンプリング周期10ms -> サンプリング周波数100Hz
dt = 0.01
# 周波数 10Hz -> 正弦波の周期0.1sec
freq1 = 10
amp1 = 1
# 周波数15Hz -> 正弦波の周期0.06666sec
freq2 = 15
amp2 = 1

t = np.arange(0, N * dt, dt)
f = amp1 * np.sin(2*np.pi*freq1*t) + amp2 * np.sin(2*np.pi*freq2*t)

plt.xlabel('time [sec]', fontsize=14)
plt.ylabel('signal', fontsize=14)
plt.plot(t, f)
# %%

# %%
# FFT
F = np.fft.fft(f)
# FFTの複素数結果を絶対値に変換
F_abs = np.abs(F)
# 振幅を元の信号に揃える
## 交流成分はデータ数で割って2倍する
F_abs_amp = (F_abs / N) * 2
## 直流成分は2倍不要
F_abs_amp[0] = F_abs_amp[0] / 2

# 周波数軸作成
## linspace(開始, 終了, 分割数)
fq = np.linspace(0, 1.0/dt, N)

plt.xlabel('frequency [Hz]', fontsize=14)
plt.ylabel('amplitude', fontsize=14)
plt.plot(fq, F_abs_amp)
# %%
# FFTの結果を10個表示
print(F[:10])
# %%


# %%
# FFTの結果をIFFTする場合はnp.fft.ifft(data)を使う
F_ifft = np.fft.ifft(F)
# %%
## IFFTの結果は複素数になっている
print(F_ifft[:10])
# %%
# IFFTの結果を実数部だけ取り出す
F_ifft_real = F_ifft.real
print(F_ifft_real[:10])
# %%
# IFFTの結果をプロット
plt.plot(t, F_ifft_real, c='g')
# %%
# 元の波形と重ね書きする
## 元の信号と一致している -> 逆FFT成功
plt.plot(t, f, label='original')
plt.plot(t, F_ifft_real, c='g', linestyle='--', label='IFFT')
plt.legend(loc='best')
plt.xlabel('time [sec]', fontsize=14)
plt.ylabel('signal', fontsize=14)
# %%
