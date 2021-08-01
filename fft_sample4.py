# 元ソース
# Pythonで高速フーリエ変換（FFT）の練習-5 周波数ピークを自動で検出
# https://momonoki2017.blogspot.com/2018/03/pythonfft-5.html
# %%
from sys import last_traceback
from numpy.lib.function_base import percentile
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
np.random.seed(0)
# %% [markdown]
# FFT分析結果からピークの周波数を自動で検出する。
# %%
# サンプル数
N = 2 ** 9
# サンプリング周期
dt = 0.01
# 周波数
fq1, fq2 = 3, 8
# 振幅
amp1, amp2 = 1.5, 1

# %%
# 時間軸
t = np.arange(0, N * dt, dt)
# 時間信号作成
f = amp1 * np.sin(2 * np.pi * fq1 * t)
f += amp2 * np.sin(2 * np.pi * fq2 * t)
f += np.random.randn(N) * 0.5

# 周波数軸
freq = np.linspace(0, 1.0/dt, N)
# %%
# FFT
F = np.fft.fft(f)

# 複素数を絶対値に変換
F_abs = np.abs(F)
# 振幅を元の信号のスケールに合わせる
# 交流成分はサンプル数で割って2倍する
F_abs = (F_abs / N) * 2
# 直流成分は2倍しなくていい
F_abs[0] = F_abs[0] / 2
# %%
# 時間軸のグラフ作成
plt.figure(figsize=(8, 6))
plt.subplot(211)
plt.plot(t, f)
plt.xlabel('Time [sec]')
plt.ylabel('Signal')

# FFT結果からピークを自動検出
# ピーク(極大値)のインデックスを取得
maximal_idx = signal.argrelmax(F_abs, order=1)[0]
# ピーク検出しきい値、ナイキスト定数を超えるものとしきい値より小さい振幅ピークを除外する
peak_cut = 0.3
maximal_idx = maximal_idx[(F_abs[maximal_idx] > peak_cut)
                          & (maximal_idx <= N/2)]

# 周波数軸のグラフ作成
plt.subplot(212)
plt.xlabel("Frequency [Hz]")
plt.ylabel('Amaplitude')
plt.axis([0, 1.0/dt/2, 0, max(F_abs) * 1.5])
plt.plot(freq, F_abs)
plt.plot(freq[maximal_idx], F_abs[maximal_idx], 'ro')

# グラフにピークの周波数をテキストで表示
for i in range(len(maximal_idx)):
    plt.annotate('{0:.0f}(Hz)'.format(np.round(freq[maximal_idx[i]])),
                 xy=(freq[maximal_idx[i]], F_abs[maximal_idx[i]]),
                 xytext=(10, 20),
                 textcoords='offset points',
                 arrowprops=dict(arrowstyle="->",
                                 connectionstyle="arc3,rad=.2")
                 )
plt.subplots_adjust(hspace=0.4)
# 元信号の周波数がきちんと検出出来ている。
plt.show()
# ピーク検出に用いた生値
print('peak', freq[maximal_idx])
# %%

# %% [markdown]
# 信号を少し複雑にする
# %%
# サンプル数
N = 2 ** 9
# サンプリング周期
dt = 0.01
# 周波数
fq1, fq2, fq3 = 12, 24, 33
# 振幅
amp1, amp2, amp3 = 1.5, 1, 2
# %%
# 時間軸
t = np.arange(0, N * dt, dt)
# 時間信号作成
f = amp1 * np.sin(2*np.pi*fq1*t)
f += amp2 * np.sin(2*np.pi * fq2 * t)
f += amp3 * np.sin(2*np.pi*fq3*t)
f += np.random.randn(N) * 0.3

# 周波数軸
freq = np.linspace(0, 1.0/dt, N)
# %%
# FFT
F = np.fft.fft(f)
# 複素数を絶対値に変換する
F_abs = np.abs(F)
# 振幅を元の信号に揃える
## 交流成分は2倍して直流成分は2倍しない
F_abs = (F_abs / N) * 2
F_abs[0] = F_abs[0] / 2
# %%
# グラフ
plt.figure(figsize=(8, 6))

plt.subplot(211)
plt.plot(t, f)
plt.xlabel('Time [sec]')
plt.ylabel('Signal')

# FFT結果からピークを自動検出
## ピーク(極大値)のインデックス取得
### scipyを使うと極大値・極小値を自動で求めてくれる
maximal_idx = signal.argrelmax(F_abs, order=1)[0]
## ピーク検出しきい値設定とナイキスト定数を超えるものとしきい値より小さい振幅ピークを除外
peak_cut = 0.3
maximal_idx = maximal_idx[(F_abs[maximal_idx] > peak_cut) & (maximal_idx <= N/2)]

# 周波数軸グラフ
plt.subplot(212)
plt.xlabel('Frequency [Hz]')
plt.ylabel("Amplitude")

plt.axis([0, 1.0/dt/2, 0, max(F_abs)*1.5])
plt.plot(freq, F_abs)
plt.plot(freq[maximal_idx], F_abs[maximal_idx], 'ro')

# グラフにピークの周波数をテキストで表示
for i in range(len(maximal_idx)):
    ## 引き出し線とテキスト挿入
    plt.annotate('{0:.0f}(Hz)'.format(np.round(freq[maximal_idx[i]])),
                                xy=(freq[maximal_idx[i]], F_abs[maximal_idx[i]]),
                                xytext=(10, 20),
                                textcoords='offset points',
                                arrowprops=dict(arrowstyle='->', connectionstyle='arc3, rad=.2')
                                )
plt.subplots_adjust(hspace=0.4)
plt.show()
print('peak', freq[maximal_idx])
# %%

# %%
# 極大値と極小値の取得
# %%
## データ数
N = 2 ** 7
x = np.linspace(0, 2*np.pi, N)
y = np.sin(x)
y_add_noise = y + np.random.randn(N) * 0.3

plt.plot(x, y, 'r', label='sin(x)', c='gray', alpha=0.5)
plt.plot(x, y_add_noise, label='sin(x) + noise', c='g')

plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
# %%
# 極大値はsignal.argrelmax, 極小値はsignal.argrelmin で求めることができる
# %%
N = 2 ** 7
x = np.linspace(0, 2*np.pi, N)
y = np.sin(x)
y_add_noise = y + np.random.randn(N) * 0.3

# ピーク検出
## 極大値インデックス取得
maximal_idx = signal.argrelmax(y_add_noise, order=1)

plt.plot(x, y, 'r', label='sin(x)', c='gray', alpha=0.5)
plt.plot(x, y_add_noise, label='sin(x) + noise', c='g')
# 極大点プロット
plt.plot(x[maximal_idx], y_add_noise[maximal_idx], 'ro', label='peak_maximal')

plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
# %%
# order値を変えて試す
N = 2 ** 7
x = np.linspace(0, 2*np.pi, N)
y = np.sin(x)
y_add_noise = y + np.random.randn(N)

order_list = [1, 2, 4, 8]
# グラフの個数
axes_count = 4

fig = plt.figure(figsize=(12, 8))

for i in range(axes_count):
    # ピーク検出
    maximal_idx = signal.argrelmax(y_add_noise, order=order_list[i])

    fig.add_subplot(2, 2, i+1)
    plt.title('order={}'.format(order_list[i]), fontsize=18)
    plt.plot(x, y, 'r', label='sin(x)+noise', c='gray')
    plt.plot(x, y_add_noise, label='sin(x) + noise', c='gray')

    plt.plot(x[maximal_idx], y_add_noise[maximal_idx], 'ro', label='peak_maximal')

# order値を増やすと検出ピークの数が減っている。
## order値は増減チェックを行うデータ数の幅(x軸)に相当する。
## 1だと前後各1点, 2だと前後各2点を対象にするようなイメージ
plt.tight_layout()
# %%
# 極小値のピークを検出する
N = 2 ** 6
x = np.linspace(0, 2*np.pi, N)
y = np.sin(x)
y_add_noise = y + np.random.randn(N) * 0.3

# ピーク検出
minimal_idx = signal.argrelmin(y_add_noise, order=1)

plt.plot(x, y, 'r', label='sin(x)', c='gray', alpha=0.5)
plt.plot(x, y_add_noise, label='sin(x) + noise', c='g')

plt.plot(x[minimal_idx], y_add_noise[minimal_idx], 'bo', label='peek_minimal')

plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
# %%
# 極大値と極小値を同時にプロットする
N = 2 ** 6
x = np.linspace(0, 2*np.pi, N)
y = np.sin(x)
y_add_noise = y + np.random.randn(N) * 0.3

# ピーク検出
maximal_idx = signal.argrelmax(y_add_noise, order=1)
minimal_idx = signal.argrelmin(y_add_noise, order=1)

plt.plot(x, y, 'r', label='sin(x)', c='gray', alpha=0.5)
plt.plot(x, y_add_noise, label='sin(x)+noise', c='g')

plt.plot(x[maximal_idx], y_add_noise[maximal_idx], 'ro', label='peak_maximal')
plt.plot(x[minimal_idx], y_add_noise[minimal_idx], 'bo', label='peek_minimal')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
# %%
# 極大値情報の表示例
## ピークの検出数
print('idx_length', len(maximal_idx))
## ピークのindex
print('idx_value', maximal_idx)
## ピークのx値
print('x_value', x[maximal_idx])
## ピークのy値
print('y_value', y_add_noise[maximal_idx])
# %%
