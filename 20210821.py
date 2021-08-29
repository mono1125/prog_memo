# 複数時系列データを1つの深層学習モデルで学習させる
# - 複数入力型の深層学習モデル
# - 個別入力型の深層学習モデルの組み合わせ
# 前者の利点はモデルがシンプルなため後者より高速
# 後者の利点は時系列ごとにカスタマイズ可能なため、前者よりも精度を高められる
import pandas as pd
from pandas import read_csv
import numpy as np
# import matplotlib.pyplot as plt

# 前処理
## データを取得して対数スケーリング。
## -> データ間の値の差が大きいため
## 1. データを読み込み
## 2. 対数スケーリング
## 2.1. 0の部分は計算できないため+1の処理が入る
## 2.2. それでも発生する計算できない値はNaNになるため0で置換
wave_data = read_csv('https://raw.githubusercontent.com/jamesrobertlloyd/gpss-research/master/data/tsdlr_5050/daily-minimum-temperatures-in-me-train.csv',
                    header=None,
                    names=["Date", "Temp"])
wave_data = wave_data.sort_values(by=['Date'])
production_of_gas_data = read_csv('https://raw.githubusercontent.com/jamesrobertlloyd/gpss-research/master/data/tsdlr_5050/monthly-production-of-gas-in-aus-train.csv',
                    header=None,
                    names=["Date", "production-of-gas"])
production_of_gas_data = production_of_gas_data.sort_values(by=['Date'])

X_orig = np.nan_to_num(np.log(wave_data["Temp"].values + 1))
X_day = wave_data["Date"].values

X_orig_second = np.nan_to_num(np.log(production_of_gas_data["production-of-gas"].values + 1))
X_day_second = production_of_gas_data["Date"].values

