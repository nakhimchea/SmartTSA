import datetime
import time
from typing import Optional
from freqtrade.persistence import Trade

from freqtrade.strategy import IStrategy, IntParameter
from pandas import DataFrame
import talib.abstract as ta
import numpy as np
import scipy
import os
import json
import requests


def quote_symbols_list(quote='USDT') -> dict.keys:
    base_url = 'https://fapi.binance.com'
    endpoint = '/fapi/v1/exchangeInfo'
    response = requests.get(base_url + endpoint).json()
    symbols = response['symbols']
    pairs = {s['symbol']: s for s in symbols if quote in s['symbol']}

    return pairs.keys()


def lunarcrush_entry_pairs() -> tuple:
    data_path = os.path.join('user_data', 'strategies', 'lunarcrush')
    files = os.listdir(data_path)
    usdt_pairs = quote_symbols_list('USDT')
    # print("USDT pair length: ", usdt_pairs.__len__())
    entry_pairs = []
    advance_pairs = []
    stablecoins = json.load(open(os.path.join('user_data', 'strategies', 'stablecoins.json')))["symbols"]
    for file in files:
        data = json.load(open(os.path.join(data_path, file)))
        acr = data["acr"]
        gs = data["gs"]
        datatime = datetime.datetime.fromtimestamp(float(data["dt"][-1]))
        time_diff = (datetime.datetime.now() - datatime).total_seconds()
        if time_diff > 1500:
            print("Sentiment Data is not live.")

        length = len(acr)
        last_acr = acr[length-14: length-1]
        last_gs = gs[length-14: length-1]
        count_acr = sum(element <= 45 for element in last_acr)
        count_gs = sum(element >= 48 for element in last_gs)
        if max(acr) < 1500 and count_acr >= 7 and count_gs >= 7:
            symbol = file.split('.')[0]
            if symbol not in stablecoins:
                pair = symbol + "USDT"
                if pair in usdt_pairs:
                    if acr[length-1] <= 3:
                        advance_pairs.append(pair)
                    elif count_acr >= 7 and count_gs >= 7 and time_diff < 1500:
                        entry_pairs.append(pair)

    return entry_pairs, advance_pairs


class SmartATSA(IStrategy):
    INTERFACE_VERSION: int = 3

    # Buy hyperspace params:
    buy_params = {
        "buy_m1": 4,
        "buy_m2": 7,
        "buy_m3": 1,
        "buy_p1": 8,
        "buy_p2": 9,
        "buy_p3": 8,
    }

    # Sell hyperspace params:
    sell_params = {
        "sell_m1": 1,
        "sell_m2": 3,
        "sell_m3": 6,
        "sell_p1": 16,
        "sell_p2": 18,
        "sell_p3": 18,
    }

    # ROI table:
    minimal_roi = {
        "0": 0.08 * 5,
        "1704": 0.06 * 5,
        "3712": 0.033 * 5,
        "5605": 0
    }

    # Stoploss:
    stoploss = -1

    # enable short
    can_short: bool = True

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.0001
    trailing_stop_positive_offset = 0.0098 * 5
    trailing_only_offset_is_reached = True

    timeframe = "4h"
    startup_candle_count = 18

    lunarcrush = lunarcrush_entry_pairs()

    buy_m1 = IntParameter(1, 7, default=1)
    buy_m2 = IntParameter(1, 7, default=3)
    buy_m3 = IntParameter(1, 7, default=4)
    buy_p1 = IntParameter(7, 21, default=14)
    buy_p2 = IntParameter(7, 21, default=10)
    buy_p3 = IntParameter(7, 21, default=10)

    sell_m1 = IntParameter(1, 7, default=1)
    sell_m2 = IntParameter(1, 7, default=3)
    sell_m3 = IntParameter(1, 7, default=4)
    sell_p1 = IntParameter(7, 21, default=14)
    sell_p2 = IntParameter(7, 21, default=10)
    sell_p3 = IntParameter(7, 21, default=10)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        current_pair = metadata["pair"]
        # print(current_pair)

        # One-time refresher
        if current_pair == "BTC/USDT:USDT":
            print("Waiting 12 minutes before tradings...")
            time.sleep(720)
            print("Refreshing data...")
            self.lunarcrush = lunarcrush_entry_pairs()

        dataframe['signals'] = self.smartentry(dataframe, current_pair, self.lunarcrush[0], self.lunarcrush[1])['signals']
        dataframe['peak_bottom'] = self.smartexit(dataframe)['peak_bottom']

        for multiplier in self.buy_m1.range:
            for period in self.buy_p1.range:
                dataframe[f"supertrend_1_buy_{multiplier}_{period}"] = self.supertrend(dataframe, multiplier, period)["STX"]

        for multiplier in self.buy_m2.range:
            for period in self.buy_p2.range:
                dataframe[f"supertrend_2_buy_{multiplier}_{period}"] = self.supertrend(dataframe, multiplier, period)["STX"]

        for multiplier in self.buy_m3.range:
            for period in self.buy_p3.range:
                dataframe[f"supertrend_3_buy_{multiplier}_{period}"] = self.supertrend(dataframe, multiplier, period)["STX"]

        for multiplier in self.sell_m1.range:
            for period in self.sell_p1.range:
                dataframe[f"supertrend_1_sell_{multiplier}_{period}"] = self.supertrend(dataframe, multiplier, period)["STX"]

        for multiplier in self.sell_m2.range:
            for period in self.sell_p2.range:
                dataframe[f"supertrend_2_sell_{multiplier}_{period}"] = self.supertrend(dataframe, multiplier, period)["STX"]

        for multiplier in self.sell_m3.range:
            for period in self.sell_p3.range:
                dataframe[f"supertrend_3_sell_{multiplier}_{period}"] = self.supertrend(dataframe, multiplier, period)["STX"]

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        length = len(dataframe)
        dataframe.loc[
            (dataframe[f"supertrend_1_buy_{self.buy_m1.value}_{self.buy_p1.value}"] == "up")
            & (dataframe[f"supertrend_2_buy_{self.buy_m2.value}_{self.buy_p2.value}"] == "up")
            & (dataframe[f"supertrend_3_buy_{self.buy_m3.value}_{self.buy_p3.value}"] == "up")
            & (dataframe['signals'].iat[length - 1] == 'Long')
            & (dataframe['peak_bottom'] != 'P0')
            & (dataframe['peak_bottom'] != 'P1')
            & (dataframe['peak_bottom'] != 'P2')
            & (dataframe['peak_bottom'] != 'P3')
            & (dataframe['peak_bottom'] != 'P4')
            & (dataframe['peak_bottom'] != 'P5')
            & (dataframe["volume"] > 0),
            "enter_long"] = 1

        dataframe.loc[
            (dataframe[f"supertrend_1_buy_{self.buy_m1.value}_{self.buy_p1.value}"] == "down")
            & (dataframe[f"supertrend_2_buy_{self.buy_m2.value}_{self.buy_p2.value}"] == "down")
            & (dataframe[f"supertrend_3_buy_{self.buy_m3.value}_{self.buy_p3.value}"] == "down")
            & (dataframe['signals'].iat[length - 1] == 'Short')
            & (dataframe['peak_bottom'] != 'B0')
            & (dataframe['peak_bottom'] != 'B1')
            & (dataframe['peak_bottom'] != 'B2')
            & (dataframe['peak_bottom'] != 'B3')
            & (dataframe['peak_bottom'] != 'B4')
            & (dataframe['peak_bottom'] != 'B5')
            & (dataframe["volume"] > 0)
            & (3 < datetime.datetime.now().date().day < 23),
            "enter_short"] = 1

        return dataframe

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                            side: str, **kwargs) -> bool:
        if current_time.hour % 4 < 1 and 12 <= current_time.minute <= 20:
            return True
        else:
            return False

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe[f"supertrend_1_sell_{self.sell_m1.value}_{self.sell_p1.value}"] == "down")
            & (dataframe[f"supertrend_2_sell_{self.sell_m2.value}_{self.sell_p2.value}"] == "down")
            & (dataframe[f"supertrend_3_sell_{self.sell_m3.value}_{self.sell_p3.value}"] == "down"),
            "exit_long"] = 1

        dataframe.loc[
            (dataframe[f"supertrend_1_sell_{self.sell_m1.value}_{self.sell_p1.value}"] == "up")
            & (dataframe[f"supertrend_2_sell_{self.sell_m2.value}_{self.sell_p2.value}"] == "up")
            & (dataframe[f"supertrend_3_sell_{self.sell_m3.value}_{self.sell_p3.value}"] == "up"),
            "exit_short"] = 1

        return dataframe

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs):
        # Kill the short position in sentiment top
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        length = len(dataframe["signals"])
        if trade.is_short and dataframe['signals'].iat[length - 1] == 'Long':
            return 'Sentiment Kill'

        # Sell any positions at a loss if they are held for more than 1 day.
        if current_profit < -0.086 * 5 and (current_time - trade.open_date_utc).days >= 1:
            return 'Kill Loss'

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str], side: str,
                 **kwargs) -> float:
        if pair == 'BTC/USDT:USDT' or pair == 'ETH/USDT:USDT':
            return 8.0

        return 5.0

    @staticmethod
    def supertrend(dataframe: DataFrame, multiplier, period):
        df = dataframe.copy()

        df["TR"] = ta.TRANGE(df)
        df["ATR"] = ta.SMA(df["TR"], period)

        st = "ST_" + str(period) + "_" + str(multiplier)
        stx = "STX_" + str(period) + "_" + str(multiplier)

        # Compute basic upper and lower bands
        df["basic_ub"] = (df["high"] + df["low"]) / 2 + multiplier * df["ATR"]
        df["basic_lb"] = (df["high"] + df["low"]) / 2 - multiplier * df["ATR"]

        # Compute final upper and lower bands
        df["final_ub"] = 0.00
        df["final_lb"] = 0.00
        for i in range(period, len(df)):
            df["final_ub"].iat[i] = (
                df["basic_ub"].iat[i]
                if df["basic_ub"].iat[i] < df["final_ub"].iat[i - 1]
                or df["close"].iat[i - 1] > df["final_ub"].iat[i - 1]
                else df["final_ub"].iat[i - 1]
            )
            df["final_lb"].iat[i] = (
                df["basic_lb"].iat[i]
                if df["basic_lb"].iat[i] > df["final_lb"].iat[i - 1]
                or df["close"].iat[i - 1] < df["final_lb"].iat[i - 1]
                else df["final_lb"].iat[i - 1]
            )

        # Set the Supertrend value
        df[st] = 0.00
        for i in range(period, len(df)):
            df[st].iat[i] = (
                df["final_ub"].iat[i]
                if df[st].iat[i - 1] == df["final_ub"].iat[i - 1]
                and df["close"].iat[i] <= df["final_ub"].iat[i]
                else df["final_lb"].iat[i]
                if df[st].iat[i - 1] == df["final_ub"].iat[i - 1]
                and df["close"].iat[i] > df["final_ub"].iat[i]
                else df["final_lb"].iat[i]
                if df[st].iat[i - 1] == df["final_lb"].iat[i - 1]
                and df["close"].iat[i] >= df["final_lb"].iat[i]
                else df["final_ub"].iat[i]
                if df[st].iat[i - 1] == df["final_lb"].iat[i - 1]
                and df["close"].iat[i] < df["final_lb"].iat[i]
                else 0.00
            )
        # Mark the trend direction up/down
        df[stx] = np.where((df[st] > 0.00), np.where((df["close"] < df[st]), "down", "up"), np.NaN)

        # Remove basic and final bands from the columns
        df.drop(["basic_ub", "basic_lb", "final_ub", "final_lb"], inplace=True, axis=1)

        df.fillna(0, inplace=True)

        return DataFrame(index=df.index, data={"ST": df[st], "STX": df[stx]})

    @staticmethod
    def smartexit(df: DataFrame):
        dataframe = df.copy()

        # Reverse Candle Detection (Stochastic)
        stochd = ta.STOCHF(dataframe, fastk_period=3)
        reversed_stochd = 100 - stochd

        # Threshold Height will be the cycle of the market
        peaks = scipy.signal.find_peaks(stochd.fastk * 1.3 - 15, height=95)
        bottoms = scipy.signal.find_peaks(reversed_stochd.fastk * 1.3 - 15, height=79)

        peak_indices = []
        for index in range(0, len(peaks[0]) - 1):
            if peaks[0][index + 1] > peaks[0][index] + 4:
                peak_indices.append(peaks[0][index])
        peak_indices.append(peaks[0][len(peaks[0]) - 1])

        bottom_indices = []
        for index in range(0, len(bottoms[0]) - 1):
            if bottoms[0][index + 1] > bottoms[0][index] + 4:
                bottom_indices.append(bottoms[0][index])
        bottom_indices.append(bottoms[0][len(bottoms[0]) - 1])

        # print('=====================.=======================')
        # plot.plot(dataframe.date, stochd.fastk * 1.3 - 15, '-D', markevery=[*peak_indices, *bottom_indices])
        # plot.legend(['K', 'D', 'J'])
        # plot.show()
        # time.sleep(5)
        # print('=====================.=======================')

        rp = 0
        rb = 0
        peak_bottom = DataFrame(data='NA', index=dataframe.index, columns=['peak_bottom'], dtype=str)
        strength = DataFrame(data='NA', index=dataframe.index, columns=['strength'], dtype=str)

        while len(peak_indices) > 0 and len(bottom_indices) > 0:
            if peak_indices[0] < bottom_indices[0]:
                for index in range(peak_indices[0], bottom_indices[0]):
                    peak_bottom['peak_bottom'].iat[index] = 'P'
                    strength['strength'].iat[index] = str(rp)
                peak_indices.pop(0)
                rp += 1
                rb = 0
            elif bottom_indices[0] < peak_indices[0]:
                for index in range(bottom_indices[0], peak_indices[0]):
                    peak_bottom['peak_bottom'].iat[index] = 'B'
                    strength['strength'].iat[index] = str(rb)
                bottom_indices.pop(0)
                rb += 1
                rp = 0

        if len(peak_indices) > 0:
            for index in range(peak_indices[0], len(dataframe)):
                peak_bottom['peak_bottom'].iat[index] = 'P'
                strength['strength'].iat[index] = str(rp)
        elif len(bottom_indices) > 0:
            for index in range(bottom_indices[0], len(dataframe)):
                peak_bottom['peak_bottom'].iat[index] = 'B'
                strength['strength'].iat[index] = str(rb)

        dataframe['peak_bottom'] = peak_bottom['peak_bottom'] + strength['strength']

        return dataframe

    @staticmethod
    def smartentry(df: DataFrame, current_pair: str, entry_pairs: list, advance_pairs: list) -> DataFrame:
        dataframe = df.copy()

        signals = DataFrame(data="NA", index=dataframe.index, columns=["signals"], dtype=str)
        is_entry = current_pair.replace('/', '') in entry_pairs
        is_advance = current_pair.replace('/', '') in advance_pairs
        length = signals.__len__()
        signals["signals"].iat[length - 1] = "Long" if is_entry else "Advance" if is_advance else "Short"
        # print(metadata["pair"].replace('/', '') + ": " + signals["signals"][length-1])

        dataframe["signals"] = signals["signals"]

        return dataframe