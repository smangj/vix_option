# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import abc
from functools import partial
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List

import fire
import requests
import pandas as pd
from tqdm import tqdm
from loguru import logger

from utils.qlib import deco_retry, get_calendar_list, get_trading_date_by_shift
from utils.qlib import get_instruments

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))


class IndexBase:
    DEFAULT_END_DATE = pd.Timestamp("2099-12-31")
    SYMBOL_FIELD_NAME = "symbol"
    DATE_FIELD_NAME = "date"
    START_DATE_FIELD = "start_date"
    END_DATE_FIELD = "end_date"
    CHANGE_TYPE_FIELD = "type"
    INSTRUMENTS_COLUMNS = [SYMBOL_FIELD_NAME, START_DATE_FIELD, END_DATE_FIELD]
    REMOVE = "remove"
    ADD = "add"
    INST_PREFIX = ""

    def __init__(
        self,
        index_name: str,
        qlib_dir: [str, Path] = None,
        freq: str = "day",
        request_retry: int = 5,
        retry_sleep: int = 3,
    ):
        """

        Parameters
        ----------
        index_name: str
            index name
        qlib_dir: str
            qlib directory, by default Path(__file__).resolve().parent.joinpath("qlib_data")
        freq: str
            freq, value from ["day", "1min"]
        request_retry: int
            request retry, by default 5
        retry_sleep: int
            request sleep, by default 3
        """
        self.index_name = index_name
        if qlib_dir is None:
            qlib_dir = Path(__file__).resolve().parent.joinpath("qlib_data")
        self.instruments_dir = (
            Path(qlib_dir).expanduser().resolve().joinpath("instruments")
        )
        self.instruments_dir.mkdir(exist_ok=True, parents=True)
        self.cache_dir = (
            Path(f"~/.cache/qlib/index/{self.index_name}").expanduser().resolve()
        )
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self._request_retry = request_retry
        self._retry_sleep = retry_sleep
        self.freq = freq

    @property
    @abc.abstractmethod
    def bench_start_date(self) -> pd.Timestamp:
        """
        Returns
        -------
            index start date
        """
        raise NotImplementedError("rewrite bench_start_date")

    @property
    @abc.abstractmethod
    def calendar_list(self) -> List[pd.Timestamp]:
        """get history trading date

        Returns
        -------
            calendar list
        """
        raise NotImplementedError("rewrite calendar_list")

    @abc.abstractmethod
    def get_new_companies(self) -> pd.DataFrame:
        """

        Returns
        -------
            pd.DataFrame:

                symbol     start_date    end_date
                SH600000   2000-01-01    2099-12-31

            dtypes:
                symbol: str
                start_date: pd.Timestamp
                end_date: pd.Timestamp
        """
        raise NotImplementedError("rewrite get_new_companies")

    @abc.abstractmethod
    def get_changes(self) -> pd.DataFrame:
        """get companies changes

        Returns
        -------
            pd.DataFrame:
                symbol      date        type
                SH600000  2019-11-11    add
                SH600000  2020-11-10    remove
            dtypes:
                symbol: str
                date: pd.Timestamp
                type: str, value from ["add", "remove"]
        """
        raise NotImplementedError("rewrite get_changes")

    @abc.abstractmethod
    def format_datetime(self, inst_df: pd.DataFrame) -> pd.DataFrame:
        """formatting the datetime in an instrument

        Parameters
        ----------
        inst_df: pd.DataFrame
            inst_df.columns = [self.SYMBOL_FIELD_NAME, self.START_DATE_FIELD, self.END_DATE_FIELD]

        Returns
        -------

        """
        raise NotImplementedError("rewrite format_datetime")

    def save_new_companies(self):
        """save new companies

        Examples
        -------
            $ python collector.py save_new_companies --index_name CSI300 --qlib_dir ~/.qlib/qlib_data/cn_data
        """
        df = self.get_new_companies()
        if df is None or df.empty:
            raise ValueError(f"get new companies error: {self.index_name}")
        df = df.drop_duplicates([self.SYMBOL_FIELD_NAME])
        df.loc[:, self.INSTRUMENTS_COLUMNS].to_csv(
            self.instruments_dir.joinpath(f"{self.index_name.lower()}_only_new.txt"),
            sep="\t",
            index=False,
            header=None,
        )

    def get_changes_with_history_companies(
        self, history_companies: pd.DataFrame
    ) -> pd.DataFrame:
        """get changes with history companies

        Parameters
        ----------
        history_companies : pd.DataFrame
            symbol        date
            SH600000   2020-11-11

            dtypes:
                symbol: str
                date: pd.Timestamp

        Return
        --------
            pd.DataFrame:
                symbol      date        type
                SH600000  2019-11-11    add
                SH600000  2020-11-10    remove
            dtypes:
                symbol: str
                date: pd.Timestamp
                type: str, value from ["add", "remove"]

        """
        logger.info("parse changes from history companies......")
        last_code = []
        result_df_list = []
        _columns = [
            self.DATE_FIELD_NAME,
            self.SYMBOL_FIELD_NAME,
            self.CHANGE_TYPE_FIELD,
        ]
        for _trading_date in tqdm(
            sorted(history_companies[self.DATE_FIELD_NAME].unique(), reverse=True)
        ):
            _currenet_code = history_companies[
                history_companies[self.DATE_FIELD_NAME] == _trading_date
            ][self.SYMBOL_FIELD_NAME].tolist()
            if last_code:
                add_code = list(set(last_code) - set(_currenet_code))
                remote_code = list(set(_currenet_code) - set(last_code))
                for _code in add_code:
                    result_df_list.append(
                        pd.DataFrame(
                            [
                                [
                                    get_trading_date_by_shift(
                                        self.calendar_list, _trading_date, 1
                                    ),
                                    _code,
                                    self.ADD,
                                ]
                            ],
                            columns=_columns,
                        )
                    )
                for _code in remote_code:
                    result_df_list.append(
                        pd.DataFrame(
                            [
                                [
                                    get_trading_date_by_shift(
                                        self.calendar_list, _trading_date, 0
                                    ),
                                    _code,
                                    self.REMOVE,
                                ]
                            ],
                            columns=_columns,
                        )
                    )
            last_code = _currenet_code
        df = pd.concat(result_df_list)
        logger.info("end of parse changes from history companies.")
        return df

    def parse_instruments(self):
        """parse instruments, eg: csi300.txt

        Examples
        -------
            $ python collector.py parse_instruments --index_name CSI300 --qlib_dir ~/.qlib/qlib_data/cn_data
        """
        logger.info(f"start parse {self.index_name.lower()} companies.....")
        instruments_columns = [
            self.SYMBOL_FIELD_NAME,
            self.START_DATE_FIELD,
            self.END_DATE_FIELD,
        ]
        changers_df = self.get_changes()
        new_df = self.get_new_companies()
        if new_df is None or new_df.empty:
            raise ValueError(f"get new companies error: {self.index_name}")
        new_df = new_df.copy()
        logger.info("parse history companies by changes......")
        for _row in tqdm(
            changers_df.sort_values(self.DATE_FIELD_NAME, ascending=False).itertuples(
                index=False
            )
        ):
            if _row.type == self.ADD:
                min_end_date = new_df.loc[
                    new_df[self.SYMBOL_FIELD_NAME] == _row.symbol, self.END_DATE_FIELD
                ].min()
                new_df.loc[
                    (new_df[self.END_DATE_FIELD] == min_end_date)
                    & (new_df[self.SYMBOL_FIELD_NAME] == _row.symbol),
                    self.START_DATE_FIELD,
                ] = _row.date
            else:
                _tmp_df = pd.DataFrame(
                    [[_row.symbol, self.bench_start_date, _row.date]],
                    columns=instruments_columns,
                )
                new_df = pd.concat([new_df, _tmp_df], sort=False)

        inst_df = new_df.loc[:, instruments_columns]
        _inst_prefix = self.INST_PREFIX.strip()
        if _inst_prefix:
            inst_df["save_inst"] = inst_df[self.SYMBOL_FIELD_NAME].apply(
                lambda x: f"{_inst_prefix}{x}"
            )
        inst_df = self.format_datetime(inst_df)
        inst_df.to_csv(
            self.instruments_dir.joinpath(f"{self.index_name.lower()}.txt"),
            sep="\t",
            index=False,
            header=None,
        )
        logger.info(f"parse {self.index_name.lower()} companies finished.")


WIKI_URL = "https://en.wikipedia.org/wiki"

WIKI_INDEX_NAME_MAP = {
    "NASDAQ100": "NASDAQ-100",
    "SP500": "List_of_S%26P_500_companies",
    "SP400": "List_of_S%26P_400_companies",
    "DJIA": "Dow_Jones_Industrial_Average",
}


class WIKIIndex(IndexBase):
    # NOTE: The US stock code contains "PRN", and the directory cannot be created on Windows system, use the "_" prefix
    # https://superuser.com/questions/613313/why-cant-we-make-con-prn-null-folder-in-windows
    INST_PREFIX = ""

    def __init__(
        self,
        index_name: str,
        qlib_dir: [str, Path] = None,
        freq: str = "day",
        request_retry: int = 5,
        retry_sleep: int = 3,
    ):
        super(WIKIIndex, self).__init__(
            index_name=index_name,
            qlib_dir=qlib_dir,
            freq=freq,
            request_retry=request_retry,
            retry_sleep=retry_sleep,
        )

        self._target_url = f"{WIKI_URL}/{WIKI_INDEX_NAME_MAP[self.index_name.upper()]}"

    @property
    @abc.abstractmethod
    def bench_start_date(self) -> pd.Timestamp:
        """
        Returns
        -------
            index start date
        """
        raise NotImplementedError("rewrite bench_start_date")

    @abc.abstractmethod
    def get_changes(self) -> pd.DataFrame:
        """get companies changes

        Returns
        -------
            pd.DataFrame:
                symbol      date        type
                SH600000  2019-11-11    add
                SH600000  2020-11-10    remove
            dtypes:
                symbol: str
                date: pd.Timestamp
                type: str, value from ["add", "remove"]
        """
        raise NotImplementedError("rewrite get_changes")

    def format_datetime(self, inst_df: pd.DataFrame) -> pd.DataFrame:
        """formatting the datetime in an instrument

        Parameters
        ----------
        inst_df: pd.DataFrame
            inst_df.columns = [self.SYMBOL_FIELD_NAME, self.START_DATE_FIELD, self.END_DATE_FIELD]

        Returns
        -------

        """
        if self.freq != "day":
            inst_df[self.END_DATE_FIELD] = inst_df[self.END_DATE_FIELD].apply(
                lambda x: (
                    pd.Timestamp(x) + pd.Timedelta(hours=23, minutes=59)
                ).strftime("%Y-%m-%d %H:%M:%S")
            )
        return inst_df

    @property
    def calendar_list(self) -> List[pd.Timestamp]:
        """get history trading date

        Returns
        -------
            calendar list
        """
        _calendar_list = getattr(self, "_calendar_list", None)
        if _calendar_list is None:
            _calendar_list = list(
                filter(
                    lambda x: x >= self.bench_start_date, get_calendar_list("US_ALL")
                )
            )
            setattr(self, "_calendar_list", _calendar_list)
        return _calendar_list

    def _request_new_companies(self) -> requests.Response:
        resp = requests.get(self._target_url)
        if resp.status_code != 200:
            raise ValueError(f"request error: {self._target_url}")

        return resp

    def set_default_date_range(self, df: pd.DataFrame) -> pd.DataFrame:
        _df = df.copy()
        _df[self.SYMBOL_FIELD_NAME] = _df[self.SYMBOL_FIELD_NAME].str.strip()
        _df[self.START_DATE_FIELD] = self.bench_start_date
        _df[self.END_DATE_FIELD] = self.DEFAULT_END_DATE
        return _df.loc[:, self.INSTRUMENTS_COLUMNS]

    def get_new_companies(self):
        logger.info(f"get new companies {self.index_name} ......")
        _data = deco_retry(retry=self._request_retry, retry_sleep=self._retry_sleep)(
            self._request_new_companies
        )()
        df_list = pd.read_html(_data.text)
        for _df in df_list:
            _df = self.filter_df(_df)
            if (_df is not None) and (not _df.empty):
                _df.columns = [self.SYMBOL_FIELD_NAME]
                _df = self.set_default_date_range(_df)
                logger.info(f"end of get new companies {self.index_name} ......")
                return _df

    def filter_df(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("rewrite filter_df")


class NASDAQ100Index(WIKIIndex):
    HISTORY_COMPANIES_URL = (
        "https://indexes.nasdaqomx.com/Index/WeightingData?id=NDX&tradeDate="
        "{trade_date}T00%3A00%3A00.000&timeOfDay=SOD"
    )
    MAX_WORKERS = 16

    def filter_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) >= 100 and "Ticker" in df.columns:
            return df.loc[:, ["Ticker"]].copy()

    @property
    def bench_start_date(self) -> pd.Timestamp:
        return pd.Timestamp("2003-01-02")

    @deco_retry
    def _request_history_companies(
        self, trade_date: pd.Timestamp, use_cache: bool = True
    ) -> pd.DataFrame:
        trade_date = trade_date.strftime("%Y-%m-%d")
        cache_path = self.cache_dir.joinpath(f"{trade_date}_history_companies.pkl")
        if cache_path.exists() and use_cache:
            df = pd.read_pickle(cache_path)
        else:
            url = self.HISTORY_COMPANIES_URL.format(trade_date=trade_date)
            resp = requests.post(url)
            if resp.status_code != 200:
                raise ValueError(f"request error: {url}")
            df = pd.DataFrame(resp.json()["aaData"])
            df[self.DATE_FIELD_NAME] = trade_date
            df.rename(
                columns={"Name": "name", "Symbol": self.SYMBOL_FIELD_NAME}, inplace=True
            )
            if not df.empty:
                df.to_pickle(cache_path)
        return df

    def get_history_companies(self):
        logger.info("start get history companies......")
        all_history = []
        error_list = []
        with tqdm(total=len(self.calendar_list)) as p_bar:
            with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
                for _trading_date, _df in zip(
                    self.calendar_list,
                    executor.map(self._request_history_companies, self.calendar_list),
                ):
                    if _df.empty:
                        error_list.append(_trading_date)
                    else:
                        all_history.append(_df)
                    p_bar.update()

        if error_list:
            logger.warning("get error: {error_list}")
        logger.info("total {len(self.calendar_list)}, error {len(error_list)}")
        logger.info("end of get history companies.")
        return pd.concat(all_history, sort=False)

    def get_changes(self):
        return self.get_changes_with_history_companies(self.get_history_companies())


class DJIAIndex(WIKIIndex):
    @property
    def bench_start_date(self) -> pd.Timestamp:
        return pd.Timestamp("2000-01-01")

    def get_changes(self) -> pd.DataFrame:
        pass

    def filter_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if "Symbol" in df.columns:
            _df = df.loc[:, ["Symbol"]].copy()
            _df["Symbol"] = _df["Symbol"].apply(lambda x: x.split(":")[-1])
            return _df

    def parse_instruments(self):
        logger.warning("No suitable data source has been found!")


class SP500Index(WIKIIndex):
    WIKISP500_CHANGES_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    @property
    def bench_start_date(self) -> pd.Timestamp:
        return pd.Timestamp("1999-01-01")

    def get_changes(self) -> pd.DataFrame:
        logger.info("get sp500 history changes......")
        # NOTE: may update the index of the table
        changes_df = pd.read_html(self.WIKISP500_CHANGES_URL)[-1]
        changes_df = changes_df.iloc[:, [0, 1, 3]]
        changes_df.columns = [self.DATE_FIELD_NAME, self.ADD, self.REMOVE]
        changes_df[self.DATE_FIELD_NAME] = pd.to_datetime(
            changes_df[self.DATE_FIELD_NAME]
        )
        _result = []
        for _type in [self.ADD, self.REMOVE]:
            _df = changes_df.copy()
            _df[self.CHANGE_TYPE_FIELD] = _type
            _df[self.SYMBOL_FIELD_NAME] = _df[_type]
            _df.dropna(subset=[self.SYMBOL_FIELD_NAME], inplace=True)
            if _type == self.ADD:
                _df[self.DATE_FIELD_NAME] = _df[self.DATE_FIELD_NAME].apply(
                    lambda x: get_trading_date_by_shift(self.calendar_list, x, 0)
                )
            else:
                _df[self.DATE_FIELD_NAME] = _df[self.DATE_FIELD_NAME].apply(
                    lambda x: get_trading_date_by_shift(self.calendar_list, x, -1)
                )
            _result.append(
                _df[
                    [
                        self.DATE_FIELD_NAME,
                        self.CHANGE_TYPE_FIELD,
                        self.SYMBOL_FIELD_NAME,
                    ]
                ]
            )
        logger.info("end of get sp500 history changes.")
        return pd.concat(_result, sort=False)

    def filter_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if "Symbol" in df.columns:
            return df.loc[:, ["Symbol"]].copy()


class SP400Index(WIKIIndex):
    @property
    def bench_start_date(self) -> pd.Timestamp:
        return pd.Timestamp("2000-01-01")

    def get_changes(self) -> pd.DataFrame:
        pass

    def filter_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if "Ticker symbol" in df.columns:
            return df.loc[:, ["Ticker symbol"]].copy()

    def parse_instruments(self):
        logger.warning("No suitable data source has been found!")


if __name__ == "__main__":
    fire.Fire(partial(get_instruments, market_index="us_index"))
