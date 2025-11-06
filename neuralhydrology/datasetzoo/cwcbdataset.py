from functools import reduce
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import xarray

from neuralhydrology.datasetzoo.basedataset import BaseDataset
from neuralhydrology.utils.config import Config


class CWCB(BaseDataset):

    def __init__(self,
                 cfg: Config,
                 is_train: bool,
                 period: str,
                 basin: str = None,
                 additional_features: List[Dict[str, pd.DataFrame]] = [],
                 id_to_int: Dict[str, int] = {},
                 scaler: Dict[str, Union[pd.Series, xarray.DataArray]] = {}):

        # Initialize `BaseDataset` class
        super(CWCB, self).__init__(cfg=cfg,
                                       is_train=is_train,
                                       period=period,
                                       basin=basin,
                                       additional_features=additional_features,
                                       id_to_int=id_to_int,
                                       scaler=scaler)

    def _load_basin_data(self, basin: str) -> pd.DataFrame:
        """Load timeseries data of one specific basin"""
        return load_timeseries(data_dir=self.cfg.data_dir, basin=basin)

    def _load_attributes(self) -> pd.DataFrame:
        """Load catchment attributes"""
        return load_basin_characteristics(self.cfg.data_dir, basins=self.basins)

def load_timeseries(data_dir: Path, basin: str) -> pd.DataFrame:
    preprocessed_dir = data_dir

    # make sure the CAMELS-CL data was already preprocessed and per-basin files exist.
    if not preprocessed_dir.is_dir():
        msg = [
            f"No preprocessed data directory found at {preprocessed_dir}. Use preprocessed_camels_cl_dataset ",
             "in neuralhydrology.datasetzoo.camelscl to preprocess the CAMELS CL data set once into ",
             "per-basin files."
        ]
        raise FileNotFoundError("".join(msg))

    # load the data for the specific basin into a time-indexed dataframe
    basin_file = preprocessed_dir / f"{basin}.csv"
    df = pd.read_csv(basin_file, index_col='date', parse_dates=['date'])
    df['gage'] = df['gage'].astype(str).str.zfill(8)
    # add logic for localizing time zone
    try:
        df.index = df.index.tz_localize(None)
    except:
        df.index = pd.to_datetime(df.index).tz_localize(None)
    return df

def load_basin_characteristics(data_dir: Path, basins: List[str] = []) -> pd.DataFrame:
    # load attributes into basin-indexed dataframe
    attributes_file = data_dir / 'basinCharacteristics.csv'
    df = pd.read_csv(attributes_file)
    #df = df.apply(pd.to_numeric)

    df['gage'] = df['gage'].astype(str).str.zfill(8)
    df.index = df['gage']

    # convert all columns, where possible, to numeric

    if basins:
        if any(b not in df.index for b in basins):
            missingGages = set(basins)-set(df.index)
            #print(f'set of basins: {set(basins)}')
            #print(f'missing basin static attributes: {missingGages}')
            raise ValueError('Some basins are missing static attributes.')
        df = df.loc[basins]

    return df
