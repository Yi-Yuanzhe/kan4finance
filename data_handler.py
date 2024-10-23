import pandas as pd
import torch

from torch import Tensor
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class DataHandler:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_data(self):
        # Load csv
        data_df = pd.read_csv(self.data_path)
        return data_df

    def preprocess_data(self, data:DataFrame) -> Tensor:
        # drop na
        data.dropna(axis=0, how='any', inplace=True)

        # train test split
        train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)

        train_features_df = train_df[['Beta', 'logme', 'log(B/M)', 'Mom12m', 'GP', 'AssetGrowth', 'IdioVol3F', 'SRreversal']]
        test_features_df  = test_df[['Beta', 'logme', 'log(B/M)', 'Mom12m', 'GP', 'AssetGrowth', 'IdioVol3F', 'SRreversal']]

        train_label_df    = train_df[['ExRet']]
        test_label_df     = test_df[['ExRet']]

        train_feature_tensor = torch.tensor(train_features_df.values, dtype=torch.float32)
        train_label_tensor   = torch.tensor(train_label_df.values, dtype=torch.float32)

        test_feature_tensor  = torch.tensor(test_features_df.values, dtype=torch.float32)
        test_label_tensor    = torch.tensor(test_label_df.values, dtype=torch.float32)

        # Winsorize
        def winsorize_tensor(tensor, lower_quantile=0.01, upper_quantile=0.99):
            lower_bounds = torch.quantile(tensor, lower_quantile, dim=0)
            upper_bounds = torch.quantile(tensor, upper_quantile, dim=0)
            
            tensor_winsorized = torch.clamp(tensor, min=lower_bounds, max=upper_bounds)
            
            return tensor_winsorized
        
        train_feature = winsorize_tensor(train_feature_tensor)
        # train_label   = winsorize_tensor(train_label_tensor)
        train_label   = train_label_tensor

        test_feature  = winsorize_tensor(test_feature_tensor)
        # test_label    = winsorize_tensor(test_label_tensor)
        test_label    = test_label_tensor

        # Normalize
        train_feature_mean = train_feature.mean(dim=0)
        train_feature_std  = train_feature.std(dim=0)
        # train_feature_min  = train_feature.min(dim=0).values
        # train_feature_max  = train_feature.max(dim=0).values

        train_label_mean = train_label.mean(dim=0)
        train_label_std  = train_label.std(dim=0)
        # train_label_min  = train_label.min(dim=0).values
        # train_label_max  = train_label.max(dim=0).values

        train_feature_mean[0] = 0
        train_feature_std[0]  = 1

        train_feature = (train_feature - train_feature_mean) / train_feature_std
        train_label   = (train_label - train_label_mean) / train_label_std

        test_feature  = (test_feature - train_feature_mean) / train_feature_std
        test_label    = (test_label - train_label_mean) / train_label_std

        # Create dataset tensor
        dataset_tensor = {
            'train_input': train_feature,
            'train_label': train_label,
            'test_input' : test_feature,
            'test_label' : test_label
        }
        return dataset_tensor

    def get_dataset(self) -> Tensor:
        data = self.load_data()
        dataset = self.preprocess_data(data)
        return dataset