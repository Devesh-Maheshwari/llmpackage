import os
import pandas as pd


class Helper:
    @staticmethod
    def concat_files_in_folder(folder_path):
        dfs = []
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if filename.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif filename.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            dfs.append(df)

        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        
    @staticmethod
    def iter_files_in_folder(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if filename.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif filename.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            
            yield filename, df


    @staticmethod
    def save_or_update_sheets(file_path, df_key, _df):
        if not os.path.exists(file_path):
            pd.DataFrame().to_excel(file_path, index=False)
        with pd.ExcelWriter(file_path, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
            _df.to_excel(writer, sheet_name=df_key, index=False)
