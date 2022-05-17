

class Cleaner:

    def __init__(self):
        pass

    def clean_data(self, df):
        processed_data = self.pre_processes_data(df)

        return processed_data

    def pre_processes_data(self, df_input):
        df = df_input.copy()

        # Extracting Closing Prices
        df['vol'] = df['VOW3.DE'].Close
        df['por'] = df['PAH3.DE'].Close
        df['bmw'] = df['BMW.DE'].Close

        # Creating Returns
        df['ret_vol'] = df['vol'].pct_change(1).mul(100)
        df['ret_por'] = df['por'].pct_change(1).mul(100)
        df['ret_bmw'] = df['bmw'].pct_change(1).mul(100)

        # Creating Squared Returns
        df['sq_vol'] = df['ret_vol'].mul(df.ret_vol)
        df['sq_por'] = df['ret_por'].mul(df.ret_por)
        df['sq_bmw'] = df['ret_bmw'].mul(df.ret_bmw)

        # Extracting Volume
        df['q_vol'] = df['VOW3.DE'].Volume
        df['q_por'] = df['PAH3.DE'].Volume
        df['q_bmw'] = df['BMW.DE'].Volume

        # Assigning the Frequency and Filling NA Values
        df = df.asfreq('b')
        df = df.fillna(method='bfill')

        # Removing Surplus Data
        del df['VOW3.DE']
        del df['PAH3.DE']
        del df['BMW.DE']

        # Flattening hierarchical index
        df.columns = [' '.join(col).strip() for col in df.columns.values]

        return df
