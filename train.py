from datetime import datetime

import mlflow
from pmdarima.arima import auto_arima
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import D_GATE
from src.loader import Loader
from src.cleaner import Cleaner
from src.utils import create_regression_summary_dataframe, extract_arima_coeffs

if __name__ == "__main__":
    loader = Loader()
    cleaner = Cleaner()

    raw_data = loader.load_data()
    cleaned_data = cleaner.clean_data(raw_data)

    train = cleaned_data[:D_GATE]
    test = cleaned_data[D_GATE:]

    timestamp = datetime.now().strftime(format="%Y%m%d-%H%M%S")
    experiment_name_t = f"ts_-{timestamp}"

    with mlflow.start_run(run_name=experiment_name_t) as run:
        for exog in [True]:
            if exog:
                mod_vol = auto_arima(train['vol'],
                                     exogenous=train[['por', 'bmw']],
                                     m=5, max_p=5, max_q=5)
            else:
                mod_vol = auto_arima(train['vol'],
                                     m=5, max_p=5, max_q=5)

            df_arima_summary = create_regression_summary_dataframe(mod_vol)
            dict_summary = extract_arima_coeffs(df_arima_summary).to_dict()
            arima_order_label = " ".join(str(x) for x in list(mod_vol.order))

            ax = sns.lineplot(data=train, x=train.index, y='vol')

            mlflow.log_param("arima order", arima_order_label)
            mlflow.log_params(dict_summary['value'])

            fig = ax.get_figure()
            mlflow.log_figure(fig, f"figures/volkswagen.png")
