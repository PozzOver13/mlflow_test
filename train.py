from datetime import datetime

import mlflow
from pmdarima.arima import auto_arima
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import D_GATE, EXPERIMENT_NAME, ARTIFACT_PATH
from src.loader import Loader
from src.cleaner import Cleaner
from src.utils import create_regression_summary_dataframe, extract_arima_coeffs

EXPERIMENT_ID = dict(mlflow.get_experiment_by_name(EXPERIMENT_NAME))['experiment_id']

# mlflow.set_tracking_uri(ARTIFACT_PATH)

if __name__ == "__main__":
    loader = Loader()
    cleaner = Cleaner()

    raw_data = loader.load_data()
    cleaned_data = cleaner.clean_data(raw_data)

    train = cleaned_data[:D_GATE]
    test = cleaned_data[D_GATE:]

    for exog in [True, False]:
        timestamp = datetime.now().strftime(format="%Y%m%d%H%M%S")
        run_name_t = f"ts_volkswagen_{exog}_{timestamp}"
        with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name=run_name_t) as run:
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

            ax = sns.lineplot(x=train.index, y=mod_vol.resid())

            # Metrics
            mlflow.log_metric("residuals mean", mod_vol.resid().mean())
            mlflow.log_metric("residuals standard deviation", mod_vol.resid().std())

            # Params
            mlflow.log_param("arima order", arima_order_label)
            mlflow.log_param("exogenous", exog)

            # Artifacts
            fig = ax.get_figure()
            mlflow.log_figure(fig, f"figures/volkswagen_residuals.png")

            df_arima_summary.reset_index().to_excel("tmp/my_tracking/volkswagen.xlsx", index=False)
            mlflow.log_artifact("tmp/my_tracking/volkswagen.xlsx",
                                "model_summary")

            df_arima_summary.reset_index().to_html("tmp/my_tracking/volkswagen.html", index=False)
            mlflow.log_artifact("tmp/my_tracking/volkswagen.html",
                                "model_summary")
