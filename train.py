import mlflow
from pmdarima.arima import auto_arima

from src.config import D_GATE
from src.loader import Loader
from src.cleaner import Cleaner

if __name__ == "__main__":
    loader = Loader()
    cleaner = Cleaner()

    raw_data = loader.load_data()
    cleaned_data = cleaner.clean_data(raw_data)

    train = cleaned_data[:D_GATE]
    test = cleaned_data[D_GATE:]

    with mlflow.start_run(run_name="ts_training_1") as run:
        for exog in [True]:
            if exog:
                mod_vol = auto_arima(train['vol'],
                                     exogenous=train[['por', 'bmw']],
                                     m=5, max_p=5, max_q=5)
            else:
                mod_vol = auto_arima(train['vol'],
                                     m=5, max_p=5, max_q=5)

            arima_order_label = " ".join(str(x) for x in list(mod_vol.order))

            mlflow.log_param("arima order", arima_order_label)
