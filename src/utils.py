import pandas as pd


def create_regression_summary_dataframe(model):
    results_as_html = model.summary().tables[1].as_html()
    return pd.read_html(results_as_html, header=0, index_col=0)[0]


def extract_arima_coeffs(df_model_summary):
    RENAME_DICT = {'coef': 'beta',
                   'P>|z|': 'pvalues'}

    df_model_summary_long = df_model_summary.\
        rename(columns=RENAME_DICT).\
        reset_index().loc[:, ['index', 'beta', 'pvalues']]. \
        melt(id_vars='index')

    df_model_summary_long['index_comb'] = df_model_summary_long['variable'] + '_' + df_model_summary_long['index']

    df_model_summary_long.drop(columns=['index', 'variable'], inplace=True)
    df_model_summary_long.set_index('index_comb', inplace=True)

    return df_model_summary_long
