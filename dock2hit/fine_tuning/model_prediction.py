import plotly.express as px
import pandas as pd
import numpy as np

from useful_rdkit_utils import add_molecule_and_errors, mol2numpy_fp

from dock2hit.generate_mpnn_fps import generate_mpnn_fps_from_dataframe
from dock2hit.fine_tuning.process_data import canon_tautomers
from dock2hit.fine_tuning.model_fitting import fit_forest


def run_prediction_on_dataframe(df,
                                model,
                                load_name: str,
                                input_rep: str = 'mpnn_fp',
                                savename: str = None,
                                canon: bool = False,
                                uncertain: bool = False):

    df_score = df.copy()
    if canon:
        # already done so just save some time lmao
        df_score['SMILES'] = canon_tautomers(df_score['SMILES'])
    if input_rep == 'mpnn_fp':
        x_new = generate_mpnn_fps_from_dataframe(df_score, load_name=load_name)
    else:
        add_molecule_and_errors(df_score, mol_col_name='mol')
        # x_new = df_score['mol'].apply(mol2numpy_fp).values
        x_new = [x for x in df_score['mol'].apply(mol2numpy_fp)]

    if uncertain:
        preds, var = model.predict(x_new, no_var=False)
        df_score['predicted_var'] = var
    else:
        # print(x_new)
        preds = model.predict(x_new)

    df_score['predicted_pIC50'] = preds
    # df_score = df_score.sort_values(by='predicted_pIC50', ascending=False)

    if savename:
        df_score.to_csv(savename, index=False)
    return df_score


def merge_ensemble_of_score_dfs(list_of_dfs, names_of_scores):
    df_merged = pd.concat(list_of_dfs, axis=1)
    df_merged = df_merged.loc[:, ~df_merged.columns.duplicated()]
    df_merged['pred_mean'] = np.mean(df_merged[names_of_scores], axis=1)
    df_merged['pred_std'] = np.std(df_merged[names_of_scores], axis=1)
    return df_merged.sort_values(by='pred_mean', ascending=False)


def train_and_score(df_to_score: pd.DataFrame,
                    df_training_data: pd.DataFrame,
                    model_ckpt: str,
                    input_rep: str = 'mpnn_fp',
                    n_models: int = 5,

                    ):
    list_of_score_dfs = []
    list_of_score_names = []
    for n in range(n_models):
        score_name = 'score_'+str(n)
        list_of_score_names.append(score_name)
        x = np.vstack(df_training_data[input_rep].to_numpy())
        random_forest_trained_on_all_data = fit_forest(
            x, df_training_data['pIC50'].to_numpy())

        df_with_predictions = run_prediction_on_dataframe(
            df_to_score, model=random_forest_trained_on_all_data, input_rep=input_rep, load_name=model_ckpt)
        df_with_predictions.rename(
            columns={'predicted_pIC50': score_name}, inplace=True)
        print(df_with_predictions)
        list_of_score_dfs.append(df_with_predictions)

    df_mean = merge_ensemble_of_score_dfs(
        list_of_score_dfs, names_of_scores=list_of_score_names)
    df_mean = df_mean.rename(
        columns={'pred_mean': f'{input_rep}_pred_mean'})
    df_mean[f'predicted_IC50_{input_rep}'] = np.power(
        10, -(df_mean[f'{input_rep}_pred_mean']-6))

    return df_mean


def plot_regression_comparison(df_with_scores, title):

    fig_scatter = px.scatter(df_with_scores,
                             x='IC50',
                             y='predicted_IC50_mpnn_fp',
                             log_x=True,
                             log_y=True,
                             height=800,
                             title=title)

    fig_scatter_morgan = px.scatter(df_with_scores,
                                    x='IC50',
                                    y='predicted_IC50_morgan_fp',
                                    color_discrete_sequence=['red'],
                                    log_x=True,
                                    log_y=True)

    fig_scatter['data'][0]['showlegend'] = True
    fig_scatter['data'][0]['name'] = 'mpnn_fp'
    fig_scatter_morgan['data'][0]['showlegend'] = True
    fig_scatter_morgan['data'][0]['name'] = 'morgan_fp'

    fig_scatter.add_trace(fig_scatter_morgan.data[0])
    fig_scatter.add_shape(type='line',
                          x0=0.1,
                          x1=100,
                          y0=0.1,
                          y1=100,
                          line=dict(color='black', dash='dash'))
    fig_scatter.show()
