from sklearn.model_selection import LeaveOneOut
from tqdm import tqdm
import pandas as pd
import numpy as np
import fire

from dock2hit.fine_tuning.process_data import read_and_process_ugi_data
from dock2hit.fine_tuning.model_prediction import train_and_score


def run_leave_one_out(save_path):
    df_ugi_data = read_and_process_ugi_data().query('data_source != "new"')

    n_models = 1
    tautomer_mpnn = '/rds-d2/user/wjm41/hpc-work/models/dock2hit/ugi/ugi_taut/model_mol9039221.ckpt'

    loo = LeaveOneOut()

    df_with_predictions = []
    for train_ind, test_ind in tqdm(loo.split(df_ugi_data['mpnn_fp']), total=len(df_ugi_data)):
        df_train, df_test = df_ugi_data.iloc[train_ind], df_ugi_data.iloc[test_ind]
        df_score = train_and_score(df_test, input_rep='mpnn_fp', n_models=n_models,
                                   model_ckpt=tautomer_mpnn, df_training_data=df_train)
        df_score_morgan = train_and_score(
            df_test, input_rep='morgan_fp', n_models=n_models, model_ckpt=tautomer_mpnn, df_training_data=df_train)

        df_score['morgan_fp_pred_mean'] = df_score_morgan['morgan_fp_pred_mean'].values
        df_score['predicted_IC50_morgan_fp'] = df_score_morgan['predicted_IC50_morgan_fp'].values
        df_with_predictions.append(df_score)

    df_with_predictions = pd.concat(df_with_predictions)
    df_with_predictions['predicted_IC50_mpnn_fp'] = np.power(
        10, -(df_with_predictions['mpnn_fp_pred_mean']-6))
    df_with_predictions['predicted_IC50_morgan_fp'] = np.power(
        10, -(df_with_predictions['morgan_fp_pred_mean']-6))

    df_with_predictions.to_csv(save_path, index=False)

    return


if __name__ == '__main__':
    fire.Fire(run_leave_one_out)
