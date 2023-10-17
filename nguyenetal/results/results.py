import pandas as pd
import pickle
import os
import numpy as np
from sklearn import metrics, model_selection
import matplotlib.pyplot as plt
import seaborn as sns

def cross_validation_results(pipeline, feature, timepoint, specific=None):

    model_list = ['ElasticNet', 'LinearSVR', 'GradientBoostingRegressor', 'RandomForestRegressor']
    atlas_list = ['schaefer', 'basc197', 'basc444']
    
    results = []
    for model_name in model_list:
        for atlas in atlas_list: 
            output_dir = f'./outputs/{pipeline}/prediction_scores/'+\
                f'prediction-{timepoint}_atlas-{atlas}_feature-{feature}'

            if pipeline == 'no_imaging_features':
                output_dir = f'./outputs/{pipeline}/prediction_scores/'+\
                f'prediction-{timepoint}'

            if specific:
                output_dir += specific

            try:
                with open(f'{output_dir}/{model_name}_results.pkl', 'rb') as f:
                    model_dict = pickle.load(f)
                    f.close()
            except FileNotFoundError:
                print(f'{output_dir}/{model}_results.pkl')
                continue
    
            df_all_features = pd.read_csv(f'{output_dir}/data.csv',
                               header=0, index_col=None)
            df_outcome = pd.read_csv(f'{output_dir}/target.csv',
                               header=0, index_col=None)
        
            target = df_outcome['UPDRS_TOT'].to_numpy(copy=True)

            if 'EVENT_ID' in df_all_features.columns:
                df_all_features = df_all_features.drop(['EVENT_ID'], axis=1)
                
            inputs = df_all_features.astype(np.float64)
            
            outer = model_selection.LeaveOneOut()
    
            for i, (train_idx, test_idx) in enumerate(outer.split(inputs, target)):
                inputs_test = inputs.iloc[test_idx]
                label_test = target[test_idx]
                inputs_train = inputs.iloc[train_idx]
                model = model_dict['estimator'][i].best_estimator_
                pred = model.predict(inputs_test.astype(np.float64))
    
                search_results = pd.DataFrame(model_dict['estimator'][i].cv_results_)
                best_model = model_dict['estimator'][i].best_index_
                rsquare_valid = search_results['mean_test_rsquare'].iloc[best_model]
                rmse_valid = -search_results['mean_test_rmse'].iloc[best_model]
        
                results += [{'Model': model_name,
                             'Atlas': atlas,
                             'LOOCV Fold':i,
                             'Prediction': pred[0],
                             'Target': label_test[0],
                             'Val RMSE': rmse_valid,
                             'Val R2': rsquare_valid}]
                
    results_df = pd.DataFrame(results)

    return results_df


def prediction_results(df, threshold, select_across='folds'):
    best_model_df = pd.DataFrame()
    
    if select_across == 'folds':
        dict_scores=[]
        for fold in np.unique(df['LOOCV Fold']):
            df_fold = df[df['LOOCV Fold']==fold]
            best_model = df_fold.loc[(df_fold['Val RMSE']==np.min(df_fold['Val RMSE']))]
            best_model_df = pd.concat([best_model_df, best_model])

        best_model_df['True class'] = best_model_df['Target'] > threshold
        best_model_df['Predicted class'] = best_model_df['Prediction'] > threshold

        trueneg, falsepos, falseneg, truepos = metrics.confusion_matrix(best_model_df['True class'], 
                                         best_model_df['Predicted class'] ).ravel()
        r_square = metrics.r2_score(best_model_df['Target'], best_model_df['Prediction'])
        rmse = np.sqrt(metrics.mean_squared_error(best_model_df['Target'], best_model_df['Prediction']))
        auc = metrics.roc_auc_score(best_model_df['True class'], best_model_df['Predicted class'])
        ppv = metrics.precision_score(best_model_df['True class'], best_model_df['Predicted class'])
        npv = trueneg / (trueneg + falseneg)
        spec = trueneg / (trueneg + falsepos)
        sens = metrics.recall_score(best_model_df['True class'], best_model_df['Predicted class'])

        dict_scores += [{'model': best_model_df['Model'].value_counts().sort_values().index[-1],
                      'atlas': best_model_df['Atlas'].value_counts().sort_values().index[-1],
                      'r2': r_square,
                      'rmse': rmse,
                      'auc': auc,
                      'ppv': ppv, 
                      'npv': npv,
                      'spec': spec,
                      'sens': sens}]

        scores_df = pd.DataFrame(dict_scores)
            
    elif select_across == 'model':
        for model in np.unique(df['Model'].tolist()):
            for atlas in np.unique(df['Atlas'].tolist()):
                model_atlas_df = df.loc[(df['Model']==model) & (df['Atlas']==atlas)]
                mean_val_r2 = np.mean(model_atlas_df['Val R2']) 
                mean_val_rmse = np.mean(model_atlas_df['Val RMSE'])

                model_atlas_df['True class'] = model_atlas_df['Target'] > threshold
                model_atlas_df['Predicted class'] = model_atlas_df['Prediction'] > threshold

                trueneg, falsepos, falseneg, truepos = metrics.confusion_matrix(model_atlas_df['True class'], 
                                                model_atlas_df['Predicted class'] ).ravel()
                r_square = metrics.r2_score(model_atlas_df['Target'], model_atlas_df['Prediction'])
                rmse = np.sqrt(metrics.mean_squared_error(model_atlas_df['Target'], model_atlas_df['Prediction']))
                auc = metrics.roc_auc_score(model_atlas_df['True class'], model_atlas_df['Predicted class'])
                ppv = metrics.precision_score(model_atlas_df['True class'], model_atlas_df['Predicted class'])
                npv = trueneg / (trueneg + falseneg)
                spec = trueneg / (trueneg + falsepos)
                sens = metrics.recall_score(model_atlas_df['True class'], model_atlas_df['Predicted class'])

                model_atlas_summary_df = pd.DataFrame([{'model': model, 
                                      'atlas':atlas,
                                      'val_r2': mean_val_r2, 
                                      'val_rmse': mean_val_rmse,
                                      'r2': r_square,
                                      'rmse': rmse,
                                      'auc': auc,
                                      'ppv': ppv, 
                                      'npv': npv,
                                      'spec': spec,
                                      'sens': sens,
                                      'Prediction': model_atlas_df['Prediction'].tolist(),
                                      'Target': model_atlas_df['Target'].tolist()}])
                
                best_model_df = pd.concat([best_model_df, model_atlas_summary_df])

        best_model_df['best_model'] = [0 for i in range(len(best_model_df))]
        best_model_df['best_model'][best_model_df['r2'] == np.max(best_model_df['r2'])] = 1

        scores_df = best_model_df[best_model_df['best_model']==1]
        scores_df = scores_df.drop(['val_rmse', 'val_r2'], axis=1)

    return best_model_df, scores_df


timepoint_dict = {'baseline':'Baseline',
                 '1y': 'Year 1',
                 '2y': 'Year 2', 
                 '4y': 'Year 4'}

feature_dict = {'falff':'fALFF',
                'alff':'ALFF',
               'ReHo':'ReHo',
               'zfalff':'fALFF',
                'zalff':'ALFF',
               'zReHo':'ReHo'}

atlas_dict = {'schaefer':'Schaefer', 
             'basc197':'BASC197',
             'basc444': 'BASC444'}

model_dict = {'ElasticNet':'ElasticNet',
             'LinearSVR': 'SVM',
             'GradientBoostingRegressor': 'Gradient Boosting',
             'RandomForestRegressor': 'Random Forest'}

def plot_results_table(pipeline, feature_list=None, original_df_file=None, specific=None, display='best'):

    global_df = pd.DataFrame(columns = ['MDS-UPDRS Prediction target','Feature','Type','Best performing model',
                                        'Best performing parcellation', 'R2', 'RMSE', 'AUC', 
                                        'PPV', 'NPV', 'Spec.','Sens.'])
    if feature_list == None:
        feature_list = ['zalff', 'zfalff', 'zReHo']
        
    if pipeline == 'no_imaging_features':
        feature_list = ['ReHo']
        
    for timepoint in ['baseline', '1y', '2y', '4y']:
        for feature in feature_list:

            if original_df_file != None:
                original_df = pd.read_csv(original_df_file)
                # Look at results for original paper for timepoint and feature
                sub_original_df = original_df[original_df['MDS-UPDRS Prediction target']==timepoint_dict[timepoint]][original_df['Feature']==feature_dict[feature]]
                global_df = pd.concat([global_df, sub_original_df]) # Add the results to the table 

            # Look at results computed for replication
            if display != 'folds':
                if not specific:
                    df_feature_results = pd.read_csv(f'./outputs/{pipeline}/prediction_scores/prediction-{timepoint}_feature-{feature}_test_results.csv')
                else: 
                    df_feature_results = pd.read_csv(f'./outputs/{pipeline}/prediction_scores/prediction-{timepoint}_feature-{feature}{specific}_test_results.csv')

                if display == 'best':
                    df_feature_results = df_feature_results[df_feature_results['best_model']==1]
                if display == 'same':
                    original_best_model = sub_original_df['Best performing model'].iloc[0]
                    original_best_model = list(model_dict.keys())[list(model_dict.values()).index(original_best_model)]
                    original_best_atlas = sub_original_df['Best performing parcellation'].iloc[0]
                    original_best_atlas = list(atlas_dict.keys())[list(atlas_dict.values()).index(original_best_atlas)]

                    df_feature_results = df_feature_results.loc[(df_feature_results['model']==original_best_model) &\
                                                                (df_feature_results['atlas']==original_best_atlas)]


                for i in range(len(df_feature_results)):
                    sub_df = pd.DataFrame({
                        'MDS-UPDRS Prediction target': [timepoint_dict[timepoint]],
                        'Feature': [str(feature_dict[feature])],
                        'Type': ['Replication'],
                        'Best performing model': [model_dict[df_feature_results['model'].iloc[i]]],
                        'Best performing parcellation': [atlas_dict[df_feature_results['atlas'].iloc[i]]],
                        'R2': [round(df_feature_results['r2'].iloc[i],5)],
                        'RMSE': [round(df_feature_results['rmse'].iloc[i],3)],
                        'AUC':[round(df_feature_results['auc'].iloc[i],3)],
                        'PPV': [str(round(df_feature_results['ppv'].iloc[i]*100, 1))+'%'],
                        'NPV':[str(round(df_feature_results['npv'].iloc[i]*100,1))+'%'],
                        'Spec.': [str(round(df_feature_results['spec'].iloc[i]*100, 1))+'%'],
                        'Sens.': [str(round(df_feature_results['sens'].iloc[i]*100, 1))+'%']
                    })
    
                    global_df = pd.concat([global_df, sub_df])

    if pipeline == 'no_imaging_features':
        global_df['Best performing parcellation'].loc[global_df['Type']=='Replication'] = '/'
        global_df = global_df.drop_duplicates(subset = ['MDS-UPDRS Prediction target', 'Feature', 'Type', 'Best performing model'])

    return global_df

def plot_unity(xdata, ydata, **kwargs):
    mn = min(xdata.min(), ydata.min())
    mx = max(xdata.max(), ydata.max())
    points = np.linspace(mn, mx, 100)
    plt.gca().plot(points, points, color='r', marker=None,
            linestyle='--', linewidth=1.0)


def plot_pred_real(pipeline, global_df, specific=None):
    if not os.path.isdir(f'./outputs/{pipeline}/figures'):
        os.mkdir(f'./outputs/{pipeline}/figures')
    
    for timepoint in ['baseline', '1y', '2y', '4y']:
        fig = plt.figure(figsize=(10,4))
        i=0
        for feature in ['ReHo', 'falff']:
            i += 1
            rsquare = round(global_df['R2'].loc[(global_df['Type']=='Replication') &\
                                    (global_df['Feature']==feature_dict[feature]) &\
                                    (global_df['MDS-UPDRS Prediction target']==timepoint_dict[timepoint])].iloc[0],5)

            model = global_df['Best performing model'].loc[(global_df['Type']=='Replication') &\
                                    (global_df['Feature']==feature_dict[feature]) &\
                                    (global_df['MDS-UPDRS Prediction target']==timepoint_dict[timepoint])].iloc[0]
            model = list(model_dict.keys())[list(model_dict.values()).index(model)]

            atlas = global_df['Best performing parcellation'].loc[(global_df['Type']=='Replication') &\
                                    (global_df['Feature']==feature_dict[feature]) &\
                                    (global_df['MDS-UPDRS Prediction target']==timepoint_dict[timepoint])].iloc[0]
            if pipeline == 'no_imaging_features':
                atlas = 'Schaefer'
            atlas = list(atlas_dict.keys())[list(atlas_dict.values()).index(atlas)]

            if specific == None:
                df_pred_target = pd.read_csv(f'./outputs/{pipeline}/prediction_scores/'+\
                                    f'prediction-{timepoint}_feature-{feature}_test_results.csv')

            else: 
                df_pred_target = pd.read_csv(f'./outputs/{pipeline}/prediction_scores/'+\
                                    f'prediction-{timepoint}_feature-{feature}{specific}_test_results.csv')


            df_pred_target = df_pred_target.loc[(df_pred_target['model']==model) &\
                                                (df_pred_target['atlas']==atlas)]

            target = [float(t) for t in df_pred_target['Target'].tolist()[0][1:-1].split(',')]
            pred = [float(p) for p in df_pred_target['Prediction'].tolist()[0][1:-1].split(',')]
    
            df_comp = pd.DataFrame({f'True MDS-UPDRS score at {timepoint_dict[timepoint]}':target,
                                   f'Predicted MDS-UPDRS score at {timepoint_dict[timepoint]}':pred})
            
            if pipeline != 'no_imaging_features':
                ax = fig.add_subplot(1,2,i)
                ax.grid(False)
                sc = sns.scatterplot(data=df_comp, x=f'True MDS-UPDRS score at {timepoint_dict[timepoint]}',
                               y=f'Predicted MDS-UPDRS score at {timepoint_dict[timepoint]}', axes=ax)
                plot_unity(sc.axis()[0], sc.axis()[1])
                sc.text(1, sc.axis()[1].max()-10, f"$R^{2}$ = {rsquare}", fontstyle = "oblique")
                ax.set_title(f'Prediction of {timepoint_dict[timepoint]} severity from {feature_dict[feature]}')

        if pipeline == 'no_imaging_features':
            fig = plt.figure(figsize=(5,4))
            ax = fig.add_subplot(1,1,1)
            ax.grid(False)
            sc = sns.scatterplot(data=df_comp, x=f'True MDS-UPDRS score at {timepoint_dict[timepoint]}',
                           y=f'Predicted MDS-UPDRS score at {timepoint_dict[timepoint]}', axes=ax)
            plot_unity(sc.axis()[0], sc.axis()[1])
            sc.text(1, sc.axis()[1].max()-10, f"$R^{2}$ = {rsquare}", fontstyle = "oblique")
            ax.set_title(f'Prediction of {timepoint_dict[timepoint]} severity without imaging features')
            
        plt.tight_layout()
        if specific != None: 
            plt.savefig(f'./outputs/{pipeline}/figures/plot_pred-target_{timepoint}{specific}.png') 
        else: 
            plt.savefig(f'./outputs/{pipeline}/figures/plot_pred-target_{timepoint}.png') 