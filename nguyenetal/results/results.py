import pandas as pd
import pickle
import sys
import os
import glob
import re
import numpy as np
from sklearn import metrics, model_selection
import matplotlib.pyplot as plt
import seaborn as sns

def get_prediction_results(inputs, target, threshold, output_dir, model_list):
    results = []
    pred_list = []
    for model_name in model_list:
        try:
            with open(f'{output_dir}/{model_name}_results.pkl', 'rb') as f:
                model_dict = pickle.load(f)
                f.close()
        except FileNotFoundError:
            continue
            
        outer = model_selection.LeaveOneOut()
        
        pred = np.zeros(inputs.shape[0])
        rsquare_train = np.zeros_like(pred)
        rmse_train = np.zeros_like(pred)
        rsquare_valid = np.zeros_like(pred)
        rmse_valid = np.zeros_like(pred)

        for i, (train_idx, test_idx) in enumerate(outer.split(inputs, target)):
            inputs_test = inputs.iloc[test_idx]
            inputs_train = inputs.iloc[train_idx]
            model = model_dict['estimator'][i].best_estimator_
            pred[i] = model.predict(inputs_test.astype(np.float64))
            pred_train = model.predict(inputs_train.astype(np.float64))
            rsquare_train[i] = metrics.r2_score(target[train_idx], pred_train)
            rmse_train[i] = np.sqrt(metrics.mean_squared_error(target[train_idx], pred_train))

            search_results = pd.DataFrame(model_dict['estimator'][i].cv_results_)
            best_model = model_dict['estimator'][i].best_index_
            rsquare_valid[i] = search_results['mean_test_rsquare'].iloc[best_model]
            rmse_valid[i] = -search_results['mean_test_rmse'].iloc[best_model]

        true_class = target > threshold
        pred_class = pred > threshold
        trueneg, falsepos, falseneg, truepos = metrics.confusion_matrix(true_class,
                                                                        pred_class).ravel()

        results += [{'Model': model,
                        'Test R2' : metrics.r2_score(target, pred),
                        'Test RMSE': np.sqrt(metrics.mean_squared_error(target, pred)),
                        'Test AUC': metrics.roc_auc_score(true_class, pred_class),
                        'Test precision': metrics.precision_score(true_class, pred_class),
                        'Test recall': metrics.recall_score(true_class, pred_class),
                        'Test accuracy': metrics.accuracy_score(true_class, pred_class),
                        'Test f1': metrics.f1_score(true_class, pred_class),
                        'Test NPV': trueneg / (trueneg + falseneg),
                        'Test specificity': trueneg / (trueneg + falsepos),
                        'Val Mean R2': np.mean(rsquare_valid),
                        'Val Std R2': np.std(rsquare_valid),
                        'Val Mean RMSE': np.mean(rmse_valid),
                        'Val Std RMSE': np.std(rmse_valid),
                        'Train Mean R2': np.mean(rsquare_train),
                        'Train Std R2': np.std(rsquare_train),
                        'Train Mean RMSE': np.mean(rmse_train),
                        'Train Std RMSE': np.std(rmse_train)}]
        
        pred_list.append(pred)

    results_df = pd.DataFrame(results)
    return results_df, pred_list


def find_best_model(df):
    best_model = df[df['Val Mean R2']==df['Val Mean R2'].max()]
    
    return best_model

def merge_results(pipeline, timepoint, feature):
    df_feature_results = pd.DataFrame()
                
    for atlas in ['schaefer', 'basc197', 'basc444']: # Concat results of different atlases  
        outer = model_selection.LeaveOneOut()
        
        output_dir=f'./outputs/{pipeline}/prediction_scores/'+\
        f'predition-{timepoint}_atlas-{atlas}_feature-{feature}'

        df = pd.read_csv(f'{output_dir}/metrics.csv', index_col=0)

        df['Atlas']=[atlas for i in range(len(df))]

        df_feature_results = pd.concat([df_feature_results, df])

    return df_feature_results

timepoint_dict = {'baseline':'Baseline',
                 '1y': 'Year 1',
                 '2y': 'Year 2', 
                 '4y': 'Year 4'}

feature_dict = {'falff':'fALFF',
               'ReHo':'ReHo'}

atlas_dict = {'schaefer':'Schaefer', 
             'basc197':'BASC197',
             'basc444': 'BASC444'}

model_dict = {'ElasticNet':'ElasticNet',
             'LinearSVR': 'SVM',
             'GradientBoostingRegressor': 'Gradient Boosting',
             'RandomForestRegressor': 'Random Forest'}

def get_results_df(pipeline, original_df):

    global_df = pd.DataFrame(columns = ['MDS-UPDRS Prediction target','Feature','Type','Best performing model',
                                        'Best performing parcellation', 'R2', 'RMSE', 'AUC', 
                                        'PPV', 'NPV', 'Spec.','Sens.'])
        
    for timepoint in ['baseline', '1y', '2y', '4y']:
        
        for feature in ['falff', 'ReHo']:
            # Look at results for original paper for timepoint and feature
            sub_original_df = original_df[
            original_df['MDS-UPDRS Prediction target']==timepoint_dict[timepoint]
            ][
            original_df['Feature']==feature_dict[feature]
            ]
            
            global_df = pd.concat([global_df, sub_original_df]) # Add the results to the table 

            # Look at results computed for replication
            df_feature_results = merge_results(pipeline, timepoint, feature)
            
            # Check for best model across all atlases 
            best_model_df = find_best_model(df_feature_results) 

            # Process dataframe to obtain results 
            model = best_model_df['Model'].iloc[0].split('[')[1].split(',')[4].split('\'')[1]
            atlas = best_model_df['Atlas'].iloc[0]
            r_square = best_model_df['Test R2'].iloc[0]
            rmse = best_model_df['Test RMSE'].iloc[0]
            auc = best_model_df['Test AUC'].iloc[0]
            ppv = best_model_df['Test precision'].iloc[0]
            npv = best_model_df['Test NPV'].iloc[0]
            specificity = best_model_df['Test specificity'].iloc[0]
            sensitivity = best_model_df['Test recall'].iloc[0]
            
            sub_df = pd.DataFrame({
                'MDS-UPDRS Prediction target': [timepoint_dict[timepoint]],
                'Feature': [str(feature_dict[feature])],
                'Type': ['Replication'],
                'Best performing model': [model_dict[model]],
                'Best performing parcellation': [atlas_dict[atlas]],
                'R2': [round(r_square,5)],
                'RMSE': [round(rmse,3)],
                'AUC':[round(auc,3)],
                'PPV': [str(round(ppv*100, 1))+'%'],
                'NPV':[str(round(npv*100,1))+'%'],
                'Spec.': [str(round(specificity*100, 1))+'%'],
                'Sens.': [str(round(sensitivity*100, 1))+'%']
            })
    
            global_df = pd.concat([global_df, sub_df])

    #global_df = global_df.set_index(['MDS-UPDRS Prediction target', 'Feature','Type'])

    return global_df

def plot_unity(xdata, ydata, **kwargs):
    mn = min(xdata.min(), ydata.min())
    mx = max(xdata.max(), ydata.max())
    points = np.linspace(mn, mx, 100)
    plt.gca().plot(points, points, color='r', marker=None,
            linestyle='--', linewidth=1.0)

pipeline='fmriprep_pipeline'

def plot_pred_real(pipeline, global_df):
    if not os.path.isdir(f'./outputs/{pipeline}/figures'):
        os.mkdir(f'./outputs/{pipeline}/figures')
    
    for timepoint in ['baseline', '1y', '2y', '4y']:
        fig = plt.figure(figsize=(10,4))
        i=0
        for feature in ['ReHo', 'falff']:
            i += 1
            best_model = global_df['Best performing model'].loc[(global_df['Type']=='Replication') &\
                                    (global_df['Feature']==feature_dict[feature]) &\
                                    (global_df['MDS-UPDRS Prediction target']==timepoint_dict[timepoint])].iloc[0]
    
            best_model = list(model_dict.keys())[list(model_dict.values()).index(best_model)]
    
            best_atlas = global_df['Best performing parcellation'].loc[(global_df['Type']=='Replication') &\
                                    (global_df['Feature']==feature_dict[feature]) &\
                                    (global_df['MDS-UPDRS Prediction target']==timepoint_dict[timepoint])].iloc[0].lower()
    
            rsquare = round(global_df['R2'].loc[(global_df['Type']=='Replication') &\
                                    (global_df['Feature']==feature_dict[feature]) &\
                                    (global_df['MDS-UPDRS Prediction target']==timepoint_dict[timepoint])].iloc[0],5)
    
            output_dir=f'./outputs/{pipeline}/prediction_scores/'+\
            f'predition-{timepoint}_atlas-{best_atlas}_feature-{feature}'
    
            with open(f'{output_dir}/pred_{best_model}.txt', 'r') as f:
                pred = f.read().split(',')[:-1]
            pred = [float(p) for p in pred]
    
            df_outcome = pd.read_csv(f'{output_dir}/target.csv',
                                   header=0, index_col=None)
            
            target = df_outcome['UPDRS_TOT'].to_numpy(copy=True)
    
            df_comp = pd.DataFrame({f'True MDS-UPDRS score at {timepoint_dict[timepoint]}':target,
                                   f'Predicted MDS-UPDRS score at {timepoint_dict[timepoint]}':pred})
    
            ax = fig.add_subplot(1,2,i)
            ax.grid(False)
            sc = sns.scatterplot(data=df_comp, x=f'True MDS-UPDRS score at {timepoint_dict[timepoint]}',
                           y=f'Predicted MDS-UPDRS score at {timepoint_dict[timepoint]}', axes=ax)
            plot_unity(sc.axis()[0], sc.axis()[1])
            sc.text(1, sc.axis()[1].max()-10, f"$R^{2}$ = {rsquare}", fontstyle = "oblique")
            ax.set_title(f'Prediction of {timepoint_dict[timepoint]} severity from {feature_dict[feature]}')
            
        plt.tight_layout()
        plt.savefig(f'./outputs/{pipeline}/figures/plot_pred-target_{timepoint}.png')  

