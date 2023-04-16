import json
import numpy as np
import pandas as pd
from webapp import app,db,bcrypt

from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
# Use same sklearn version while saving and loading the model

# For loading NN keras model
import tensorflow.keras as keras

# For loading xgboost model
import pickle

import tensorflow as tf
import os
from sklearn.metrics import roc_curve, roc_auc_score

import seaborn as sns
import matplotlib.pyplot  as plt
import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')

# Stage 1 – PD for LGD
class solver_stage1:
    def __init__(self,models):
        self.models_data = {}
        self.models = models

    def evaluate(self,lgd_inputs_stage_1_test,lgd_targets_stage_1_test,tr = 0.5):

        self.eval = {'support':[],'fscore':[],'Recall':[],'accuracy':[],'Precision':[],'auc':[],'model':[]}

        for key in self.models.keys():
            self.eval['model'].append(key)

            if key == "xgb":
                lgd_inputs_stage_1_test.columns = lgd_inputs_stage_1_test.columns.str.translate("".maketrans({"[":"{", "]":"}","<":"^"}))

            # For NN, PassiveAggressiveClassifier, RidgeClassifier, RidgeClassifierCV, SGDClassifier it goes to Except
            try:
                y_hat_test_lgd_stage_1 = self.models[key].predict(lgd_inputs_stage_1_test)
                y_hat_test_proba_lgd_stage_1 = self.models[key].predict_proba(lgd_inputs_stage_1_test)
                y_hat_test_proba_lgd_stage_1 = y_hat_test_proba_lgd_stage_1[: ][: , 1]
            except:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(lgd_inputs_stage_1_test)
                y_hat_test_proba_lgd_stage_1 = self.models[key].predict(X_scaled)
                y_hat_test_lgd_stage_1 = self.models[key].predict(X_scaled)

            lgd_targets_stage_1_test_temp = lgd_targets_stage_1_test
            lgd_targets_stage_1_test_temp.reset_index(drop = True, inplace = True)
            df_actual_predicted_probs = pd.concat([lgd_targets_stage_1_test_temp, pd.DataFrame(y_hat_test_proba_lgd_stage_1)], axis = 1)
            df_actual_predicted_probs.columns = ['lgd_targets_stage_1_test', 'y_hat_test_proba_lgd_stage_1']
            df_actual_predicted_probs.index = lgd_inputs_stage_1_test.index
            
            df_actual_predicted_probs['y_hat_test_lgd_stage_1'] = np.where(df_actual_predicted_probs['y_hat_test_proba_lgd_stage_1'] > tr, 1, 0)
            yy , yypred = lgd_targets_stage_1_test,df_actual_predicted_probs['y_hat_test_lgd_stage_1']

            precision, recall, fscore, support = score(yy, yypred)
            self.eval['fscore'].append(fscore.tolist())
            self.eval['Recall'].append(recall.tolist())
            self.eval['Precision'].append(precision.tolist())
            self.eval['support'].append(support.tolist())
            self.eval['accuracy'].append(accuracy_score(yy, yypred).tolist())
            self.eval['auc'].append(roc_auc_score(df_actual_predicted_probs['lgd_targets_stage_1_test'], df_actual_predicted_probs['y_hat_test_lgd_stage_1']))

            self.models_data[key] = {'df_preds':df_actual_predicted_probs,'cm':confusion_matrix(yypred,yy)}
            print("eval",key)

        return self.eval  
    
class solver_stage2:
    
    def __init__(self,models):
        self.models_data = {}
        self.models = models

    def evaluate(self,X_test,y_true):
        
        self.eval = {'model':[],'R2 score':[],'MAE':[],'explained_variance_score':[],
                    'RMSE':[],'mean_squared_log_error':[],'median_absolute_error':[],
                    'mean_poisson_deviance':[],'mean_gamma_deviance':[],'d2_pinball_score':[],'d2_tweedie_score':[]}

        for key in self.models.keys():
            self.eval['model'].append(key)
            print('eval',key)
            y_pred = self.models[key].predict(X_test)
            y_pred = np.where(y_pred < 0, 0.000001, y_pred)
            y_pred = np.where(y_pred > 1, 1, y_pred)
            self.y_pred = y_pred
            self.eval['R2 score'].append(r2_score(y_true, y_pred))
            self.eval['MAE'].append(mean_absolute_error(y_true,y_pred))
            self.eval['explained_variance_score'].append(explained_variance_score(y_true, y_pred))
            self.eval['RMSE'].append(np.sqrt(mean_squared_error(y_true, y_pred)))
            self.eval['mean_squared_log_error'].append(mean_squared_log_error(y_true, y_pred, squared=False))
            self.eval['median_absolute_error'].append(median_absolute_error(y_true, y_pred))
            self.eval['mean_poisson_deviance'].append(mean_poisson_deviance(y_true, y_pred))
            self.eval['mean_gamma_deviance'].append(mean_gamma_deviance(y_true, y_pred))
            self.eval['d2_pinball_score'].append(d2_pinball_score(y_true, y_pred))
            self.eval['d2_tweedie_score'].append(d2_tweedie_score(y_true, y_pred))
            df = pd.DataFrame(columns = ['y_test', 'y_pred'])
            df['y_test'] , df['y_pred'] = y_true, y_pred
            self.models_data[key] = df
            
        return self.eval  

col = ['fscore', 'Recall', 'accuracy', 'Precision', 'auc']
weights = [1, 1, 1, 1, 1]

def xscore(row):
    row['Xscore'] = (sum(weights*row)/sum(weights))*100
    return row

def avg(row):
    for x in col:
        try:
            row[x] = sum(row[x])/len(row[x])
        except:
            continue
    return row

def expected_loss_func(csv_file_train, csv_file_test):
    print("Started-------------------------")
    # loan_data_preprocessed_train = pd.read_csv(csv_file_train)
    # Can write code for train data preprocessing and fitting.
    # ...

    loan_data_preprocessed_test = pd.read_csv(csv_file_test)
    # Code for test data preprocessing and prediction on test data
    loan_data_defaults = loan_data_preprocessed_test[loan_data_preprocessed_test['loan_status'].isin(['Charged Off','Does not meet the credit policy. Status:Charged Off'])]

    # fill the missing values with zeroes
    loan_data_defaults['mths_since_last_delinq'].fillna(0, inplace = True)
    loan_data_defaults['mths_since_last_record'].fillna(0, inplace=True)

    loan_data_defaults['recovery_rate'] = loan_data_defaults['recoveries'] / loan_data_defaults['funded_amnt']
    loan_data_defaults['recovery_rate'] = np.where(loan_data_defaults['recovery_rate'] > 1, 1, loan_data_defaults['recovery_rate'])
    loan_data_defaults['recovery_rate'] = np.where(loan_data_defaults['recovery_rate'] < 0, 0, loan_data_defaults['recovery_rate'])
    loan_data_defaults['CCF'] = (loan_data_defaults['funded_amnt'] - loan_data_defaults['total_rec_prncp']) / loan_data_defaults['funded_amnt']

    # Making all_csv_charts_data
    all_csv_charts_data={}

    grade_type_count=loan_data_defaults['grade'].value_counts()
    home_ownership_type_count = loan_data_defaults['home_ownership'].value_counts()
    verification_status_type_count = loan_data_defaults['verification_status'].value_counts()
    purpose_type_count = loan_data_defaults['purpose'].value_counts()
    initial_list_status_type_count = loan_data_defaults['initial_list_status'].value_counts()

    all_csv_charts_data = {
        'grade':{
            'type': grade_type_count.index.tolist(),
            'count': grade_type_count.tolist(),
        },
        'home_ownership':{
            'type': home_ownership_type_count.index.tolist(),
            'count': home_ownership_type_count.tolist(),
        },
        'verification_status':{
            'type': verification_status_type_count.index.tolist(),
            'count': verification_status_type_count.tolist(),
        },
        'purpose':{
            'type': purpose_type_count.index.tolist(),
            'count': purpose_type_count.tolist(),
        },
        'initial_list_status':{
            'type': initial_list_status_type_count.index.tolist(),
            'count': initial_list_status_type_count.tolist(),
        }
    }

    # all_charts_data is for storing assessment charts data
    all_charts_data={}
    all_tables_data={}

    # Chart1: CCF 
    all_charts_data['ccf_chart_data'] = loan_data_defaults['CCF'].tolist() # Set bins 25 in this graph
    # Chart 2: Recovery rate chart
    all_charts_data['recovery_rate_chart_data'] = loan_data_defaults['recovery_rate'].tolist() # Set bins 100 in this graph

    loan_data_defaults['recovery_rate_0_1'] = np.where(loan_data_defaults['recovery_rate'] == 0, 0, 1)

    # LGD  stage 1 -----------------------------------------------------------------------------------------------

    lgd_inputs_stage_1_train, lgd_inputs_stage_1_test, lgd_targets_stage_1_train, lgd_targets_stage_1_test = train_test_split(loan_data_defaults.drop(['good_bad', 'recovery_rate', 'recovery_rate_0_1', 'CCF'], axis = 1), loan_data_defaults['recovery_rate_0_1'], test_size = 0.2, random_state = 42)
    # lgd_inputs_stage_1_test = loan_data_defaults.drop(['good_bad', 'recovery_rate', 'recovery_rate_0_1', 'CCF'], axis = 1)
    # lgd_targets_stage_1_test = loan_data_defaults['recovery_rate_0_1']


    features_all = ['grade:A',
    'grade:B',
    'grade:C',
    'grade:D',
    'grade:E',
    'grade:F',
    'grade:G',
    'home_ownership:MORTGAGE',
    'home_ownership:NONE',
    'home_ownership:OTHER',
    'home_ownership:OWN',
    'home_ownership:RENT',
    'verification_status:Not Verified',
    'verification_status:Source Verified',
    'verification_status:Verified',
    'purpose:car',
    'purpose:credit_card',
    'purpose:debt_consolidation',
    'purpose:educational',
    'purpose:home_improvement',
    'purpose:house',
    'purpose:major_purchase',
    'purpose:medical',
    'purpose:moving',
    'purpose:other',
    'purpose:renewable_energy',
    'purpose:small_business',
    'purpose:vacation',
    'purpose:wedding',
    'initial_list_status:f',
    'initial_list_status:w',
    'term_int',
    'emp_length_int',
    'mths_since_issue_d',
    'mths_since_earliest_cr_line',
    'funded_amnt',
    'int_rate',
    'installment',
    'annual_inc',
    'dti',
    'delinq_2yrs',
    'inq_last_6mths',
    'mths_since_last_delinq',
    'mths_since_last_record',
    'open_acc',
    'pub_rec',
    'total_acc',
    'acc_now_delinq',
    'total_rev_hi_lim']

    features_reference_cat = ['grade:G',
    'home_ownership:RENT',
    'verification_status:Verified',
    'purpose:credit_card',
    'initial_list_status:f']

    lgd_inputs_stage_1_test = lgd_inputs_stage_1_test[features_all]
    lgd_inputs_stage_1_test = lgd_inputs_stage_1_test.drop(features_reference_cat, axis = 1)
    
    solver_stage1_models = {}
    solver_stage2_models = {}
    solver_stage3_models = {}
    solver_stage4_models = {}
    
    classifiers = ['BernoulliNB', 'CalibratedClassifierCV', 'ComplementNB', 'DecisionTreeClassifier', 'DummyClassifier', 'ExtraTreeClassifier', 'ExtraTreesClassifier', 'GaussianNB', 'GradientBoostingClassifier', 'HistGradientBoostingClassifier', 'LinearDiscriminantAnalysis', 'LogisticRegression', 'LogisticRegressionCV', 'NN', 'PassiveAggressiveClassifier', 'QuadraticDiscriminantAnalysis', 'RandomForestClassifier', 'RidgeClassifier', 'RidgeClassifierCV', 'SGDClassifier', 'xgb']
    regressors = ['ARDRegression', 'AdaBoostRegressor', 'BaggingRegressor', 'BayesianRidge', 'DecisionTreeRegressor', 'DummyRegressor', 'ElasticNet', 'ElasticNetCV', 'ExtraTreeRegressor', 'ExtraTreesRegressor', 'GammaRegressor', 'GradientBoostingRegressor', 'HistGradientBoostingRegressor', 'HuberRegressor', 'KNeighborsRegressor', 'Lars', 'LarsCV', 'Lasso', 'LassoCV', 'LassoLars', 'LassoLarsCV', 'LassoLarsIC', 'LinearRegression', 'MLPRegressor', 'NuSVR', 'PLSRegression', 'PassiveAggressiveRegressor', 'PoissonRegressor', 'RANSACRegressor', 'RandomForestRegressor', 'Ridge', 'RidgeCV', 'SGDRegressor', 'TheilSenRegressor', 'TransformedTargetRegressor', 'TweedieRegressor']

    for name in classifiers:
        if name == 'NN':
            solver_stage1_models[name] = keras.models.load_model(os.path.join(app.config['UPLOAD_FOLDER'][0], 'solver_stage1_models', name+'.h5'))
            solver_stage4_models[name] = keras.models.load_model(os.path.join(app.config['UPLOAD_FOLDER'][0], 'solver_stage4_models', name+'.h5'))
        else:
            solver_stage1_models[name] = pickle.load(open(os.path.join(app.config['UPLOAD_FOLDER'][0], 'solver_stage1_models', name+'.pkl'), 'rb'))
            solver_stage4_models[name] = pickle.load(open(os.path.join(app.config['UPLOAD_FOLDER'][0], 'solver_stage4_models', name+'.pkl'), 'rb'))

    for name in regressors:
        solver_stage2_models[name] = pickle.load(open(os.path.join(app.config['UPLOAD_FOLDER'][0], 'solver_stage2_models', name+'.pkl'), 'rb'))
        solver_stage3_models[name] = pickle.load(open(os.path.join(app.config['UPLOAD_FOLDER'][0], 'solver_stage3_models', name+'.pkl'), 'rb'))


    sol1 = solver_stage1(solver_stage1_models)
    eval1 = sol1.evaluate(lgd_inputs_stage_1_test,lgd_targets_stage_1_test)

    print("Solver stage 1 completed----------------------")

    details = eval1

    df = pd.DataFrame(details)
    df = df.set_index('model')
    df = df.drop(columns=['support'])
    df = df.apply(avg,axis=1)
    df = df.apply(xscore,axis=1)

    # Chart 3: LGD stage 1 Classifiers Models comparison
    
    # Not showing AUC
    all_charts_data['model_eval_chart_data']={}
    for index, row in df.iterrows():
        # all_charts_data['model_eval_chart_data'][index]=[row['fscore'], row['Recall'],row['accuracy'],row['Precision'],row['auc'],row['Xscore']/100]
        all_charts_data['model_eval_chart_data'][index]=[row['fscore'], row['Recall'],row['accuracy'],row['Precision'],row['auc'],row['Xscore']/100]

    # Chart 4: Heat map

    mo = 'NN'
    sol1.models_data[mo]['df_preds']
    cm = sol1.models_data[mo]['cm']
    all_charts_data['actual_predicted_chart_data'] = cm.tolist()

    # Chart 5: ROC 

    df_actual_predicted_probs = sol1.models_data['xgb']['df_preds']
    tr = 0.5

    df_actual_predicted_probs.columns = ['loan_data_targets_test', 'y_hat_test_proba','y_hat_test']

    df_actual_predicted_probs['y_hat_test'] = np.where(df_actual_predicted_probs['y_hat_test_proba'] > tr, 1, 0)

    y_true, y_pred = 	df_actual_predicted_probs['loan_data_targets_test']	, df_actual_predicted_probs['y_hat_test']

    fpr, tpr, thresholds = roc_curve(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test_proba'])
    # Here we store each of the three arrays in a separate variable.

    # Converting to list
    fpr_list = fpr.tolist()
    tpr_list = tpr.tolist()

    all_charts_data['auroc_chart_data'] =  {
        'line_plot': [fpr_list, tpr_list],
        'dash_plot': [fpr_list, fpr_list]
    }

    # Chart 6: Gini

    df_actual_predicted_probs = df_actual_predicted_probs.sort_values('y_hat_test_proba')
    df_actual_predicted_probs = df_actual_predicted_probs.reset_index()


    df_actual_predicted_probs['Cumulative N Population'] = df_actual_predicted_probs.index + 1
    # We calculate the cumulative number of all observations.
    # We use the new index for that. Since indexing in ython starts from 0, we add 1 to each index.
    df_actual_predicted_probs['Cumulative N Good'] = df_actual_predicted_probs['loan_data_targets_test'].cumsum()
    # We calculate cumulative number of 'good', which is the cumulative sum of the column with actual observations.
    df_actual_predicted_probs['Cumulative N Bad'] = df_actual_predicted_probs['Cumulative N Population'] - df_actual_predicted_probs['loan_data_targets_test'].cumsum()

    df_actual_predicted_probs['Cumulative Perc Population'] = df_actual_predicted_probs['Cumulative N Population'] / (df_actual_predicted_probs.shape[0])
    # We calculate the cumulative percentage of all observations.
    df_actual_predicted_probs['Cumulative Perc Good'] = df_actual_predicted_probs['Cumulative N Good'] / df_actual_predicted_probs['loan_data_targets_test'].sum()
    # We calculate cumulative percentage of 'good'.
    df_actual_predicted_probs['Cumulative Perc Bad'] = df_actual_predicted_probs['Cumulative N Bad'] / (df_actual_predicted_probs.shape[0] - df_actual_predicted_probs['loan_data_targets_test'].sum())
    # We calculate the cumulative percentage of 'bad'.

    # Converting pandas series to list
    cumulative_perc_population_list=df_actual_predicted_probs['Cumulative Perc Population'].tolist()
    cumulative_perc_bad_list = df_actual_predicted_probs['Cumulative Perc Bad'].tolist()

    all_charts_data['gini_chart_data'] = {
        'line_plot' : [cumulative_perc_population_list, cumulative_perc_bad_list],
        'dash_plot': [cumulative_perc_population_list, cumulative_perc_population_list]
    }

    # Chart 7: Smirnov
    
    # Getting in pandas series to list
    y_hat_test_proba_list = df_actual_predicted_probs['y_hat_test_proba'].tolist()
    cumulative_perc_good_list = df_actual_predicted_probs['Cumulative Perc Good'].tolist()

    all_charts_data['smirnov_chart_data'] = {
        'red_plot' : [y_hat_test_proba_list, cumulative_perc_bad_list],
        'blue_plot' : [y_hat_test_proba_list, cumulative_perc_good_list]
    }



    # Table 1: LGD stage 1 Classifier Models comparison

    df = df.sort_values(by=['Xscore'], ascending=False)
    model_pd = df.index[0]
    best_stage1 = sol1.models_data[df.index[0]]['df_preds']['y_hat_test_lgd_stage_1']

    all_tables_data['lgd_stage1_classifier_models_comparison']={}

    for index, row in df.iterrows():
        all_tables_data['lgd_stage1_classifier_models_comparison'][index]=[row['fscore'].round(2), row['Recall'].round(2), row['accuracy'].round(2),row['Precision'].round(2),row['auc'].round(2),row['Xscore'].round(2)]

    # all_tables_data['lgd_stage1_classifier_models_comparison'] = df.to_html()
    # all_tables_data['lgd_stage1_classifier_models_comparison'] = {
    #     'Model': df.index.tolist(),
    #     'Fscore':df['fscore'].round(2).tolist(),
    #     'Recall':df['Recall'].round(2).tolist(),
    #     'Accuracy':df['accuracy'].round(2).tolist(),
    #     'Precision':df['Precision'].round(2).tolist(),
    #     'AUC':df['auc'].round(2).tolist(),
    #     'Xscore':df['Xscore'].round(2).tolist(),
    # }

    # LGD Stage 2  -----------------------------------------------------------------------------------------------

    lgd_stage_2_data = loan_data_defaults[loan_data_defaults['recovery_rate_0_1'] == 1]
    lgd_inputs_stage_2_train, lgd_inputs_stage_2_test, lgd_targets_stage_2_train, lgd_targets_stage_2_test = train_test_split(lgd_stage_2_data.drop(['good_bad', 'recovery_rate','recovery_rate_0_1', 'CCF'], axis = 1), lgd_stage_2_data['recovery_rate'], test_size = 0.2, random_state = 42)
    #Is this line of code below represents the baove line of code?
    # lgd_inputs_stage_2_test = lgd_stage_2_data.drop(['good_bad', 'recovery_rate','recovery_rate_0_1', 'CCF'], axis = 1)
    # lgd_targets_stage_2_test = lgd_stage_2_data['recovery_rate']

    lgd_inputs_stage_2_test = lgd_inputs_stage_2_test[features_all]
    lgd_inputs_stage_2_test = lgd_inputs_stage_2_test.drop(features_reference_cat, axis = 1)

    sol2 = solver_stage2(solver_stage2_models)
    eval2 = sol2.evaluate(lgd_inputs_stage_2_test,lgd_targets_stage_2_test)

    print("Solver stage 2 completed----------------------")

    # Table 2: LGD stage 2 Regressor models comparison

    details = eval2
    df = pd.DataFrame(details)
    df = df.set_index('model')
    df = df.sort_values(by=['R2 score'], ascending=False)

    all_tables_data['lgd_stage2_regressor_models_comparison']={}
    for index, row in df.iterrows():
        all_tables_data['lgd_stage2_regressor_models_comparison'][index]=[row['R2 score'].round(2),row['MAE'].round(2),row['explained_variance_score'].round(2),row['RMSE'].round(2),row['mean_squared_log_error'].round(2),row['median_absolute_error'].round(2),row['mean_poisson_deviance'].round(2),row['mean_gamma_deviance'].round(2),row['d2_pinball_score'].round(2),row['d2_tweedie_score'].round(2)]
    
    # all_tables_data['lgd_stage2_regressor_models_comparison'] = df.to_html()

    # all_tables_data['lgd_stage2_regressor_models_comparison'] = {
    #     'Model':df.index.tolist(),
    #     'R2 score':df['R2 score'].round(2).tolist(),
    #     'MAE':df['MAE'].round(2).tolist(),
    #     'Explained variance score':df['explained_variance_score'].round(2).tolist(),
    #     'RMSE':df['RMSE'].round(2).tolist(),
    #     'Mean squared log error':df['mean_squared_log_error'].round(2).tolist(),
    #     'Median absolute error':df['median_absolute_error'].round(2).tolist(),
    #     'Mean poisson deviance':df['mean_poisson_deviance'].round(2).tolist(),
    #     'Mean gamma deviance':df['mean_gamma_deviance'].round(2).tolist(),
    #     'D2 pinball score':df['d2_pinball_score'].round(2).tolist(),
    #     'D2 tweedie score':df['d2_tweedie_score'].round(2).tolist()
    # }

    best_stage2 = sol2.models[df.index[0]].predict(lgd_inputs_stage_1_test)
    
    model_reg = df.index[0]
    LGD = best_stage1*best_stage2
    
    # EAD --------------------------------------------------------------------------------------------------

    ead_inputs_train, ead_inputs_test, ead_targets_train, ead_targets_test = train_test_split(loan_data_defaults.drop(['good_bad', 'recovery_rate', 'recovery_rate_0_1', 'CCF'], axis = 1), loan_data_defaults['CCF'], test_size = 0.2, random_state = 42)
    ead_inputs_test = ead_inputs_test[features_all]
    ead_inputs_test = ead_inputs_test.drop(features_reference_cat, axis = 1)

    sol3 = solver_stage2(solver_stage3_models)
    eval3 = sol3.evaluate(ead_inputs_test,ead_targets_test)
    print("Solver stage 3 completed----------------------")


    # Table 3: EAD Regressor models comparison

    details = eval3
    df = pd.DataFrame(details)
    df = df.set_index('model')
    df = df.sort_values(by=['R2 score'], ascending=False)
    model_ead = df.index[0]

    all_tables_data['ead_regressor_models_comparison']={}
    for index, row in df.iterrows():
        all_tables_data['ead_regressor_models_comparison'][index]=[row['R2 score'].round(2),row['MAE'].round(2),row['explained_variance_score'].round(2),row['RMSE'].round(2),row['mean_squared_log_error'].round(2),row['median_absolute_error'].round(2),row['mean_poisson_deviance'].round(2),row['mean_gamma_deviance'].round(2),row['d2_pinball_score'].round(2),row['d2_tweedie_score'].round(2)]
    
    # all_tables_data['ead_regressor_models_comparison'] = df.to_html()
    
    # all_tables_data['ead_regressor_models_comparison'] = {
    #     'Model':df.index.tolist(),
    #     'R2 score':df['R2 score'].round(2).tolist(),
    #     'MAE':df['MAE'].round(2).tolist(),
    #     'Explained variance score':df['explained_variance_score'].round(2).tolist(),
    #     'RMSE':df['RMSE'].round(2).tolist(),
    #     'Mean squared log error':df['mean_squared_log_error'].round(2).tolist(),
    #     'Median absolute error':df['median_absolute_error'].round(2).tolist(),
    #     'Mean poisson deviance':df['mean_poisson_deviance'].round(2).tolist(),
    #     'Mean gamma deviance':df['mean_gamma_deviance'].round(2).tolist(),
    #     'D2 pinball score':df['d2_pinball_score'].round(2).tolist(),
    #     'D2 tweedie score':df['d2_tweedie_score'].round(2).tolist()
    # }

    # Expected Loss ------------------------------------------------------------------------------------------

    loan_data_preprocessed_test['mths_since_last_delinq'].fillna(0, inplace = True)
    loan_data_preprocessed_test['mths_since_last_record'].fillna(0, inplace = True)
    loan_data_preprocessed_lgd_ead = loan_data_preprocessed_test[features_all]
    loan_data_preprocessed_lgd_ead = loan_data_preprocessed_lgd_ead.drop(features_reference_cat, axis = 1)
    loan_data_preprocessed_test['recovery_rate_st_1'] = sol1.models[model_pd].predict(loan_data_preprocessed_lgd_ead)
    loan_data_preprocessed_test['recovery_rate_st_2'] = sol2.models[model_reg].predict(loan_data_preprocessed_lgd_ead)
    loan_data_preprocessed_test['recovery_rate'] = loan_data_preprocessed_test['recovery_rate_st_1'] * loan_data_preprocessed_test['recovery_rate_st_2']
    loan_data_preprocessed_test['recovery_rate'] = np.where(loan_data_preprocessed_test['recovery_rate'] < 0, 0, loan_data_preprocessed_test['recovery_rate'])
    loan_data_preprocessed_test['recovery_rate'] = np.where(loan_data_preprocessed_test['recovery_rate'] > 1, 1, loan_data_preprocessed_test['recovery_rate'])
    loan_data_preprocessed_test['LGD'] = 1 - loan_data_preprocessed_test['recovery_rate']
    loan_data_preprocessed_test['CCF'] = sol3.models[model_ead].predict(loan_data_preprocessed_lgd_ead)
    loan_data_preprocessed_test['CCF'] = np.where(loan_data_preprocessed_test['CCF'] < 0, 0, loan_data_preprocessed_test['CCF'])
    loan_data_preprocessed_test['CCF'] = np.where(loan_data_preprocessed_test['CCF'] > 1, 1, loan_data_preprocessed_test['CCF'])
    loan_data_preprocessed_test['EAD'] = loan_data_preprocessed_test['CCF'] * loan_data_preprocessed_lgd_ead['funded_amnt']

    X_train, x_test, Y_train, y_test = train_test_split(loan_data_preprocessed_test.drop(['good_bad'], axis = 1), loan_data_preprocessed_test['good_bad'], test_size = 0.2, random_state = 42)

    x_test = x_test[features_all]
    x_test = x_test.drop(features_reference_cat, axis = 1)

    sol4 = solver_stage1(solver_stage4_models)
    eval4 = sol4.evaluate( x_test, y_test)
    print("Solver stage 4 completed----------------------")


    # Table 4: PD classifier models comparison

    details = eval4
    df = pd.DataFrame(details)
    df = df.set_index('model')

    df = df.drop(columns=['support'])
    df = df.apply(avg,axis=1)
    df = df.apply(xscore,axis=1)
    df = df.sort_values(by=['Xscore'], ascending=False)
    model_pd_EL = df.index[0]

    all_tables_data['pd_classifier_models_comparison']={}
    for index, row in df.iterrows():
        all_tables_data['pd_classifier_models_comparison'][index]=[row['fscore'].round(2), row['Recall'].round(2), row['accuracy'].round(2),row['Precision'].round(2),row['auc'].round(2),row['Xscore'].round(2)]
    
    # all_tables_data['pd_classifier_models_comparison'] = df.to_html()
    # all_tables_data['pd_classifier_models_comparison'] = {
    #     'Model': df.index.tolist(),
    #     'Fscore':df['fscore'].round(2).tolist(),
    #     'Recall':df['Recall'].round(2).tolist(),
    #     'Accuracy':df['accuracy'].round(2).tolist(),
    #     'Precision':df['Precision'].round(2).tolist(),
    #     'AUC':df['auc'].round(2).tolist(),
    #     'Xscore':df['Xscore'].round(2).tolist(),
    # }

    last = loan_data_preprocessed_test
    last = last[features_all]
    last = last.drop(features_reference_cat, axis = 1)


    if model_pd_EL == 'NN':
        loan_data_preprocessed_test['PD'] = sol4.models[model_pd_EL].predict(last)
    else:
        loan_data_preprocessed_test['PD'] = sol4.models[model_pd_EL].predict_proba(last)[: ][: , 0]


    # Table 5: Funded amount, PD, LGD, EAD, EL of all accounts 
    loan_data_preprocessed_test['EL'] = loan_data_preprocessed_test['PD'] * loan_data_preprocessed_test['LGD'] * loan_data_preprocessed_test['EAD']
    # all_tables_data['amount_PD_LGD_EAD_EL']=loan_data_preprocessed_test[['funded_amnt', 'PD', 'LGD', 'EAD', 'EL']].to_html()
    
    print(loan_data_preprocessed_test.head(10))

    exp_loss= loan_data_preprocessed_test['EL'].sum()
    fund_amt=loan_data_preprocessed_test['funded_amnt'].sum()
    exp_loss_perc=((loan_data_preprocessed_test['EL'].sum() / loan_data_preprocessed_test['funded_amnt'].sum())*98)

    return exp_loss, fund_amt, exp_loss_perc, all_charts_data, all_tables_data, all_csv_charts_data



