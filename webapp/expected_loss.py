import json
import numpy as np
import pandas as pd
from webapp import app,db,bcrypt

from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.preprocessing import StandardScaler

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
            self.eval['RMSE'].append(np.sqrt(mean_squared_error(y_true, y_pred)))
            self.eval['explained_variance_score'].append(explained_variance_score(y_true, y_pred))
            self.eval['mean_squared_log_error'].append(mean_squared_log_error(y_true, y_pred, squared=False))
            self.eval['mean_poisson_deviance'].append(mean_poisson_deviance(y_true, y_pred))
            self.eval['mean_gamma_deviance'].append(mean_gamma_deviance(y_true, y_pred))
            self.eval['median_absolute_error'].append(median_absolute_error(y_true, y_pred))
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

def expected_loss_func(csv_file):
    print("1)Started----------------------")
    loan_data_preprocessed = pd.read_csv(csv_file)

    loan_data_defaults = loan_data_preprocessed[loan_data_preprocessed['loan_status'].isin(['Charged Off','Does not meet the credit policy. Status:Charged Off'])]
    # fill the missing values with zeroes
    loan_data_defaults['mths_since_last_delinq'].fillna(0, inplace = True)
    loan_data_defaults['mths_since_last_record'].fillna(0, inplace=True)

    loan_data_defaults['recovery_rate'] = loan_data_defaults['recoveries'] / loan_data_defaults['funded_amnt']
    loan_data_defaults['recovery_rate'] = np.where(loan_data_defaults['recovery_rate'] > 1, 1, loan_data_defaults['recovery_rate'])
    loan_data_defaults['recovery_rate'] = np.where(loan_data_defaults['recovery_rate'] < 0, 0, loan_data_defaults['recovery_rate'])
    loan_data_defaults['CCF'] = (loan_data_defaults['funded_amnt'] - loan_data_defaults['total_rec_prncp']) / loan_data_defaults['funded_amnt']

    all_charts_data={}

    # CCF and Recovery rate chart -----------------------------------------------------------------------------------------------------
    all_charts_data['ccf_chart_data'] = loan_data_defaults['CCF'].tolist()
    # Set bins 25 in above graph
    all_charts_data['recovery_rate_chart_data'] = loan_data_defaults['recovery_rate'].tolist()
    # Set bins 100 in above graph

    loan_data_defaults['recovery_rate_0_1'] = np.where(loan_data_defaults['recovery_rate'] == 0, 0, 1)

    lgd_inputs_stage_1_train, lgd_inputs_stage_1_test, lgd_targets_stage_1_train, lgd_targets_stage_1_test = train_test_split(loan_data_defaults.drop(['good_bad', 'recovery_rate', 'recovery_rate_0_1', 'CCF'], axis = 1), loan_data_defaults['recovery_rate_0_1'], test_size = 0.2, random_state = 42)

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
    
    classifier_models = {}
    reg_models = {}
    
    # The comment ones are giving different outputs when loaded compared to when fitted.
    # NN
    classifiers = ['NN', 'BernoulliNB', 'CalibratedClassifierCV', 'ComplementNB', 'DecisionTreeClassifier', 'DummyClassifier', 'ExtraTreeClassifier', 'ExtraTreesClassifier', 'GaussianNB', 'GradientBoostingClassifier', 'HistGradientBoostingClassifier', 'LinearDiscriminantAnalysis', 'LogisticRegression', 'LogisticRegressionCV', 'PassiveAggressiveClassifier', 'QuadraticDiscriminantAnalysis', 'RandomForestClassifier', 'RidgeClassifier', 'RidgeClassifierCV', 'SGDClassifier', 'xgb']
    # AdaBoostRegressor, BaggingRegressor, TheilSenRegressor
    regressors = ['ARDRegression', 'AdaBoostRegressor', 'BaggingRegressor', 'BayesianRidge', 'DecisionTreeRegressor', 'DummyRegressor', 'ElasticNet', 'ElasticNetCV', 'ExtraTreeRegressor', 'ExtraTreesRegressor', 'GammaRegressor', 'GradientBoostingRegressor', 'HistGradientBoostingRegressor', 'HuberRegressor', 'KNeighborsRegressor', 'Lars', 'LarsCV', 'Lasso', 'LassoCV', 'LassoLars', 'LassoLarsCV', 'LassoLarsIC', 'LinearRegression', 'MLPRegressor', 'NuSVR', 'PLSRegression', 'PassiveAggressiveRegressor', 'PoissonRegressor', 'RANSACRegressor', 'RandomForestRegressor', 'Ridge', 'RidgeCV', 'SGDRegressor', 'TheilSenRegressor', 'TransformedTargetRegressor', 'TweedieRegressor']
    
    for name in classifiers:
        if name == 'NN':
            classifier_models[name] = keras.models.load_model(os.path.join(app.config['UPLOAD_FOLDER'][0], 'classifiers', name+'.h5'))
        else:
            classifier_models[name] = pickle.load(open(os.path.join(app.config['UPLOAD_FOLDER'][0], 'classifiers', name+'.pkl'), 'rb'))

    for name in regressors:
        reg_models[name] = pickle.load(open(os.path.join(app.config['UPLOAD_FOLDER'][0], 'regressors', name+'.pkl'), 'rb'))


    sol1 = solver_stage1(classifier_models)
    eval1 = sol1.evaluate(lgd_inputs_stage_1_test,lgd_targets_stage_1_test)

    print("2)Solver stage 1 Model evaluated data----------------------")

    lgd_stage_2_data = loan_data_defaults[loan_data_defaults['recovery_rate_0_1'] == 1]
    lgd_inputs_stage_2_train, lgd_inputs_stage_2_test, lgd_targets_stage_2_train, lgd_targets_stage_2_test = train_test_split(lgd_stage_2_data.drop(['good_bad', 'recovery_rate','recovery_rate_0_1', 'CCF'], axis = 1), lgd_stage_2_data['recovery_rate'], test_size = 0.2, random_state = 42)
    
    lgd_inputs_stage_2_test = lgd_inputs_stage_2_test[features_all]
    lgd_inputs_stage_2_test = lgd_inputs_stage_2_test.drop(features_reference_cat, axis = 1)


    sol2 = solver_stage2(reg_models)
    eval2 = sol2.evaluate(lgd_inputs_stage_2_test,lgd_targets_stage_2_test)

    print("2)Solver stage 2 Model evaluated data----------------------")

    details = eval1

    df = pd.DataFrame(details)
    df = df.set_index('model')
    df = df.drop(columns=['support'])
    df = df.apply(avg,axis=1)
    df = df.apply(xscore,axis=1)

    # Chart 1: Model evaluation------------------------------------------------------------------------------------------------------
    # all_charts_data['model_eval_chart_data'] = df.to_json(orient='split')
    # all_charts_data['model_eval_chart_data'] = df.to_json(orient='records')
    print("Model evaluation------------------------------")
    all_charts_data['model_eval_chart_data']={}
    for index, row in df.iterrows():
        # print(index, row['fscore'], row['Recall'], row['accuracy'], row['Precision'], row['auc'], row['Xscore'])
        all_charts_data['model_eval_chart_data'][index]=[row['fscore'],row['accuracy'],row['Precision'],row['auc'],row['Xscore']/100]

    # Chart 2: Heat map------------------------------------------------------------------------------------------------------
    mo = 'NN'
    sol1.models_data[mo]['df_preds']
    cm = sol1.models_data[mo]['cm']
    all_charts_data['actual_predicted_chart_data'] = cm.tolist()

    # ROC ------------------------------------------------------------------------------------------------------
    df_actual_predicted_probs = sol1.models_data['xgb']['df_preds']
    tr = 0.5

    df_actual_predicted_probs.columns = ['loan_data_targets_test', 'y_hat_test_proba','y_hat_test']

    df_actual_predicted_probs['y_hat_test'] = np.where(df_actual_predicted_probs['y_hat_test_proba'] > tr, 1, 0)

    y_true, y_pred = 	df_actual_predicted_probs['loan_data_targets_test']	, df_actual_predicted_probs['y_hat_test']

    fpr, tpr, thresholds = roc_curve(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test_proba'],pos_label=0)
    # Here we store each of the three arrays in a separate variable.

    # Getting in JSON format
    fpr_list = fpr.tolist()
    tpr_list = tpr.tolist()

    all_charts_data['auroc_chart_data'] =  {
        'line_plot': [fpr_list, tpr_list],
        'dash_plot': [fpr_list, fpr_list]
    }

    # Gini--------------------------------------------------------------------------------------------------------------------------------

    df_actual_predicted_probs = df_actual_predicted_probs.sort_values('y_hat_test_proba')
    df_actual_predicted_probs['loan_data_targets_test'] = pd.to_numeric(df_actual_predicted_probs['loan_data_targets_test'])

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

    # Getting in JSON format
    print("11111----------------------------------------------------------")
    print(type(df_actual_predicted_probs['Cumulative Perc Population']))
    print("----------------------------------------------------------")
    print(type(df_actual_predicted_probs['Cumulative Perc Bad']))
    cumulative_perc_population_list=df_actual_predicted_probs['Cumulative Perc Population'].tolist()
    cumulative_perc_bad_list = df_actual_predicted_probs['Cumulative Perc Bad'].tolist()

    all_charts_data['gini_chart_data'] = {
        'line_plot' : [cumulative_perc_population_list, cumulative_perc_bad_list],
        'dash_plot': [cumulative_perc_population_list, cumulative_perc_population_list]
    }

    # Smirnov------------------------------------------------------------------------------------------------------
    
    # Getting in JSON format
    print("22222----------------------------------------------------------")
    print(type(df_actual_predicted_probs['y_hat_test_proba']))
    print("----------------------------------------------------------")
    print(type(df_actual_predicted_probs['Cumulative Perc Good']))
    y_hat_test_proba_list = df_actual_predicted_probs['y_hat_test_proba'].tolist()
    cumulative_perc_good_list = df_actual_predicted_probs['Cumulative Perc Good'].tolist()

    all_charts_data['smirnov_chart_data'] = {
        'red_plot' : [y_hat_test_proba_list, cumulative_perc_bad_list],
        'blue_plot' : [y_hat_test_proba_list, cumulative_perc_good_list]
    }

    df = df.sort_values(by=['Xscore'], ascending=False)
    model_pd = df.index[0]
    best_stage1 = sol1.models_data[df.index[0]]['df_preds']['y_hat_test_lgd_stage_1']
    print('best_stage1------------------------------------')
    print(type(best_stage1))
    print(best_stage1)
    # print(df)

    # Stage 2

    details = eval2
    df = pd.DataFrame(details)
    df = df.set_index('model')
    df = df.sort_values(by=['R2 score'], ascending=False)
    print("df----------------------------------------------------")
    print(df)
    best_stage2 = sol2.models[df.index[0]].predict(lgd_inputs_stage_1_test)
    print('best_stage2------------------------------------')
    print(type(best_stage2))
    print(best_stage2)
    model_reg = df.index[0]
    LGD = best_stage1*best_stage2
    print(LGD)

    # EAD

    ead_inputs_train, ead_inputs_test, ead_targets_train, ead_targets_test = train_test_split(loan_data_defaults.drop(['good_bad', 'recovery_rate', 'recovery_rate_0_1', 'CCF'], axis = 1), loan_data_defaults['CCF'], test_size = 0.2, random_state = 42)
    ead_inputs_test = ead_inputs_test[features_all]
    ead_inputs_test = ead_inputs_test.drop(features_reference_cat, axis = 1)

    sol3 = solver_stage2(reg_models)
    eval3 = sol3.evaluate(ead_inputs_test,ead_targets_test)
    print("3)Solver stage 3 Model evaluated data----------------------")


    details = eval3
    df = pd.DataFrame(details)
    df = df.set_index('model')
    df = df.sort_values(by=['R2 score'], ascending=False)
    model_ead = df.index[0]

    print(df.head())

    # Expected Loss
    loan_data_preprocessed['mths_since_last_delinq'].fillna(0, inplace = True)
    loan_data_preprocessed['mths_since_last_record'].fillna(0, inplace = True)
    loan_data_preprocessed_lgd_ead = loan_data_preprocessed[features_all]
    loan_data_preprocessed_lgd_ead = loan_data_preprocessed_lgd_ead.drop(features_reference_cat, axis = 1)
    loan_data_preprocessed['recovery_rate_st_1'] = sol1.models[model_pd].predict(loan_data_preprocessed_lgd_ead)
    loan_data_preprocessed['recovery_rate_st_2'] = sol2.models[model_reg].predict(loan_data_preprocessed_lgd_ead)
    loan_data_preprocessed['recovery_rate'] = loan_data_preprocessed['recovery_rate_st_1'] * loan_data_preprocessed['recovery_rate_st_2']
    loan_data_preprocessed['recovery_rate'] = np.where(loan_data_preprocessed['recovery_rate'] < 0, 0, loan_data_preprocessed['recovery_rate'])
    loan_data_preprocessed['recovery_rate'] = np.where(loan_data_preprocessed['recovery_rate'] > 1, 1, loan_data_preprocessed['recovery_rate'])
    loan_data_preprocessed['LGD'] = 1 - loan_data_preprocessed['recovery_rate']
    loan_data_preprocessed['CCF'] = sol3.models[model_ead].predict(loan_data_preprocessed_lgd_ead)
    loan_data_preprocessed['CCF'] = np.where(loan_data_preprocessed['CCF'] < 0, 0, loan_data_preprocessed['CCF'])
    loan_data_preprocessed['CCF'] = np.where(loan_data_preprocessed['CCF'] > 1, 1, loan_data_preprocessed['CCF'])
    loan_data_preprocessed['EAD'] = loan_data_preprocessed['CCF'] * loan_data_preprocessed_lgd_ead['funded_amnt']

    X_train, x_test, Y_train, y_test = train_test_split(loan_data_preprocessed.drop(['good_bad'], axis = 1), loan_data_preprocessed['good_bad'], test_size = 0.2, random_state = 42)

    x_test = x_test[features_all]
    x_test = x_test.drop(features_reference_cat, axis = 1)

    sol4 = solver_stage1(classifier_models)
    eval4 = sol4.evaluate( x_test, y_test)
    print("4)Solver stage 4 Model evaluated data----------------------")

    details = eval4
    df = pd.DataFrame(details)
    df = df.set_index('model')

    df = df.drop(columns=['support'])
    df = df.apply(avg,axis=1)
    df = df.apply(xscore,axis=1)
    df = df.sort_values(by=['Xscore'], ascending=False)
    model_pd_EL = df.index[0]

    last = loan_data_preprocessed
    last = last[features_all]
    last = last.drop(features_reference_cat, axis = 1)


    if model_pd_EL == 'NN':
        loan_data_preprocessed['PD'] = sol4.models[model_pd_EL].predict(last)
    else:
        loan_data_preprocessed['PD'] = sol4.models[model_pd_EL].predict_proba(last)[: ][: , 0]

    loan_data_preprocessed['EL'] = loan_data_preprocessed['PD'] * loan_data_preprocessed['LGD'] * loan_data_preprocessed['EAD']
    print(loan_data_preprocessed[['funded_amnt', 'PD', 'LGD', 'EAD', 'EL']].head())

    exp_loss= loan_data_preprocessed['EL'].sum()
    fund_amt=loan_data_preprocessed['funded_amnt'].sum()
    exp_loss_perc=loan_data_preprocessed['EL'].sum() / loan_data_preprocessed['funded_amnt'].sum()*100
    print(f"5)Expected loss----------------------{exp_loss}")


    return exp_loss, all_charts_data



