import numpy as np
import pandas as pd
from webapp import app,db,bcrypt

from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.preprocessing import StandardScaler
import joblib
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.utils import all_estimators
import os
# from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
# from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
import warnings
warnings.filterwarnings('ignore')

# Stage 1 – PD for LGD
class solver_stage1:
    def __init__(self,models):
        self.models_data = {}
        self.models = models

    def fit(self,X,y):
        for key in self.models.keys():
            if key == "xgb":
                X.columns = X.columns.str.translate("".maketrans({"[":"{", "]":"}","<":"^"}))
            elif key == "NN":
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                self.models[key].fit(X_scaled,y,epochs=8, verbose=1,batch_size=512)
                continue
            self.models[key].fit(X,y)

    def evaluate(self,lgd_inputs_stage_1_test,lgd_targets_stage_1_test,tr = 0.5):

        self.eval = {'support':[],'fscore':[],'Recall':[],'accuracy':[],'Precision':[],'auc':[],'model':[]}

        for key in self.models.keys():
            self.eval['model'].append(key)
            

            if key == "xgb":
                lgd_inputs_stage_1_test.columns = lgd_inputs_stage_1_test.columns.str.translate("".maketrans({"[":"{", "]":"}","<":"^"}))

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
    
    def load(self,solver_stage_num):
        for key in self.models.keys():
            self.models[key] = joblib.load(os.path.join(app.config['UPLOAD_FOLDER'][0], solver_stage_num, key+'.sav'))
    
class solver_stage2:
    
    def __init__(self,models):
        self.models_data = {}
        self.models = models

    def fit(self,X,y):
        for key in self.models.keys():
            if key == "xgb":
                X.columns = X.columns.str.translate("".maketrans({"[":"{", "]":"}","<":"^"}))
            elif key == "NN":
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                self.models[key].fit(X_scaled,y,epochs=3, verbose=1,batch_size=512)
                continue
            self.models[key].fit(X,y)

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

    def load(self,solver_stage_num):
        for key in self.models.keys():
            print(key)
            print(os.path.join(app.config['UPLOAD_FOLDER'][0], solver_stage_num, key+'.sav'))
            self.models[key] = joblib.load(os.path.join(app.config['UPLOAD_FOLDER'][0], solver_stage_num, key+'.sav'))

regs = ['Lars', 'LarsCV', 'Lasso', 'LassoCV', 'LassoLars', 'LassoLarsCV', 'LassoLarsIC', 'LinearRegression', 
            'MLPRegressor', 'NuSVR', 
            'PLSRegression', 'PassiveAggressiveRegressor', 
            'PoissonRegressor', 'RANSACRegressor', 'RandomForestRegressor', 'Ridge', 'RidgeCV', 
            'SGDRegressor', 'TheilSenRegressor', 'TransformedTargetRegressor', 'TweedieRegressor',
            'ARDRegression', 'AdaBoostRegressor', 'BaggingRegressor', 'BayesianRidge', 
            'DecisionTreeRegressor', 'DummyRegressor', 'ElasticNet', 'ElasticNetCV', 'ExtraTreeRegressor', 
            'ExtraTreesRegressor', 'GammaRegressor', 'GradientBoostingRegressor', 
            'HistGradientBoostingRegressor', 'HuberRegressor', 'KNeighborsRegressor', 
            ]

def get_all_regressors_sklearn(models):
    
    estimators = all_estimators(type_filter='regressor')

    for name, ClassifierClass in estimators:
      if name in regs:
          try:
            models[name] = ClassifierClass()
            # print('Appended', name)
          except Exception as e:
            print('Unable to import', name)
            print(e)
    return models

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

    # lgd_inputs_stage_1_train = lgd_inputs_stage_1_train[features_all]
    # lgd_inputs_stage_1_train = lgd_inputs_stage_1_train.drop(features_reference_cat, axis = 1)
    lgd_inputs_stage_1_test = lgd_inputs_stage_1_test[features_all]
    lgd_inputs_stage_1_test = lgd_inputs_stage_1_test.drop(features_reference_cat, axis = 1)
    lgd_inputs_stage_1_train.isnull().sum().sum()
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy',
                    optimizer=tf.keras.optimizers.Adam(), # use Adam instead of SGD
                    metrics=['accuracy'])
    
    reg_models = {}

    
    reg_models = get_all_regressors_sklearn(reg_models)

    models = {
        'NN' : model,
    }

    estimators = all_estimators(type_filter='classifier')

    all_clfs = []
    for name, ClassifierClass in estimators:
        if name in ['LogisticRegression', 'LogisticRegressionCV','BernoulliNB', 'CalibratedClassifierCV','ComplementNB', 'DecisionTreeClassifier', 'DummyClassifier', 'ExtraTreeClassifier', 'ExtraTreesClassifier', 'GaussianNB',
                        'GradientBoostingClassifier', 'HistGradientBoostingClassifier', 'LinearDiscriminantAnalysis','PassiveAggressiveClassifier',
                        'QuadraticDiscriminantAnalysis','RidgeClassifier', 'RidgeClassifierCV', 'SGDClassifier','RandomForestClassifier']:
            try:
                models[name] = ClassifierClass()
                # print('Appended', name)
            except Exception as e:
                print('Unable to import', name)
                print(e)
    
    models['xgb'] = XGBClassifier()

    sol1 = solver_stage1(models)
    # sol1.fit(lgd_inputs_stage_1_train, lgd_targets_stage_1_train)
    sol1 = sol1.load('solver_stage1')
    eval = sol1.evaluate(lgd_inputs_stage_1_test,lgd_targets_stage_1_test)
    
    print("2)Solver stage 1 Model evaluated data----------------------")

    lgd_stage_2_data = loan_data_defaults[loan_data_defaults['recovery_rate_0_1'] == 1]
    lgd_inputs_stage_2_train, lgd_inputs_stage_2_test, lgd_targets_stage_2_train, lgd_targets_stage_2_test = train_test_split(lgd_stage_2_data.drop(['good_bad', 'recovery_rate','recovery_rate_0_1', 'CCF'], axis = 1), lgd_stage_2_data['recovery_rate'], test_size = 0.2, random_state = 42)
    # lgd_inputs_stage_2_train = lgd_inputs_stage_2_train[features_all]
    # lgd_inputs_stage_2_train = lgd_inputs_stage_2_train.drop(features_reference_cat, axis = 1)
    lgd_inputs_stage_2_test = lgd_inputs_stage_2_test[features_all]
    lgd_inputs_stage_2_test = lgd_inputs_stage_2_test.drop(features_reference_cat, axis = 1)


    sol2 = solver_stage2(reg_models)
    # sol2.fit(lgd_inputs_stage_2_train, lgd_targets_stage_2_train)
    sol2 = sol2.load('solver_stage2')
    eval2 = sol2.evaluate(lgd_inputs_stage_2_test,lgd_targets_stage_2_test)
    print("2)Solver stage 2 Model evaluated data----------------------")


    details = eval
    df = pd.DataFrame(details)
    df = df.set_index('model')
    

    df = df.drop(columns=['support'])
    df = df.apply(avg,axis=1)
    df = df.apply(xscore,axis=1)

    # Heat map
    mo = 'NN'
    sol1.models_data[mo]['df_preds']
    cm = sol1.models_data[mo]['cm']

    # AUROC
    df_actual_predicted_probs = sol1.models_data['xgb']['df_preds']
    tr = 0.5

    df_actual_predicted_probs.columns = ['loan_data_targets_test', 'y_hat_test_proba','y_hat_test']

    df_actual_predicted_probs['y_hat_test'] = np.where(df_actual_predicted_probs['y_hat_test_proba'] > tr, 1, 0)

    from sklearn.metrics import roc_curve, roc_auc_score
    y_true, y_pred = 	df_actual_predicted_probs['loan_data_targets_test']	, df_actual_predicted_probs['y_hat_test']
    roc_curve(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test_proba'], pos_label=1)
    # Returns the Receiver Operating Characteristic (ROC) Curve from a set of actual values and their predicted probabilities.
    # As a result, we get three arrays: the false positive rates, the true positive rates, and the thresholds.

    fpr, tpr, thresholds = roc_curve(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test_proba'],pos_label=0)
    # Here we store each of the three arrays in a separate variable.

    # Gini

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

    # Smirnov
    # ...

    df = df.sort_values(by=['Xscore'], ascending=False)
    model_pd = df.index[0]
    best_stage1 = sol1.models_data[df.index[0]]['df_preds']['y_hat_test_lgd_stage_1']
    print(df)

    # Stage 2

    details = eval2
    df = pd.DataFrame(details)
    df = df.set_index('model')
    df = df.sort_values(by=['R2 score'], ascending=False)
    best_stage2 = sol2.models[df.index[0]].predict(lgd_inputs_stage_1_test)
    model_reg = df.index[0]
    LGD = best_stage1*best_stage2
    print(LGD)

    # EAD

    ead_inputs_train, ead_inputs_test, ead_targets_train, ead_targets_test = train_test_split(loan_data_defaults.drop(['good_bad', 'recovery_rate', 'recovery_rate_0_1', 'CCF'], axis = 1), loan_data_defaults['CCF'], test_size = 0.2, random_state = 42)
    # ead_inputs_train = ead_inputs_train[features_all]
    # ead_inputs_train = ead_inputs_train.drop(features_reference_cat, axis = 1)
    ead_inputs_test = ead_inputs_test[features_all]
    ead_inputs_test = ead_inputs_test.drop(features_reference_cat, axis = 1)

    sol3 = solver_stage2(reg_models)
    # sol3.fit(ead_inputs_train, ead_targets_train)
    sol3 = sol3.load('solver_stage3')
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

    # X_train = X_train[features_all]
    # X_train = X_train.drop(features_reference_cat, axis = 1)
    x_test = x_test[features_all]
    x_test = x_test.drop(features_reference_cat, axis = 1)

    sol4 = solver_stage1(models)
    # sol4.fit(X_train, Y_train)
    sol4 = sol4.load('solver_stage4')
    eval4 = sol4.evaluate( x_test, y_test)
    print("4)Solver stage 4 Model evaluated data----------------------")


    details = eval4
    df = pd.DataFrame(details)
    df = df.set_index('model')
    # col = ['fscore', 'Recall', 'accuracy', 'Precision', 'auc']
    # weights = [1, 1, 1, 1, 1]

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


    return exp_loss



