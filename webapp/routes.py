import json
from webapp import app,db,bcrypt
from flask import redirect, render_template, url_for,flash, request
from webapp.models import User, CreditDetails
from webapp.forms import SignUpForm,LoginForm, CreditDetailsForm, CsvUploadForm
from flask_login import current_user, login_required, login_user, logout_user
from .expected_loss import expected_loss_func
from flask import Flask, send_file
import pickle

import io
# import pickle
import joblib
import numpy as np
import pandas as pd
import os
# from sklearn.linear_model import LogisticRegression
# from sklearn import metrics
# from sklearn import linear_model
# import scipy.stats as stat
from webapp.global_constants import grade, home_ownership, verification_status, purpose, initial_list_status
# addr_state, term, emp_length, mths_since_issue_d, int_rate, mths_since_earliest_cr_line, inq_last_6mths, acc_now_delinq, annual_inc, dti, mths_since_last_delinq, mths_since_last_record

# from urllib import request
# from sklearn.linear_model import LogisticRegression

@app.route("/")
def home():
    return render_template('home.html')

@app.route('/signup',methods=['GET','POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form=SignUpForm()
    if form.validate_on_submit():
        user=User(
            role='User',
            email=form.email.data,
            first_name=form.first_name.data,
            last_name=form.last_name.data,
            password=bcrypt.generate_password_hash(form.password1.data).decode('utf-8'),            
        )
        db.session.add(user)
        db.session.commit()
        login_user(user)
        flash('Registered Successfully.You have logged into the website',category='successful')
        return redirect(url_for('home'))
    # Forms will handle the validations. So if it's not validated, the errors will be stored in form.errors and will be passed to the signup.html
    return render_template('signup.html',form=form)

@app.route('/login',methods=['GET','POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form=LoginForm()
    if form.validate_on_submit():
        user=User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password,form.password.data):
            login_user(user)
            flash('You have logged in successfully',category='successful')
            return redirect(url_for('home'))
        else:
            flash('Login unsuccessful. Please check email and password',category='danger')
    return render_template('login.html',form=form)

@app.route("/logout",methods=['POST'])
@login_required
def logout():
    if request.method == 'POST':
        logout_user()
        flash('You have logged out successfully',category='successful')
        return redirect(url_for('home'))

# User routes

@app.route("/user_dashboard_home")
@login_required
def user_dashboard_home():
    credit_score=0
    prob_of_default=0

    credit_details=CreditDetails.query.filter_by(user_id=current_user.user_id).first()

    # Let's test the loaded model on a dummy user credit details. This is the input.

    if credit_details:
        x_test = pd.DataFrame({
        'grade': [credit_details.grade],
        'home_ownership': [credit_details.home_ownership],
        'verification_status': [credit_details.verification_status],
        'purpose': [credit_details.purpose],
        'initial_list_status': [credit_details.initial_list_status],
        })
        # 'addr_state': [credit_details.addr_state],


        # Creating a dictionary containing all the columns with their values according to credit details

        x_test={}
        for i in grade:
            x_test["grade:"+i] = [1] if credit_details.grade==i else [0]
        for i in home_ownership:
            x_test["home_ownership:"+i] = [1] if credit_details.home_ownership==i else [0]
        # for i in addr_state:
        #     x_test["addr_state:"+i] = [1] if credit_details.addr_state==i else [0]
        for i in verification_status:
            x_test["verification_status:"+i] = [1] if credit_details.verification_status==i else [0]
        for i in purpose:
            x_test["purpose:"+i] = [1] if credit_details.purpose==i else [0]
        for i in initial_list_status:
            x_test["initial_list_status:"+i] = [1] if credit_details.initial_list_status==i else [0]
        # for i in term:
        #     x_test["term:"+i] = [1] if credit_details.term==i else [0]
        # for i in emp_length:
        #     x_test["emp_length:"+i] = [1] if credit_details.emp_length==i else [0]
        # for i in mths_since_issue_d:
        #     x_test["mths_since_issue_d:"+i] = [1] if credit_details.mths_since_issue_d==i else [0]
        # for i in int_rate:
        #     x_test["int_rate:"+i] = [1] if credit_details.int_rate==i else [0]
        # for i in mths_since_earliest_cr_line:
        #     x_test["mths_since_earliest_cr_line:"+i] = [1] if credit_details.mths_since_earliest_cr_line==i else [0]
        # for i in inq_last_6mths:
        #     x_test["inq_last_6mths:"+i] = [1] if credit_details.inq_last_6mths==i else [0]
        # for i in acc_now_delinq:
        #     x_test["acc_now_delinq:"+i] = [1] if credit_details.acc_now_delinq==i else [0]
        # for i in annual_inc:
        #     x_test["annual_inc:"+i] = [1] if credit_details.annual_inc==i else [0]
        # for i in dti:
        #     x_test["dti:"+i] = [1] if credit_details.dti==i else [0]
        # for i in mths_since_last_delinq:
        #     x_test["mths_since_last_delinq:"+i] = [1] if credit_details.mths_since_last_delinq==i else [0]
        # for i in mths_since_last_record:
        #     x_test["mths_since_last_record:"+i] = [1] if credit_details.mths_since_last_record==i else [0]
        # print(x_test)

        # Saving ref_categories in a variable and will use later
        # ref_categories = ['grade:G',
        # 'home_ownership:RENT_OTHER_NONE_ANY',
        # 'addr_state:ND_NE_IA_NV_FL_HI_AL',
        # 'verification_status:Verified',
        # 'purpose:educ__sm_b__wedd__ren_en__mov__house',
        # 'initial_list_status:f',
        # 'term:60',
        # 'emp_length:0',
        # 'mths_since_issue_d:>84',
        # 'int_rate:>20.281',
        # 'mths_since_earliest_cr_line:<140',
        # 'inq_last_6mths:>6',
        # 'acc_now_delinq:0',
        # 'annual_inc:<20K',
        # 'dti:>35',
        # 'mths_since_last_delinq:0-3',
        # 'mths_since_last_record:0-2']

        x_test.update({
        'term_int': [credit_details.term],
        'emp_length_int': [credit_details.emp_length],
        'mths_since_issue_d': [credit_details.mths_since_issue_d],
        'mths_since_earliest_cr_line': [credit_details.mths_since_earliest_cr_line],
        'funded_amnt':[credit_details.funded_amnt],
        'int_rate': [credit_details.int_rate],
        'installment':[credit_details.installment],
        'annual_inc': [credit_details.annual_inc],
        'dti': [credit_details.dti],
        'delinq_2yrs':[credit_details.delinq_2yrs],
        'inq_last_6mths': [credit_details.inq_last_6mths],
        'mths_since_last_delinq': [credit_details.mths_since_last_delinq],
        'mths_since_last_record': [credit_details.mths_since_last_record],
        'open_acc':[credit_details.open_acc],
        'pub_rec':[credit_details.pub_rec],
        'total_acc':[credit_details.total_acc],
        'acc_now_delinq': [credit_details.acc_now_delinq],
        'total_rev_hi_lim':[credit_details.total_rev_hi_lim]
        })

        ref_categories = ['grade:G',
        'home_ownership:RENT',
        'verification_status:Verified',
        'purpose:credit_card',
        'initial_list_status:f']
        # Converting x_test dictionary to dataframe
        

        inputs_with_ref_cat = pd.DataFrame(x_test)
        inputs_without_ref_cat = inputs_with_ref_cat.drop(columns = ref_categories)
        for key in inputs_without_ref_cat:
            print(key)
        print(len(inputs_without_ref_cat))
        # filename='model.sav'
        # path = os.path.join(app.config['UPLOAD_FOLDER'][0], filename)
        # loaded_model = joblib.load(path)
        # prediction = loaded_model.model.predict(inputs_without_ref_cat)
        # print(prediction)
        loaded_model = pickle.load(open(os.path.join(app.config['UPLOAD_FOLDER'][0], 'solver_stage4_models', 'xgb.pkl'), 'rb'))
        prob_of_default= loaded_model.predict_proba(inputs_without_ref_cat)[: ][: , 0]
        print(prob_of_default)
        # Storing feature names with their coefficients (using loaded_model.coef_) in summary_table
        # feature_name = inputs_without_ref_cat.columns.values
        # summary_table = pd.DataFrame(columns = ['Feature name'], data = feature_name)
        # summary_table['Coefficients'] = np.transpose(loaded_model.coef_)
        # summary_table.index = summary_table.index + 1
        # summary_table.loc[0] = ['Intercept', loaded_model.intercept_[0]]
        # summary_table = summary_table.sort_index()
        # # Storing p_values in summary table
        # p_values = loaded_model.p_values
        # p_values = np.append(np.nan,np.array(p_values))
        # summary_table['p_values'] = p_values

        # We were not allowed to include the reference categories when estimating the model. However when using the model for probability of default, we just take their coefficients as 0, and p_values as nan, in order to make thee scorecard interpretable for normal person
        # Need to store reference categories in summary table so we do this:

        # df_ref_categories = pd.DataFrame(ref_categories, columns = ['Feature name'])
        # df_ref_categories['Coefficients'] = 0
        # df_ref_categories['p_values'] = np.nan
        # df_scorecard = pd.concat([summary_table, df_ref_categories])
        # df_scorecard = df_scorecard.reset_index()
        # df_scorecard['Original feature name'] = df_scorecard['Feature name'].str.split(':').str[0]
        # df_scorecard.groupby('Original feature name')['Coefficients'].min()
        # min_sum_coef = df_scorecard.groupby('Original feature name')['Coefficients'].min().sum()
        # df_scorecard.groupby('Original feature name')['Coefficients'].max()
        # max_sum_coef = df_scorecard.groupby('Original feature name')['Coefficients'].max().sum()

        # In order to create a scorecard, we need to turn the regression coefficients from our PD model into simple scores.
        # min_score = 300
        # max_score = 850
        # df_scorecard['Score - Calculation'] = df_scorecard['Coefficients'] * (max_score - min_score) / (max_sum_coef - min_sum_coef)
        # df_scorecard['Score - Calculation'][0] = ((df_scorecard['Coefficients'][0] - min_sum_coef) / (max_sum_coef - min_sum_coef)) * (max_score - min_score) + min_score
        # df_scorecard['Score - Preliminary'] = df_scorecard['Score - Calculation'].round()
        # df_scorecard['Difference'] = df_scorecard['Score - Preliminary'] - df_scorecard['Score - Calculation']
        # df_scorecard['Score - Final'] = df_scorecard['Score - Preliminary']
        # df_scorecard['Score - Final'][77] = 16
        # There, our scorecard is ready and the score-final column contains the score for each category.

        # According to scorecard we'll add the credit points of each variable.
        # print(df_scorecard)

        # Calculating credit score
        # inputs_with_ref_cat_w_intercept = inputs_with_ref_cat
        # if not 'Intercept' in inputs_with_ref_cat.columns:
        #     inputs_with_ref_cat_w_intercept.insert(0, 'Intercept', 1)
        # scorecard_scores = df_scorecard['Score - Final']
        # scorecard_scores = scorecard_scores.values.reshape(102, 1)
        # y_scores = inputs_with_ref_cat_w_intercept.dot(scorecard_scores)

        # credit_score = y_scores[0][0]
        # credit_score = 600
        credit_score = (1-prob_of_default[0])*500 + 350
        print(credit_score)

        # Calculating probability of default from credit score

        # sum_coef_from_score = ((y_scores - min_score) / (max_score - min_score)) * (max_sum_coef - min_sum_coef) + min_sum_coef
        # y_hat_proba_from_score = np.exp(sum_coef_from_score) / (np.exp(sum_coef_from_score) + 1)
        
        # prob_of_default = round(y_hat_proba_from_score[0][0] *100)

    return render_template('user_dashboard/home.html', credit_details=credit_details, credit_score=round(credit_score), prob_of_default= round(prob_of_default[0]*100))
    # return render_template('dashboard_home.html', credit_details=credit_details)


@app.route("/user_dashboard_fill_credit_details", methods=['GET','POST'])
@login_required
def user_dashboard_fill_credit_details():
    filled_credit_details = False

    credit_details=CreditDetails.query.filter_by(user_id=current_user.user_id).first()

    if credit_details:
        filled_credit_details=True
        form=CreditDetailsForm(obj=credit_details)   
    else: 
        form=CreditDetailsForm()

    if form.validate_on_submit():
        if credit_details:
            form.populate_obj(credit_details)
            db.session.commit()
            flash('You have successfully edited the credit details',category='successful')
        else:
            user_credit_details = CreditDetails(
                user_id = current_user.user_id,
                grade= form.grade.data,
                home_ownership= form.home_ownership.data,
                verification_status = form.verification_status.data,
                emp_length = form.emp_length.data,
                purpose = form.purpose.data,
                initial_list_status= form.initial_list_status.data,
                term = form.term.data,
                mths_since_issue_d = form.mths_since_issue_d.data,
                int_rate = form.int_rate.data,
                mths_since_earliest_cr_line = form.mths_since_earliest_cr_line.data,
                inq_last_6mths = form.inq_last_6mths.data,
                acc_now_delinq = form.acc_now_delinq.data,
                annual_inc = form.annual_inc.data,
                dti = form.dti.data,
                mths_since_last_delinq = form.mths_since_last_delinq.data,
                mths_since_last_record = form.mths_since_last_record.data,
                installment=form.installment.data,
                funded_amnt=form.funded_amnt.data,
                delinq_2yrs=form.delinq_2yrs.data,
                open_acc=form.open_acc.data,
                pub_rec=form.pub_rec.data,
                total_acc=form.total_acc.data,
                total_rev_hi_lim=form.total_rev_hi_lim.data
            )
                # addr_state= form.addr_state.data,

            db.session.add(user_credit_details)
            db.session.commit()
            flash('You have successfully submitted the credit details',category='successful')
        return redirect(url_for('user_dashboard_home'))
    
    return render_template('user_dashboard/fill_credit_details.html', form=form, credit_details=credit_details,filled_credit_details=filled_credit_details)






@app.route("/bank_dashboard_home", methods=['GET','POST'])
@login_required
def bank_dashboard_home():
   
    if request.method == 'POST':
        if request.files:
            uploaded_file_train = request.files['csv-file-train']
            uploaded_file_test = request.files['csv-file-test']
            exp_loss, fund_amt, exp_loss_perc, all_charts_data, all_tables_data, all_csv_charts_data=expected_loss_func(uploaded_file_train, uploaded_file_test)
            print(f"Expected loss: {exp_loss}, Funded Amount: {fund_amt}, Expected Loss Percentage:{exp_loss_perc}")
            print(f"all_charts_data_type:{type(json.dumps(all_charts_data))}")
            with open("data.json", "w") as f:
                json.dump(all_charts_data, f)
   
            return render_template('bank_dashboard/home.html', exp_loss=exp_loss, fund_amt=fund_amt, exp_loss_perc=exp_loss_perc, all_charts_data=json.dumps(all_charts_data), all_tables_data=all_tables_data, all_csv_charts_data = json.dumps(all_csv_charts_data))
            
    return render_template('bank_dashboard/home.html')



@app.route("/bank_dashboard_lgd", methods=['GET','POST'])
def bank_dashboard_lgd():
    return render_template('bank_dashboard/lgd.html')
    
