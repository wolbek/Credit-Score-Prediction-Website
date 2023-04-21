import json
from webapp import app,db,bcrypt
from flask import redirect, render_template, url_for,flash, request
from webapp.models import User, CreditDetails
from webapp.forms import SignUpForm,LoginForm, CreditDetailsForm, CsvUploadForm
from flask_login import current_user, login_required, login_user, logout_user
from .expected_loss import expected_loss_func
from flask import Flask, send_file
import pickle
from flask import Flask, send_from_directory, make_response
import io
import joblib
import numpy as np
import pandas as pd
import os
from webapp.global_constants import grade, home_ownership, verification_status, purpose, initial_list_status

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
    prob_of_default=[0]

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

        # Creating a dictionary containing all the columns with their values according to credit details

        x_test={}
        for i in grade:
            x_test["grade:"+i] = [1] if credit_details.grade==i else [0]
        for i in home_ownership:
            x_test["home_ownership:"+i] = [1] if credit_details.home_ownership==i else [0]
        for i in verification_status:
            x_test["verification_status:"+i] = [1] if credit_details.verification_status==i else [0]
        for i in purpose:
            x_test["purpose:"+i] = [1] if credit_details.purpose==i else [0]
        for i in initial_list_status:
            x_test["initial_list_status:"+i] = [1] if credit_details.initial_list_status==i else [0]
        print("Working - 1")

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
       
        loaded_model = pickle.load(open(os.path.join(app.config['UPLOAD_FOLDER'][0], 'solver_stage4_models', 'xgb.pkl'), 'rb'))
        prob_of_default= loaded_model.predict_proba(inputs_without_ref_cat)[: ][: , 0]
        print(prob_of_default)
        credit_score = (1-prob_of_default[0])*500 + 350
        print(credit_score)

    return render_template('user_dashboard/home.html', credit_details=credit_details, credit_score=round(credit_score), prob_of_default= round(prob_of_default[0]*100))


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

            db.session.add(user_credit_details)
            db.session.commit()
            flash('You have successfully submitted the credit details',category='successful')
        return redirect(url_for('user_dashboard_home'))
    
    tooltip_info = [
        'Bank assigned loan grade',
        'The home ownership status provided by the borrower during registration. Our values are: RENT, OWN, MORTGAGE, OTHER.',
        'Shows the status of an applicant\'s information, indicating whether or not it has been verified for accuracy by the lender or a third-party source.',
        'A category provided by the borrower for the loan request. ',
        'The initial listing status of the loan. If the loan is posted as "fractional" (i.e., "f"), investors can purchase a fraction of the loan. If the loan is posted as "whole" (i.e., "w"), investors must purchase the entire loan amount.',
        'The number of payments on the loan. Values are in months and can be either 36 or 60.',
        'Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years. ',
        'Months since the month when the loan was funded',
        'Interest Rate on the loan',
        'Months since the month the borrower\'s earliest reported credit line was opened',
        'The number of inquiries in past 6 months (excluding auto and mortgage inquiries)',
        'The number of accounts on which the borrower is now delinquent.',
        'The self-reported annual income provided by the borrower during registration.',
        'A ratio calculated using the borrower\'s total monthly debt payments on the total debt obligations, excluding mortgage and the requested Bank loan, divided by the borrower\'s self-reported monthly income.',
        'The number of months since the borrower\'s last delinquency.',
        'The number of months since the last public record.',
        'The monthly payment owed by the borrower if the loan originates.',
        'The total amount committed to that loan at that point in time.',
        'The number of 30+ days past-due incidences of delinquency in the borrower\'s credit file for the past 2 years',
        'The number of open credit lines in the borrower\'s credit file.',
        'Number of derogatory public records',
        'The total number of credit lines currently in the borrower\'s credit file',
        'The maximum amount of credit that a borrower is authorized to use on a revolving credit account, such as a credit card or a line of credit. The "Total Revolving High Credit Limit" is the sum of the credit limits on all of a borrower\'s revolving credit accounts.',
    ]
    return render_template('user_dashboard/fill_credit_details.html', form=form, credit_details=credit_details,filled_credit_details=filled_credit_details, tooltip_info=tooltip_info)


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