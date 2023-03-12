# from urllib import request
from webapp import app,db,bcrypt
from flask import redirect, render_template, url_for,flash, request
from webapp.models import User, CreditDetails
from webapp.forms import SignUpForm,LoginForm, CreditDetailsForm
from flask_login import current_user, login_required, login_user, logout_user

@app.route('/signup',methods=['GET','POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form=SignUpForm()
    if form.validate_on_submit():
        user=User(
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

@app.route("/dashboard_home")
@login_required
def dashboard_home():
    credit_details=CreditDetails.query.filter_by(user_id=current_user.user_id).first()
    credit_score=451
    prob_of_default=65
    return render_template('dashboard_home.html', credit_details=credit_details, credit_score=credit_score, prob_of_default= prob_of_default)

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/dashboard_fill_credit_details", methods=['GET','POST'])
def dashboard_fill_credit_details():
    # credit_details=CreditDetails.query.filter_by(user_id=current_user.user_id).first()
    # if credit_details:
    #     form=CreditDetailsForm(obj=credit_details)   
    # else: 
    form=CreditDetailsForm()
    if form.validate_on_submit():
        # form.populate_obj(credit_details)
        user_credit_details = CreditDetails(
            user_id = current_user.user_id,

            grade= form.grade.data,
            home_ownership= form.home_ownership.data,
            addr_state= form.addr_state.data,
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
        )
        db.session.add(user_credit_details)
        db.session.commit()
        return redirect(url_for('dashboard_home'))
    return render_template('dashboard_fill_credit_details.html', form=form)


