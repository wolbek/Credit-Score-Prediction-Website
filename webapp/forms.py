from ast import Pass
from tokenize import String
from xml.dom import ValidationErr
from flask_wtf import FlaskForm
from wtforms import StringField,PasswordField,SubmitField, IntegerField, SelectField,FileField, FloatField
from wtforms.validators import InputRequired, Email, Length, ValidationError, EqualTo, Regexp, NumberRange
from webapp.global_constants import grade, home_ownership, verification_status, purpose, initial_list_status, term
#  addr_state,term, emp_length, mths_since_issue_d, int_rate, mths_since_earliest_cr_line, inq_last_6mths, acc_now_delinq, annual_inc, dti, mths_since_last_delinq, mths_since_last_record
from webapp.models import User

class SignUpForm(FlaskForm):
    email=StringField(label='Email',validators=[InputRequired(),Email()], render_kw={"placeholder": "aakash@gmail.com"})
    first_name = StringField(label='First Name', validators=[InputRequired(),Length(max=100),Regexp('[a-zA-Z]+', message="The name should contain alphabets only.")], render_kw={"placeholder": "Aakash"})
    last_name = StringField(label='Last Name', validators=[ InputRequired(),Length(max=100),Regexp('[a-zA-Z]+', message="The name should contain alphabets only.")],  render_kw={"placeholder": "Vishwakarma"})
    password1=PasswordField(label='Password',validators=[InputRequired(),Length(min=8)],  render_kw={"placeholder": "Password"})
    submit=SubmitField('Sign Up')

    def validate_email(self,email):
        user=User.query.filter_by(email=email.data).first()
        if user:
            raise ValidationError('That email is taken. Please choose a different email')

class LoginForm(FlaskForm):
    email=StringField(label='Email',validators=[InputRequired(),Email()], render_kw={"placeholder": "aakash@gmail.com"})
    password=PasswordField(label='Password',validators=[InputRequired()],  render_kw={"placeholder": "Password"})
    submit=SubmitField('Log In')

class CreditDetailsForm(FlaskForm):
    grade= SelectField(label="Grade", validators=[InputRequired()], choices=[(op, op) for op in grade])
    home_ownership= SelectField(label="Home ownership status", validators=[InputRequired()], choices=[(op, op) for op in home_ownership])
    # addr_state= SelectField(label="Address state", validators=[InputRequired()], choices=[(op, op) for op in addr_state])
    verification_status= SelectField(label="Verification status", validators=[InputRequired()], choices=[(op, op) for op in verification_status])

    purpose = SelectField(label="Purpose of loan", validators=[InputRequired()], choices=[(op, op) for op in purpose])
    initial_list_status= SelectField(label="Initial list status", validators=[InputRequired()], choices=[(op, op) for op in initial_list_status])
    term = SelectField(label="Term of loan", validators=[InputRequired()], choices=[(op, op) for op in term], render_kw={"placeholder":"5"})
    
    emp_length = IntegerField(label="Length of employment", validators=[InputRequired(), NumberRange(min=0, max=10)], render_kw={"placeholder":"5"})

    mths_since_issue_d = IntegerField(label="Months since issue date", validators=[InputRequired()], render_kw={"placeholder":"5"})
    int_rate = FloatField(label="Interest rate", validators=[InputRequired()], render_kw={"placeholder":"5"})
    mths_since_earliest_cr_line = IntegerField(label="Months since earliest credit line", validators=[InputRequired()], render_kw={"placeholder":"5"})
    inq_last_6mths = IntegerField(label="Inquiries in last 6 months", validators=[InputRequired()], render_kw={"placeholder":"5"})
    acc_now_delinq = IntegerField(label="Number of accounts delinquent", validators=[InputRequired()], render_kw={"placeholder":"5"})
    annual_inc = FloatField(label="Annual income", validators=[InputRequired()], render_kw={"placeholder":"5000"})
    dti = FloatField(label="Debt to income ratio", validators=[InputRequired()], render_kw={"placeholder":"5"})
    mths_since_last_delinq = IntegerField(label="Months since last delinquency", validators=[InputRequired()], render_kw={"placeholder":"5"})
    mths_since_last_record = IntegerField(label="Months since last public record", validators=[InputRequired()], render_kw={"placeholder":"5"})

    installment=FloatField(label="Installment", validators=[InputRequired()], render_kw={"placeholder":"1000"})
    funded_amnt=FloatField(label="Funded Amount", validators=[InputRequired()], render_kw={"placeholder":"500000"})
    delinq_2yrs=IntegerField(label="Delinquency in 2 years", validators=[InputRequired()], render_kw={"placeholder":"5"})
    open_acc=IntegerField(label="Accounts Open", validators=[InputRequired()], render_kw={"placeholder":"5"})
    pub_rec=IntegerField(label="Public Records count", validators=[InputRequired()], render_kw={"placeholder":"5"})
    total_acc=IntegerField(label="Total Accounts", validators=[InputRequired()], render_kw={"placeholder":"5"})
    total_rev_hi_lim=FloatField(label="Total Revolving High Credit Limit", validators=[InputRequired()], render_kw={"placeholder":"500000"})

    # term = SelectField(label="Term of loan", validators=[InputRequired()], choices=[(op, op) for op in term])

    # emp_length = SelectField(label="Length of employment", validators=[InputRequired()], choices=[(op, op) for op in emp_length])

    # mths_since_issue_d = SelectField(label="Months since issue date", validators=[InputRequired()], choices=[(op, op) for op in mths_since_issue_d])
    # int_rate = SelectField(label="Interest rate", validators=[InputRequired()], choices=[(op, op) for op in int_rate])
    # mths_since_earliest_cr_line = SelectField(label="Months since earliest credit line", validators=[InputRequired()], choices=[(op, op) for op in mths_since_earliest_cr_line])
    # inq_last_6mths = SelectField(label="Inquiries in last 6 months", validators=[InputRequired()], choices=[(op, op) for op in inq_last_6mths])
    # acc_now_delinq = SelectField(label="Number of accounts delinquent", validators=[InputRequired()], choices=[(op, op) for op in acc_now_delinq])
    # annual_inc = SelectField(label="Annual income", validators=[InputRequired()], choices=[(op, op) for op in annual_inc])
    # dti = SelectField(label="Debt to income ratio", validators=[InputRequired()], choices=[(op, op) for op in dti])
    # mths_since_last_delinq = SelectField(label="Months since last delinquency", validators=[InputRequired()], choices=[(op, op) for op in mths_since_last_delinq])
    # mths_since_last_record = SelectField(label="Months since last public record", validators=[InputRequired()], choices=[(op, op) for op in mths_since_last_record])

    # submit=SubmitField('Submit')

class CsvUploadForm(FlaskForm):
    csv_file=FileField(label='Photo', validators=[InputRequired()])
