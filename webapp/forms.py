from ast import Pass
from tokenize import String
from xml.dom import ValidationErr
from flask_wtf import FlaskForm
from wtforms import StringField,PasswordField,SubmitField, IntegerField, SelectField
from wtforms.validators import InputRequired, Email, Length, ValidationError, EqualTo, Regexp

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
    grade= SelectField(label="Grade", validators=[InputRequired()], choices=["A", "B", "C", "D", "E"])
    home_ownership= SelectField(label="Home ownership status", validators=[InputRequired()], choices=["RENT_OTHER_NONE_ANY", "OWN", "MORTGAGE"])
    addr_state= SelectField(label="Address state", validators=[InputRequired()], choices=["ND_NE_IA_NV_FL_HI_AL", "NM_VA", "NY", "OK_TN_MO_LA_MD_NC", "CA", "UT_KY_AZ_NJ", "AR_MI_PA_OH_MN", "RI_MA_DE_SD_IN", "GA_WA_OR", "WI_MT", "TX", "IL_CT", "KS_SC_CO_VT_AK_MS", "WV_NH_WY_DC_ME_ID"])
    verification_status= SelectField(label="Verification status", validators=[InputRequired()], choices=["Verified", "Source Verified","Not Verified"])
    emp_length = SelectField(label="Length of employment", validators=[InputRequired()], choices=["0", "1", "2-4", "5-6", "7-9", "10"])

    purpose = SelectField(label="Purpose of loan", validators=[InputRequired()], choices=["educ__sm_b__wedd__ren_en__mov__house", "credit_card", "debt_consolidation", "oth__med__vacation", "major_purch__car__home_impr"])
    initial_list_status= SelectField(label="Initial list status", validators=[InputRequired()], choices=["f","w"])
    term = SelectField(label="Term of loan", validators=[InputRequired()], choices=["36","60"])
    mths_since_issue_d = SelectField(label="Months since issue date", validators=[InputRequired()], choices=["<38", "38-39", "40-41", "42-48", "49-52", "53-64", "65-84", ">84"])
    int_rate = SelectField(label="Interest rate", validators=[InputRequired()], choices=["<9.548", "9.548-12.025", "12.025-15.74", "15.74-20.281", ">20.281"])
    mths_since_earliest_cr_line = SelectField(label="Months since earliest credit line", validators=[InputRequired()], choices=["<140", "141-164", "165-247", "248-270", "271-352", ">352"])
    inq_last_6mths = SelectField(label="Inquiries in last 6 months", validators=[InputRequired()], choices=["0","1-2","3-6",">6"])
    acc_now_delinq = SelectField(label="Number of accounts delinquent", validators=[InputRequired()], choices=["0",">=1"])
    annual_inc = SelectField(label="Annual income", validators=[InputRequired()], choices=["<20K", "20K-30K", "30K-40K", "40K-50K", "50K-60K", "60K-70K", "70K-80K", "80K-90K", "90K-100K", "100K-120K", "120K-140K", ">140K"])
    dti = SelectField(label="Debt to income ratio", validators=[InputRequired()], choices=["<=1.4", "1.4-3.5", "3.5-7.7", "7.7-10.5", "10.5-16.1", "16.1-20.3", "20.3-21.7", "21.7-22.4", "22.4-35", ">35"])
    mths_since_last_delinq = SelectField(label="Months since last delinquency", validators=[InputRequired()], choices=["Missing", "0 - 3", "4 - 30", "31 - 56", ">=57"])
    mths_since_last_record = SelectField(label="Months since last public record", validators=[InputRequired()], choices=["Missing", "0 - 2", "3 - 20", "21 - 31", "32 - 80", "81 - 86", ">86"])
