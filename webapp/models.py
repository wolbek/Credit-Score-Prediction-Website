from bcrypt import checkpw, hashpw, gensalt
from flask.cli import with_appcontext
from webapp import db,login_manager
from flask_login import UserMixin
import click

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(db.Model,UserMixin):
    user_id = db.Column(db.Integer, primary_key=True)
    role = db.Column(db.String(20), nullable=False)
    email = db.Column(db.String(255), nullable=False, unique=True)
    first_name = db.Column(db.String(100), nullable=False, server_default='')
    last_name = db.Column(db.String(100), nullable=False, server_default='')
    password = db.Column(db.String(255), nullable=False, server_default='')
    credit_details = db.relationship("CreditDetails", cascade="all, delete", backref="User", lazy=True)

    #Because we've defined user_id and not id, we'll need to specify that we're taking user_id for login
    def get_id(self):
        return (self.user_id)

    def set_password(self, password):
        self.password = hashpw(password.encode('utf-8'), gensalt()).decode('ascii')

    def check_password(self, password):
        return checkpw(password.encode('utf-8'), self.password.encode('utf-8'))

class CreditDetails(db.Model):
    credit_detail_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.user_id"))

    grade= db.Column(db.String(5), nullable=False, server_default='')
    home_ownership= db.Column(db.String(50), nullable=False, server_default='')
    addr_state= db.Column(db.String(50), nullable=False, server_default='')
    verification_status= db.Column(db.String(50), nullable=False, server_default='')

    purpose = db.Column(db.String(50), nullable=False, server_default='')
    initial_list_status= db.Column(db.String(5), nullable=False, server_default='')

    term = db.Column(db.Integer, nullable=False, default=0)

    emp_length = db.Column(db.Integer, nullable=False, default=0)

    mths_since_issue_d = db.Column(db.Integer, nullable=False, default=0)
    int_rate = db.Column(db.Float, nullable=False, default=0)
    mths_since_earliest_cr_line = db.Column(db.Integer, nullable=False, default=0)
    inq_last_6mths = db.Column(db.Integer, nullable=False, default=0)
    acc_now_delinq = db.Column(db.Integer, nullable=False, default=0)
    annual_inc = db.Column(db.Float, nullable=False, default=0)
    dti = db.Column(db.Float, nullable=False, default=0)
    mths_since_last_delinq = db.Column(db.Integer, nullable=False, default=0)
    mths_since_last_record = db.Column(db.Integer, nullable=False, default=0)

    installment = db.Column(db.Float, nullable=False, default=0)
    funded_amnt = db.Column(db.Float, nullable=False, default=0)
    delinq_2yrs = db.Column(db.Integer, nullable=False, default=0)
    open_acc = db.Column(db.Integer, nullable=False, default=0)
    pub_rec = db.Column(db.Integer, nullable=False, default=0)
    total_acc = db.Column(db.Integer, nullable=False, default=0)
    total_rev_hi_lim = db.Column(db.Float, nullable=False, default=0)

@click.command('init-db')
@with_appcontext
def init_db_command():
    db.create_all()   
    click.echo('Initialized the database.')

@click.command('create-users')
@with_appcontext
def create_users_command():
    
    #Creating user 1
    user_1=User(
        email='aakash@gmail.com',
        first_name='Aakash',
        last_name='Vishwakarma', 
        role='User'
    )
    user_1.set_password('user')
    db.session.add(user_1)
    db.session.commit()

    #Creating user 2
    user_2=User(
        email='roy@gmail.com',
        first_name='Siddhant',
        last_name='Roy', 
        role='User'
    )
    user_2.set_password('user')
    db.session.add(user_2)
    db.session.commit()

    #Creating bank
    user_3=User(
        email='bank@gmail.com',
        first_name='Dinesh',
        last_name='Kumar', 
        role='Bank'
    )
    user_3.set_password('bank')
    db.session.add(user_3)
    db.session.commit()

    click.echo('Users created.')

@click.command('seed-data')
@with_appcontext
def seed_data_command():
    click.echo('Seeded data.')