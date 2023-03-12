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
    user_id = db.Column(db.Integer, db.ForeignKey("user.user_id"), primary_key=True)

    grade= db.Column(db.String(5), nullable=False, server_default='')
    home_ownership= db.Column(db.String(50), nullable=False, server_default='')
    addr_state= db.Column(db.String(50), nullable=False, server_default='')
    verification_status= db.Column(db.String(50), nullable=False, server_default='')
    emp_length = db.Column(db.String(50), nullable=False, server_default='')

    purpose = db.Column(db.String(50), nullable=False, server_default='')
    initial_list_status= db.Column(db.String(5), nullable=False, server_default='')
    term = db.Column(db.String(50), nullable=False, server_default='')
    mths_since_issue_d = db.Column(db.String(50), nullable=False, server_default='')
    int_rate = db.Column(db.String(50), nullable=False, server_default='')
    mths_since_earliest_cr_line = db.Column(db.String(50), nullable=False, server_default='')
    inq_last_6mths = db.Column(db.String(50), nullable=False, server_default='')
    acc_now_delinq = db.Column(db.String(50), nullable=False, server_default='')
    annual_inc = db.Column(db.String(50), nullable=False, server_default='')
    dti = db.Column(db.String(50), nullable=False, server_default='')
    mths_since_last_delinq = db.Column(db.String(50), nullable=False, server_default='')
    mths_since_last_record = db.Column(db.String(50), nullable=False, server_default='')

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
    )
    user_1.set_password('user')
    db.session.add(user_1)
    db.session.commit()

    #Creating user 2
    user_2=User(
        email='roy@gmail.com',
        first_name='Siddhant',
        last_name='Roy', 
    )
    user_2.set_password('user')
    db.session.add(user_2)
    db.session.commit()

    click.echo('Users created.')

@click.command('seed-data')
@with_appcontext
def seed_data_command():
    click.echo('Seeded data.')