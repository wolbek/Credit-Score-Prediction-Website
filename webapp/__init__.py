from flask import Flask,render_template
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager
from flask_wtf.csrf import CSRFProtect

app=Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI']='sqlite:///database.db'
app.config['SECRET_KEY']='secretkey'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS']=False
db=SQLAlchemy(app)
csrf = CSRFProtect(app)

bcrypt=Bcrypt(app)
login_manager=LoginManager(app)

login_manager.login_view='login'
login_manager.login_message = u"Login to access the website"
login_manager.login_message_category='info'

# register database
from webapp.models import init_db_command, create_users_command,seed_data_command
app.cli.add_command(init_db_command)
app.cli.add_command(create_users_command)
app.cli.add_command(seed_data_command)

from webapp import routes