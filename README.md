## Run the Project Locally

### Clone the repository:  
```git clone https://github.com/wolbek/Credit-Score-Prediction-Website.git```

### Download these links materials:
```solver_stage1:``` https://drive.google.com/drive/folders/1LVoMpMwJlY9NqhIpGaXTgxB5_uUrzwrS?usp=sharing
```solver_stage2:``` https://drive.google.com/drive/folders/1r2rAN0kh8UFHzZE35s9CW8I0rlv_04i0?usp=sharing
```solver_stage3:``` https://drive.google.com/drive/folders/1UWfqXaTmuo2MrATXi6mLOBQvjm1Z4o10?usp=sharing
```solver_stage4:``` https://drive.google.com/drive/folders/11JrnWkmm3vOFwCpoWHjlizN-9V7BHPQb?usp=sharing

### Make an "uploads" folder and put those materials there

### Download below csv files to put them as example input in bank's side
```Train.csv``` https://drive.google.com/file/d/1MXHtDci1OUuvXNWBFLPM4ZXpEKMfpmhl/view?usp=sharing
```Test.csv``` https://drive.google.com/file/d/1iZhAaHe-toj8zF1vlxzBd0LjwBNi2J6v/view?usp=sharing

### CD into the project:  
```cd Credit-Score-Prediction-Website```

### Make virtual environment:  
```python -m venv venv```

### To enable running scripts on system (type below command on Powershell administrator mode. Ignore if already done)
```Set-ExecutionPolicy Unrestricted```

### Activate virtual environment:  
```.\venv\Scripts\activate```

### Install requirements.txt:  
```pip install -r requirements.txt```

### Initialize database and create folders in uploads:  
```.\run.bat init-db```

### Create users:  
```.\run.bat create-users```

### (optional) Seed data:   
```.\run.bat seed-data```

### Run the application:   
```python run.py```

## Credentials for the Project

### User:
### To login:
**Email:** aakash@gmail.com  
**Password:** user
### OR
**Email:** roy@gmail.com 
**Password:** user

### Bank:
### To login:
**Email:** bank@gmail.com
**Password:** bank
