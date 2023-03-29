## Run the Project Locally

### Clone the repository:  
```git clone https://github.com/wolbek/Credit-Score-Prediction-Website.git```

### Download these links materials, make an up:
```model.sav:``` https://drive.google.com/file/d/19Yjjxv3RsTggwZv80bjo7el31NVyTrAk/view?usp=share_link
```solver_stage1:``` https://drive.google.com/drive/folders/1LmP8pS9floRUdOs_MQ0jn0YdzytXhY5Z?usp=share_link
```solver_stage2:``` https://drive.google.com/drive/folders/10Q3uMR71mlsY1CYjz6kl4lg5Hia_YZpf?usp=share_link
```solver_stage3:``` https://drive.google.com/drive/folders/12UbkCr_CiQfblbtAm59aWF9M1wT0KJQB?usp=share_link
```solver_stage4:``` https://drive.google.com/drive/folders/14Q9ii9Ge1_G8Vw9yMH36-nSAfQ3J-sF3?usp=share_link

### Make an "uploads" folder and put those materials there

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
