import subprocess

def main():
    subprocess.run(['python', 'src/data_preprocessing.py'])
    subprocess.run(['python', 'src/feature_engineering.py'])
    subprocess.run(['python', 'src/model_training.py'])
    subprocess.run(['python', 'src/model_evaluation.py'])

if __name__ == "__main__":
    main()
