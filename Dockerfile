FROM nvcr.io/nvidia/pytorch:21.03-py3
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY TrainingData_test NNgamma/TrainingData_test
COPY TrainingData NNgamma/TrainingData
