# TrafficPoliceActionRecognition_Lightweigh
We introduce very simple model which Resnet+LSTM that can recognize the traffic police wand action. 
![Figure 1](./images/mai2022_SCSRN_2nd.png)
The result of our simple model shows similar result with SlowFast. Also, we tested SlowFast method.
In this test, we use traffic police wand action recognition dataset. For simplicity, we didnot use full dataset, we only use police wand set at daytime. 
[Link](https://www.data.go.kr/data/15075814/fileData.do)
# Requirements
pip install -r requirements.txt

# Prepare Traffic Action Dataset
1. Crop bbox of Traffic Police from full dataset.
```bash
python data_preprocess_python.py
```
2. Increase boundary size.
```bash
python data_preprocess_python_growth.py
```

# Training
```bash
python train_inference_python.py --pathTrain {Train_Path} --pathVal {Valid_Path}
```

# Inferencing
```bash
python inference_python.py --pathTrain {Train_Path} --pathVal {Valid_Path}
```
