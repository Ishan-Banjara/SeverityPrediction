# **Project information**
This project is dedicated to predict the severity of a given accident based on data about accidents in the US. The data involved in the study is from US Accidents (2016 - 2023) from Kaggle. The methodology involved in this project can be divided into 5 steps.

# **Understanding the Problem**
The objective of the study is to predict a crash severity on traffic. Since the project involved is to labelize the severity of the given accident based on information available, it will be formulated as **classification** problem. Model performance will be evaluated using accuracy and sparse categorical crossentropy metrics.

# **Data Acquisition and Insights**
The data involved in the study is from [US Accidents (2016 - 2023)](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents?select=US_Accidents_March23.csv) from Kaggle. It contains accident data collected from February 2016 to March 2023 for the Contiguous United States. The features and description of features of the data are provided [here](https://smoosavi.org/datasets/us_accidents). In this section, statistical descriptions such as mean, standard deviation, min, max, and median of variables will be analyzed to understand the trend of the data. Next, GIS Spatial pattern analysis, histograms for data distributions of numerical variables, pie-charts for data distribution of categorical variables, and a look through of correlation of the variables using heatmaps will be performed to get the feel of the type of data being dealt. 
![download](https://github.com/user-attachments/assets/fd80ed06-0c1f-49a5-800c-68335478814d)
![download](https://github.com/user-attachments/assets/155aad0e-b539-4f24-a51a-25f0548b1ec0)
**Limitation in this project**: Implementation of Folium would have been effective for analysis. However, due to limited free RAM resources, matplotlib is being used.

**Limitation in this project**: Lack of system memory constrained the study from developing further visualizations such as histographs, pie-charts, and heatmaps.

Additionally, new features such as seasonal crash frequency will be engineered to understand the driver’s expectancy from a curve design during a particular season. Finally, the provided data will be split into training and testing subsets (80%-20%) to ensure
reliable model evaluation. Here, we will be implementing stratified splitting. This is because, normal splitting does not guarantee the preservataion of distribution of classes in both train and test sets.

# **Data Preprocessing**
The exclusion of other irrelevant variables and handling of missing data for relevant variables. Multi-class data will be encoded using One Hot Encoding while binary-class data will be encoded using Label Encoder. Scaling of continuous data will be done using Standard scaling.

# **Model Development**
For the study, ANN model would be used. The loss function used during training is sparse categorical crossentropy. The model is trained on Google Colab T4 free GPU for 10 epochs with model checkpoint and early stopping patience of 3 with dataset of batch_size 32.

**Limitation in this project**: Here, due to lack of resources, model tuning will not be performed.

# Evaluation and Interpretation

The best model for ANN is loaded for evaluation. After evaluation it was realized that inclusion of other road characteristics data such as shoulder width, injury severity, different road characteristics would have enhanced the quality of this study. Additionally, a computer with better computational resources would also have helped in the study to delve deeper through tuning. Since ANN is a "black-box" model, it was incorporated with Shapley Additive exPlanations (SHAP) to explain the feature important for prediction of severity.

**Limitation in this project**: Due to lack of computational resources, background sample was reduced to 100 for shap explanation


![download](https://github.com/user-attachments/assets/287a720c-373c-44cf-bb65-44431f399c9f)
![download](https://github.com/user-attachments/assets/5d297246-2f60-449e-ac73-e6164d145c98)
![download](https://github.com/user-attachments/assets/3c3571fa-10c5-4e25-8c2c-565415e955d4)
![download](https://github.com/user-attachments/assets/61ff84a4-3c04-4ecb-92cc-8242d40414fb)

# **Difficulty during this study and how was it solved?**

Since, the data used for this study was about 7.7 mil rows x 46 cols, loading/processing of these data was a big issue. To solve the issue with loading of the data using pandas, following code was implemented:
```python
import pandas as pd
chunksize = 10000
chunks = pd.read_csv("US_Accidents_March23.csv", chunksize=chunksize)
data = pd.concat(chunks, ignore_index=True)
```
During processing of data, the RAM system kept crashing. This was handled by deleting unnecessary variables as following:
```python
for var in list(globals().keys()):
  if var != 'data' and not var.startswith('__'):
    del globals()[var]

import gc
gc.collect()
print(globals().keys())
```
