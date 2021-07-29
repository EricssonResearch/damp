# Project-in-CS
Decentralized Air Quality Prediction using Machine Learning

### Instructions: 

- Open Google Colab and mount your drive
- Go to `air_quality_prediction.ipynb` and change the locations where your training and test data resides. 

### Features completed:

- AirData class for loading and wrangling data, Model class for ease-tuning the deep learning models.
- Two prediction method containing predict next hour using test data and next n hours by point by point prediction in Model class.

### Upcoming features:

- DNN and GAN model implementation
- Include plot_mse function vs. epoch
- Create new variable called windows in AirData class so that the experiments could be run using windows and single data records.
- Create parameter for tuning create_windows function such as window_size, window_interval, etc.
- New prediction method sequence-2-sequence prediction
