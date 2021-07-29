# Project-in-CS
Decentralized Air Quality Prediction using Machine Learning (DAMP)


## Getting Started

These instructions will get you a copy of the project up and running on your local or Virtual machines for execution development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

First of all, you need at least 5GB memory available. Then, you need to use Ubuntu 20.04 in Cloud or install Python 3.8 in your local machine explicitly. 
Also you need to download and set up Docker with below bash file command. 

```
sudo bash cloud/docker.sh
```

### Installing
1) If you want to create your own container in Cloud, you can set up your Virtual Machine using cloud/cloud-config file. 
And you find your container is already running when you check the output with below command.

```
curl -i http://<your-public-ip>:5000/federated/v1.2
```

For example, we already set up a VM with a public ip of 130.238.29.95. You can run the command until the VM expires.

```
curl -i http://130.238.29.95:5000/federated/v1.1

curl -i http://130.238.29.95:5000/federated/v1.2
```

2) If you want to run it in your local machine. You can download Docker image of our current version of damp from this link: [DockerHub](https://hub.docker.com/r/ekomurcu/damp/tags?page=1&ordering=last_updated)
after you login.
```
sudo bash
docker login
docker pull ekomurcu/damp:federatedv2
```

You can check if you downloaded it correctly by executing below command and see there there exists imageID with eb73885f1ed6

```
docker images
```

## Running the system

In order to run the container image, run 

```
docker run -d -p 5000:5000 ekomurcu/damp:federatedv2
```
and check if it is working with your public ip by 

```
docker container ls -a
curl -i http://<your-public-ip>:5000/federated/v1.2
```



## Development

In order to develop damp project further, first get the name of your running container other than ubuntu.
 
```
docker container ls -a
```

and then go into the container where you want to change the preprocessing or model files by

```
docker exec -it <container-name> /bin/bash
```

Then, stop previously running container to free port of 5000 and build the new image by

```
docker container stop
docker build --no-cache -t ekomurcu/damp:federatedv2 .
```

Lastly, run your custom container again with commands in "Running the system section". 


## Results
The data being used for each model is at the corresponding Data folder of that version. 

### Centralised Models

| Versions  | Preprocessing  | Model  |  SMAPE | 
| ------------- | ------------- |  ------------- | ------------- | 
| v1.1 | Propogate null | LSTM | 0.51 |
| v1.2 | Meteorological Data | LSTM | 0.46 | 

### Federated Models

| Versions  |  Preprocessing  | Model |  MAE |
| -------------  |  ------------- | ------------- |------------- |
| v1.0  | Feature importance XG Boost | TFF LSTM | 0.1 |
| v1.1 |  Propogate null | FEDn, TFF  | 0.021 |
| v1.2  |  One-hot encoding, GAN imputation| TFF LSTM | 0.014 |

## Built With

* [Tensorflow](https://www.tensorflow.org/) -  The core open source library to help you develop and train ML models.
* [TensorflowFederated](https://www.tensorflow.org/federated) - Open-source framework for machine learning on decentralized data
* [FEDn](https://github.com/scaleoutsystems/fedn) - Scalable federated learning in production 
* [Flutter](https://flutter.dev/) - The GoogleUI Kit for mobile and web applications
* [Tflite](https://www.tensorflow.org/lite) - Open source deep learning framework for on-device inference.
* [Docker](https://www.docker.com/) - The set of PaaS products that deliver software in packages called containers.

## Authors

See also the list of [contributors](https://github.com/RaheelTheDeveloper/damp/blob/main/contributors.md) who participated in this project.

## License

This project is licensed under the UnLicense - see the [LICENSE.md](https://github.com/RaheelTheDeveloper/damp/blob/main/LICENSE) file for details

## Acknowledgments
We are really thankful to our supervisors below. 

* Konstantinos Vandikas
* Vera van Zoest
* Tobias Mages
