FROM ubuntu
RUN apt-get update
RUN apt-get -y upgrade 
RUN apt-get install -y git
RUN apt-get install -y python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install numpy
RUN pip3 install nest_asyncio
RUN pip3 install pandas
RUN pip3 install sklearn
RUN pip3 install tensorflow==2.2.0
RUN pip3 install keras
RUN pip3 install tensorflow-federated
RUN git clone https://github.com/RaheelTheDeveloper/damp.git
RUN pip3 install flask
RUN apt install unzip
RUN unzip -j damp/federated/v1.2/2015-2019_sensorData_metrologicalData.zip -d /damp/federated/v1.2/
WORKDIR /damp/cloud/
EXPOSE 5000
ENV PATH="${PATH}:/usr/games/"
CMD ["python3","app.py"]
