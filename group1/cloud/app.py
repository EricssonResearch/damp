#!flask/bin/python
from flask import Flask, jsonify
import subprocess
import sys

app = Flask(__name__)


@app.route('/federated/v1.1', methods=['GET'])
def federated():
    data=subprocess.check_output(["python3","main.py"], cwd="../federated/v1.1/")
    return data


@app.route('/federated/v1.2', methods=['GET'])
def federated_v2():
    data=subprocess.check_output(["python3","main.py"], cwd="../federated/v1.2/")
    return data


if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
