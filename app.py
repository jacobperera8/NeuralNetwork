import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
import boto3
import requests
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from flask import Flask, jsonify, make_response

app = Flask(__name__)


@app.route("/")
def hello_from_root():
    return jsonify(message='Hello from root!')


@app.route("/hello")
def hello():
    return jsonify(message='Hello from path!')


# def lambda_handler(event, context):
#     print(event)
#
#     object_get_context = event["getObjectContext"]
#     request_route = object_get_context["outputRoute"]
#     request_token = object_get_context["outputToken"]
#     s3_url = object_get_context["inputS3Url"]
#
#     # Get object from S3
#     response = requests.get(s3_url)
#     original_object = response.content.decode('utf-8')
#
#     # Transform object
#     transformed_object = original_object.upper()
#
#     # Write object back to S3 Object Lambda
#     s3 = boto3.client('s3')
#     s3.write_get_object_response(
#         Body=transformed_object,
#         RequestRoute=request_route,
#         RequestToken=request_token)
#
#     return {'status_code': 200}

@app.route('/process', methods=['POST'])
def insert():
    if request.method == "POST":
        url_key = request.get_json(silent=True)['urlKey']
        dir = request.get_json(silent=True)['dir']
        s3 = boto3.client('s3')
        # bucket_name = 'test-upload-bucket-devy'
        bucket_name = 'html-corrector-bucket'

        object_key = url_key
        # object_key = 'unprocessed-data/2022-07-25T17:27:43.040Z-68bbf31f-f727-4746-93b8-1b3df1f39f6b/Dataset.xlsx'

        obj = s3.get_object(Bucket=bucket_name, Key=object_key)

        df = pd.read_excel(io.BytesIO(obj['Body'].read()))

        print(df.head())

        # convert an array of values into a dataset matrix
        def create_dataset(dataset, look_back=1):
            dataX, dataY = [], []
            for i in range(len(dataset)-look_back-1):
                a = dataset[i:(i+look_back), 0]
                dataX.append(a)
                dataY.append(dataset[i + look_back, 0])
            return numpy.array(dataX), numpy.array(dataY)

        # fix random seed for reproducibility
        numpy.random.seed(7)

        # load the dataset
        dataframe = read_csv('result.csv', usecols=[300], engine='python', skipfooter=3)
        dataset = dataframe.values
        dataset = dataset.astype('float32')

        # normalize the dataset
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)

        # split into train and test sets
        train_size = int(len(dataset) * 0.8)
        test_size = len(dataset) - train_size
        train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

        # reshape into X=t and Y=t+1
        look_back = 1
        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)

        # reshape input to be [samples, time steps, features]
        trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

        # create and fit the LSTM network
        model = Sequential()
        model.add(LSTM(4, input_shape=(1, look_back)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        model.fit(trainX, trainY, epochs=10, batch_size=5, verbose=2)

        # make predictions
        trainPredict = model.predict(trainX)
        testPredict = model.predict(testX)

        # invert predictions
        trainPredict = scaler.inverse_transform(trainPredict)
        trainY = scaler.inverse_transform([trainY])
        testPredict = scaler.inverse_transform(testPredict)
        testY = scaler.inverse_transform([testY])

        # calculate root mean squared error
        trainScore = numpy.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
        print('Train Score: %.2f RMSE' % (trainScore))
        testScore = numpy.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
        print('Test Score: %.2f RMSE' % (testScore))

        # shift train predictions for plotting
        trainPredictPlot = numpy.empty_like(dataset)
        trainPredictPlot[:, :] = numpy.nan
        trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

        # shift test predictions for plotting
        testPredictPlot = numpy.empty_like(dataset)
        testPredictPlot[:, :] = numpy.nan
        testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

        # plot baseline and predictions
        # plt.plot(scaler.inverse_transform(dataset))
        # plt.plot(trainPredictPlot)
        # plt.plot(testPredictPlot)
        # plt.show()

@app.errorhandler(404)
def resource_not_found(e):
    return make_response(jsonify(error='Not found!'), 404)
