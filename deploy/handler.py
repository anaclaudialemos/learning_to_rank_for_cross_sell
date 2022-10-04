import os
import pickle
import pandas as pd

from flask                import Flask, request, Response
from insurance.CrossSell  import CrossSell

# loading model
model = pickle.load(open('models/cycle1/model/knn_model_c1.pkl', 'rb'))

# initializing API
app = Flask(__name__)

@app.route('/predict', methods=['POST']) 
def crosssell_prediction():
    test_json = request.get_json()

    if test_json: # there is data
        if isinstance(test_json, dict): # if unique example
            test_df = pd.DataFrame(test_json, index=[0])
            
        else: # if multiple examples
            test_df = pd.DataFrame(test_json, columns=test_json[0].keys())

        # instantiate CrossSell Class
        pipeline = CrossSell()

        # data cleaning
        df1 = pipeline.data_cleaning(test_df)

        # feature engineering
        df2 = pipeline.feature_engineering(df1)

        # data preparation
        df3 = pipeline.data_preparation(df2)

        # prediction
        df_response = pipeline.get_prediction(model, test_df, df3)

        return df_response

    else: # if empty
        return Response('{}', status=200, mimetype='application/json')

if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    app.run('192.168.1.101', port=port)
