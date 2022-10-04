import pickle
import numpy  as np
import pandas as pd


class CrossSell(object):
    
    def __init__(self):
        self.home_path ='models/cycle1/parameters/'
        self.annual_premium_scaler       = pickle.load(open(self.home_path + 'annual_premium_scaler.pkl','rb'))
        self.age_scaler                  = pickle.load(open(self.home_path + 'age_scaler.pkl','rb'))
        self.vintage_scaler              = pickle.load(open(self.home_path + 'vintage_scaler.pkl','rb'))
        self.region_code_scaler          = pickle.load(open(self.home_path + 'region_code_scaler.pkl','rb'))
        self.policy_sales_channel_scaler = pickle.load(open(self.home_path + 'policy_sales_channel_scaler.pkl','rb'))
        
    
    def data_cleaning(self, df1):

        df1 = df1.dropna()

        return df1 
    
    def feature_engineering(self, df2):

        # vehicle_damage as number
        df2['vehicle_damage'] = df2['vehicle_damage'].apply(lambda x: 1 if x == 'Yes' else 0).astype(np.int64)

        # vehicle_age
        df2['vehicle_age'] = df2['vehicle_age'].apply(lambda x: 'over_2_years' if x == '> 2 Years' else 
                                                                'between_1_2_year' if x == '1-2 Year' else 
                                                                'below_1_year')
        # gender as lowecase
        df2['gender'] = df2['gender'].apply(lambda x: x.lower())

        return df2
    
    def data_preparation (self, df3):
            
        # annual premium
        df3['annual_premium'] = self.annual_premium_scaler.transform(df3[['annual_premium']].values)

        # age
        df3['age'] = self.age_scaler.transform(df3[['age']].values)

        # vintage
        df3['vintage'] = self.vintage_scaler.transform(df3[['vintage']].values)

        # gender
        df3 = pd.get_dummies(df3, prefix='gender', prefix_sep='_', columns=['gender'])

        if 'gender_female' not in df3.columns:
            df3['gender_female'] = 0
        if 'gender_male' not in df3.columns:
            df3['gender_male'] = 0

        # vehicle_age
        df3 = pd.get_dummies(df3, prefix='vehicle_age', prefix_sep='_', columns=['vehicle_age'])

        if 'vehicle_age_below_1_year' not in df3.columns:
            df3['vehicle_age_below_1_year'] = 0
        if 'vehicle_age_between_1_2_year' not in df3.columns:
            df3['vehicle_age_between_1_2_year'] = 0
        if 'vehicle_age_over_2_years' not in df3.columns:
            df3['vehicle_age_over_2_years'] = 0

        # region_code
        df3.loc[:, 'region_code'] = df3['region_code'].map(self.region_code_scaler)

        # policy_sales_channel 
        df3.loc[:, 'policy_sales_channel'] = df3['policy_sales_channel'].map(self.policy_sales_channel_scaler)

        # make sure that dont have NA's
        df3 = df3.dropna()

        # selected columns on cycle 01
        selected_cols = ['vintage', 'annual_premium', 'age', 'region_code', 'vehicle_damage', 'policy_sales_channel', 'previously_insured',
                         'vehicle_age_below_1_year', 'vehicle_age_between_1_2_year', 'vehicle_age_over_2_years']
        
        return df3[selected_cols]

  
    def get_prediction(self, model, original_data, test_data):
        
        # model prediction
        pred = model.predict_proba(test_data)

        # join prediction into original data
        original_data['score'] = pred[:, 1].tolist()
        
        return original_data.to_json( orient='records', date_format='iso')