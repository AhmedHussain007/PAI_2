import pandas as pd
from sklearn.preprocessing import MinMaxScaler , StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder


def le(data):
    le = LabelEncoder()
    categorical_columns = ['Gender',  'Preferred_Activities', 'Location']
    data[categorical_columns] = data[categorical_columns].apply(le.fit_transform)
    return data


def ohe(data):
    encoder = OneHotEncoder(sparse_output=False)  # Use sparse=False to get a dense array

    encoded_data = encoder.fit_transform(data[["Favorite_Season"]])

    encoded_df = pd.DataFrame(encoded_data, columns=encoder.categories_[0])
    encoded_df.head()
    data.drop(columns=["Favorite_Season"] , inplace=True)
    data = pd.concat([data, encoded_df], axis=1)
    return data

def oe(data):
    category_order = [['high school', 'bachelor', 'master', 'doctorate']]
    encoder = OrdinalEncoder(categories=category_order)
    data["Education_Level"] = encoder.fit_transform(data[["Education_Level"]])
    return data



def scaling(data):
    columns_to_normalize_using_minmax = ['Age', 'Income', 'Vacation_Budget']
    columns_to_normalize_using_z_score = [ 'Proximity_to_Mountains', 'Proximity_to_Beaches', 'Travel_Frequency']
    minmaxscaler = MinMaxScaler()
    standard_z_scaler = StandardScaler()

    data[columns_to_normalize_using_minmax] = minmaxscaler.fit_transform(data[columns_to_normalize_using_minmax])
    data[columns_to_normalize_using_z_score] = standard_z_scaler.fit_transform(data[columns_to_normalize_using_z_score])
    return data


def get_transformed_data():
    data = pd.read_csv('dataset.csv')
    data = data.rename(columns={'Preference': 'Output'})
    data = le(data)
    data = ohe(data)
    data = oe(data)
    data = scaling(data)
    return data