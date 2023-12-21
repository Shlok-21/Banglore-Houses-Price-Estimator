import json,pickle, numpy as np

__locations = None
__data_columns = None
__model = None

def get_location_names():
    global __locations
    return __locations

def get_estimated_price(location,sqft,bhk,bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >=0:
        x[loc_index] =1

    a =  round(__model.predict([x])[0],2)
    return (a)

def load_saved_artifacts():
    global __locations
    global __data_columns
    global __model
    print("Loading the model...")
    with open ('./housing_project/server/artifacts/columns.json','r') as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]
    with open ('./housing_project/server/artifacts/model.pickle','rb') as f:
        __model = pickle.load(f) 
    print("Model loaded successfully...")
    
if __name__ == '__main__':
    load_saved_artifacts()
    
    


