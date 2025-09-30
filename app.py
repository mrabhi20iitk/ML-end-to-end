from flask import Flask, request, jsonify
import numpy as np
import joblib


app = Flask(__name__)

model = joblib.load("regression_model.pkl")

make = sorted(['Volkswagen', 'Lexus', 'Subaru', 'Cadillac', 'Toyota',
       'Land Rover', 'Mazda', 'Ram', 'Chrysler', 'GMC', 'Volvo', 'Audi',
       'Chevrolet', 'Tesla', 'Hyundai', 'Ford', 'Porsche', 'Acura',
       'Nissan', 'Kia', 'Jeep', 'BMW', 'Dodge', 'Mercedes-Benz', 'Honda'])

fuel_type = sorted(['Electric', 'Gasoline', 'Diesel'])
transmission = sorted(['Manual', 'Automatic'])
drivetrain = sorted(['RWD', 'FWD', 'AWD'])
accident_history = sorted(['None', 'Minor', 'Major'])
seller_type = sorted(['Dealer', 'Private'])
condition = sorted(['Excellent', 'Good', 'Fair'])


@app.route('/predict',methods=['POST','GET'])
def predict():
    data = request.get_json()
    features_dict = data['features']
    base_features = [features_dict['engine_hp'],
                     features_dict['owner_count'],
                     features_dict['vehicle_age'],
                     features_dict['mileage_per_year'],
                     features_dict['brand_popularity']
                     ]
     
    make_vector = [1 if features_dict['make'] == i else 0 for i in make]
    fuel_type_vector = [1 if features_dict['fuel_type'] == i else 0 for i in fuel_type]
    
    transmission_vector = [1 if features_dict['transmission'] == i else 0 for i in transmission]
    drivetrain_vector = [1 if features_dict['drivetrain'] == i else 0 for i in drivetrain]
    accident_history_vector = [1 if features_dict['accident_history'] == i else 0 for i in accident_history]
    seller_type_vector = [1 if features_dict['seller_type'] == i else 0 for i in seller_type]
    condition_vector = [1 if features_dict['condition']==i else 0 for i in condition]

    final_features = np.array(base_features + make_vector + transmission_vector + fuel_type_vector +  drivetrain_vector 
                             + accident_history_vector 
                             + seller_type_vector + condition_vector).reshape(1,-1)
    
    prediction = model.predict(final_features)

    return jsonify({
        "prediction" : float(prediction[0])
    })
    


if __name__ == "__main__":
    app.run(debug=True)
