import os
import json
import pickle
import joblib
import pandas as pd
import datetime
from flask import Flask, jsonify, request
from peewee import (
    Model, IntegerField, FloatField,
    TextField, IntegrityError
)
from playhouse.shortcuts import model_to_dict
from playhouse.db_url import connect


########################################
# Begin database stuff

# the connect function checks if there is a DATABASE_URL env var
# if it exists, it uses it to connect to a remote postgres db
# otherwise, it connects to a local sqlite db stored in predictions.db
DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')

class Prediction(Model):
    observation_id = TextField(unique=True)
    observation = TextField()
    proba = FloatField()
    true_class = IntegerField(null=True)

    class Meta:
        database = DB

DB.create_tables([Prediction], safe=True)

# End database stuff
########################################

########################################
# Unpickle the previously-trained model


with open('columns.json') as fh:
    columns = json.load(fh)

pipeline = joblib.load('pipeline.pickle')

with open('dtypes.pickle', 'rb') as fh:
    dtypes = pickle.load(fh)


# End model un-pickling
########################################


########################################
# Begin webserver stuff

app = Flask(__name__)

def check_request(request):
    """
        Validates that our request is well formatted
        
        Returns:
        - assertion value: True if request is ok, False otherwise
        - error message: empty if request is ok, False otherwise
    """
    
    if "observation_id" not in request:
        error = "Field `id` missing from request: {}".format(request)
        return False, error
    
    return True, ""

def check_valid_column(observation):
    """
        Validates that our observation only has valid columns
        
        Returns:
        - assertion value: True if all provided columns are valid, False otherwise
        - error message: empty if all provided columns are valid, False otherwise
    """
    
    valid_columns = {
        "Type",
        "Date",
        "Latitude",
        "Longitude",
        "Gender",
        "Age range",
        "Self-defined ethnicity",
        "Officer-defined ethnicity",
        "Legislation",
        "Object of search",
        "station",
    }
    
    keys = set(observation.keys())
    
    if len(valid_columns - keys) > 0: 
        missing = valid_columns - keys
        error = "Missing columns: {}".format(missing)
        return False, error

    return True, ""

def check_is_text(observation):

    valid_columns = [
        "Age range",
        "Gender",
        "Legislation",
        "Object of search",
        "Officer-defined ethnicity",
        "Self-defined ethnicity",
        "station",
        "Type"
    ]

    for col in valid_columns:
        if not isinstance(observation[col], str):
            error = "Column {} must be a string: {}".format(col)
            return False, error

    return True, ""

def check_longitude(observation):
    long = observation["Longitude"]
    if long < -180 or long > 180:
        error = "Longitude out of bounds"
        return False, error

    return True, ""

def check_latitude(observation):
    latt = observation["Latitude"]
    if latt < -90 or latt > 90:
        error = "Latitude out of bounds"
        return False, error

    return True, ""

def validate_date(observation):
    try:
        datetime.datetime.fromisoformat(observation["Date"])
    except Exception as e:
        error = "Invalid date"
        return False, error

    return True, ""

def check_categorical_values(observation):
    """
        Validates that all categorical fields are in the observation and values are valid
        
        Returns:
        - assertion value: True if all provided categorical columns contain valid values, 
                           False otherwise
        - error message: empty if all provided columns are valid, False otherwise
    """
    
    valid_category_map = {
        "Age range": ["25-34", "10-17", "18-24", "over 34", "under 10"],
        "Gender": ["Male", "Female", "Other"]
    }
    
    for key, valid_categories in valid_category_map.items():
        if key in observation:
            value = observation[key]
            if value not in valid_categories:
                error = "Invalid value provided for {}: {}. Allowed values are: {}".format(
                    key, value, ",".join(["'{}'".format(v) for v in valid_categories]))
                return False, error
        else:
            error = "Categorical field {} missing"
            return False, error

    return True, ""

@app.route('/predict', methods=['POST'])
def predict():
    obs_dict = request.get_json()
    try:
        request_ok, error = check_request(obs_dict)
        if not request_ok:
            response = {'error': error}
            return jsonify(response)

        _id = obs_dict['observation_id']
        observation = obs_dict
        del observation['observation_id']

        columns_ok, error = check_valid_column(observation)
        if not columns_ok:
            response = {'error': error}
            return jsonify(response)
            
        text_ok, error = check_is_text(observation)
        if not text_ok:
            response = {'error': error}
            return jsonify(response)

        categories_ok, error = check_categorical_values(observation)
        if not categories_ok:
            response = {'error': error}
            return jsonify(response)

        longitude_ok, error = check_longitude(observation)
        if not longitude_ok:
            response = {'error': error}
            return jsonify(response)

        latitude_ok, error = check_latitude(observation)
        if not latitude_ok:
            response = {'error': error}
            return jsonify(response)

        date_ok, error = validate_date(observation)
        if not date_ok:
            response = {'error': error}
            return jsonify(response)

        obs = pd.DataFrame([observation], columns=columns).astype(dtypes)
        proba = pipeline.predict_proba(obs)[0, 1]
        # prediction = pipeline.predict(obs)[0]
        prediction = proba > 0.7
        response = {'prediction': bool(prediction), 'observation_id': _id}
        p = Prediction(
            observation_id=_id,
            proba=proba,
            observation=request.data,
        )
        try:
            p.save()
        except IntegrityError:
            error_msg = "ERROR: Observation ID: '{}' already exists".format(_id)
            response["error"] = error_msg
            DB.rollback()
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': "Unknown error"})


@app.route('/update', methods=['POST'])
def update():
    obs = request.get_json()
    try:
        p = Prediction.get(Prediction.observation_id == obs['observation_id'])
        p.true_class = obs['label']
        p.save()
        return jsonify({
            'observation_id': obs['observation_id'],
            'label': obs['label']
        })
    except Prediction.DoesNotExist:
        error_msg = 'Observation ID: "{}" does not exist'.format(obs['id'])
        return jsonify({'error': error_msg})
    except Exception as e:
        return jsonify({'error': "Unknown error"})


@app.route('/list-db-contents')
def list_db_contents():
    return jsonify([
        model_to_dict(obs) for obs in Prediction.select()
    ])


# End webserver stuff
########################################

if __name__ == "__main__":
    app.run(debug=True, port=5000)
