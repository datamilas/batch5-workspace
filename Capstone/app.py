import os
import json
import pickle
import joblib
import pandas as pd
import numpy as np
import datetime
from flask import Flask, jsonify, request
from peewee import (
    Model, IntegerField, FloatField,
    TextField, IntegrityError
)
from playhouse.shortcuts import model_to_dict
from playhouse.db_url import connect




threshold = 0.5



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

app.config['JSON_SORT_KEYS'] = False

valid_columns = {
        "patient_id",
        "race",
        "gender",
        "age",
        "weight",
        "admission_type_code",
        "discharge_disposition_code",
        "admission_source_code",
        "time_in_hospital",
        "payer_code",
        "medical_specialty",
        "has_prosthesis",
        "complete_vaccination_status",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "diag_1",
        "diag_2",
        "diag_3",
        "number_diagnoses",
        "blood_type",
        "hemoglobin_level",
        "blood_transfusion",
        "max_glu_serum",
        "A1Cresult",
        "diuretics",
        "insulin",
        "change",
        "diabetesMed",
}



def check_request(request):
    print(request)
    """
        Validates that our request is well formatted
        
        Returns:
        - assertion value: True if request is ok, False otherwise
        - error message: empty if request is ok, False otherwise
    """
    
    if "admission_id" not in request:
        error = "Field `admission_id` missing from request: {}".format(request)
        return False, error
    
    return True, ""

def check_valid_column(observation):
    """
        Validates that our observation only has valid columns
        
        Returns:
        - assertion value: True if all provided columns are valid, False otherwise
        - error message: empty if all provided columns are valid, False otherwise
    """
    

    
    keys = set(observation.keys())
    
    if len(valid_columns - keys) > 0: 
        missing = valid_columns - keys
        error = "Missing columns: {}".format(missing)
        return False, error

    return True, ""




def diagnosis_decoder(code):
    if "V" in str(code): 
        return "External causes of injury and supplemental classification"
    elif "E" in str(code):
        return "External causes of injury and supplemental classification"
    else:
        try:
        
            code = int(code)
            if code<140: return "infectious and parasitic diseases"
            if code<240: return "neoplasms"
            if code<280: return "endocrine, nutritional and metabolic diseases, and immunity disorders"
            if code<290: return "diseases of the blood and blood-forming organs"
            if code<320: return "mental disorders"
            if code<390: return "diseases of the nervous system and sense organs"
            if code<460: return "diseases of the circulatory system"
            if code<520: return "diseases of the respiratory system"
            if code<580: return "diseases of the digestive system"
            if code<630: return "diseases of the genitourinary system"
            if code<680: return "complications of pregnancy, childbirth, and the puerperium"
            if code<710: return "diseases of the skin and subcutaneous tissue"
            if code<740: return "diseases of the musculoskeletal system and connective tissue"
            if code<760: return "congenital anomalies"
            if code<780: return "certain conditions originating in the perinatal period"
            if code<800: return "symptoms, signs, and ill-defined conditions"
            if code<1000: return "injury and poisoning"
        except:
            return(np.nan)
        



def check_categorical_data(data, column_name, allowed_categories):
    data[column_name]= np.where(data[column_name].isin(allowed_categories), data[column_name], np.nan)




def prepare_data(data, columns):

    data.replace("?", np.nan, inplace=True)


    #clean race data
    data.race = data.race.astype(str)
    data.race = data.race.str.capitalize()
    data.race = data.race.replace(["Euro", "European", "White"], "Caucasian")
    data.race = data.race.replace("Latino", "Hispanic")
    data.race = data.race.replace(["Africanamerican", "Afro american", "African american", "Black"], "African American")
    column_name = "race"
    allowed_categories = ["Caucasian", "Hispanic", "African American", "Asian", "Other"]
    check_categorical_data(data, column_name, allowed_categories)


    #clean gender data
    data.gender = data.gender.astype(str)
    data.gener = data.gender.str.capitalize()
    data.gender.replace("Unknown/invalid", np.nan, inplace=True)

    column_name = "gender"
    allowed_categories = ["Male", "Female"]
    check_categorical_data(data, column_name, allowed_categories)


    #check age data
    data.age = data.age.astype(str)
    column_name = "age"
    allowed_categories = ['[50-60)', '[80-90)', '[60-70)', '[70-80)', '[40-50)', '[30-40)',
        '[90-100)', '[20-30)', '[10-20)', '[0-10)']
    check_categorical_data(data, column_name, allowed_categories)



    #clean admission_type_code 
    data.admission_type_code = data.admission_type_code.replace([5., 6., 8.], np.nan)
    column_name = "admission_type_code"
    allowed_categories = [1., 2., 3., 4., 5., 6., 7., 8.]
    check_categorical_data(data, column_name, allowed_categories)
    #keep only common values for admission_type_code, set others as "Other"
    common_categories = [1.0, 3.0, 2.0, np.nan]
    data[column_name] = np.where(data[column_name].isin(common_categories), data[column_name], 'Other')
    data[column_name] = data[column_name].astype(str)
    data[column_name] = data[column_name].replace("nan", np.nan)


    #discharge_disposition_code
    data.discharge_disposition_code = data.discharge_disposition_code.replace([18., 25., 26.], np.nan)
    column_name = "discharge_disposition_code"
    allowed_categories = list(np.arange(1.0, 30.0))
    check_categorical_data(data, column_name, allowed_categories)
    #keep only common values for dischage_disposition_code, set others as "Other"
    
    common_categories = [1.0, 3.0, 6.0, 2.0, 22.0, 5.0, 4.0, 7.0, 23.0, 28.0, 11.0, 13.0, 14.0, 19.0, 20.0, 21.0, np.nan]
    data[column_name] = np.where(data[column_name].isin(common_categories), data[column_name], 'Other')
    data[column_name] = data[column_name].astype(str)
    data[column_name] = data[column_name].replace("nan", np.nan)


    #admission_source_code
    column_name = "admission_source_code"
    allowed_categories = list(np.arange(1, 27))
    check_categorical_data(data, column_name, allowed_categories)

    data.admission_source_code = data.admission_source_code.replace([9, 15, 17, 20, 21], np.nan)
    #keep only common values for admission_source_code, set others as "Other"
    common_categories = [7.0, 1.0, 4.0, 6.0, 2.0, 5.0, 3.0, np.nan]
    data[column_name] = np.where(data[column_name].isin(common_categories), data[column_name], 'Other')
    data[column_name] = data[column_name].astype(str)
    data[column_name] = data[column_name].replace("nan", np.nan)





    #check vaccination status
    column_name = "complete_vaccination_status"
    data[column_name] = data[column_name].astype(str)
    data[column_name] = data[column_name].str.capitalize()
    allowed_categories = ['Complete', 'Incomplete', 'None']
    check_categorical_data(data, column_name, allowed_categories)



    #check bools
    for column in ["has_prosthesis", "blood_transfusion"]:
        if not isinstance(data[column][0], bool): data[column][0]=np.nan


    #check floats:
    for column in ["num_lab_procedures","num_medications", "hemoglobin_level"]:
        if not isinstance(data[column][0], float): 
            if not isinstance(data[column][0], int): data[column][0]=np.nan


    #check integers:
    for column in ["time_in_hospital", "num_procedures", "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses"]:
        if not isinstance(data[column][0], int): data[column][0]=np.nan

 
    #simplify diagnosis codes
    diag_columns = ['diag_1','diag_2','diag_3']
    for col in diag_columns:
        try:
            data[col] = data[col].astype(str)
            data[f"{col}_simplified"] = data[col].str.replace(r"\.(.*)", "")  #remove any numbers that come after .
            data[f"{col}_simplified"] = data.apply(lambda row: diagnosis_decoder(row[f"{col}_simplified"]),axis=1)
        except: data[f"{col}_simplified"] = np.nan



    #max_glu_serum
    column_name = "max_glu_serum"
    data[column_name] = data[column_name].astype(str)
    data[column_name] = data[column_name].str.capitalize()

    allowed_categories = ['None', '>300', '>200', 'Norm']
    check_categorical_data(data, column_name, allowed_categories)





    #A1Cresult
    column_name = "A1Cresult"
    data[column_name] = data[column_name].astype(str)
    data[column_name] = data[column_name].str.capitalize()
    allowed_categories = ['None', '>7', '>8', 'Norm']
    check_categorical_data(data, column_name, allowed_categories)


    #diuretics
    column_name = "diuretics"
    data[column_name] = data[column_name].astype(str)
    data[column_name] = data[column_name].str.capitalize()
    allowed_categories = ['No', 'Yes']
    check_categorical_data(data, column_name, allowed_categories)


    #insulin
    column_name = "insulin"
    data[column_name] = data[column_name].astype(str)
    data[column_name] = data[column_name].str.capitalize()
    allowed_categories = ['No', 'Yes']
    check_categorical_data(data, column_name, allowed_categories)


    #change
    column_name = "change"
    data[column_name] = data[column_name].astype(str)
    data[column_name] = data[column_name].str.capitalize()
    allowed_categories = ['No', 'Ch']
    check_categorical_data(data, column_name, allowed_categories)


    #diabetesMed
    column_name = "diabetesMed"
    data[column_name] = data[column_name].astype(str)
    data[column_name] = data[column_name].str.capitalize()
    allowed_categories = ['No', 'Yes']
    check_categorical_data(data, column_name, allowed_categories)




    #code age groups as integers
    data["age_as_int"] = data.age.replace(['[50-60)', '[80-90)', '[60-70)', '[70-80)', '[40-50)', '[30-40)',
    '[90-100)', '[20-30)', '[10-20)', '[0-10)'], [50, 80, 60, 70, 40, 30, 90, 20, 10, 0])

    
    


    
    return data[columns]






@app.route('/predict', methods=['POST'])
def predict():
    obs_dict = request.get_json()
    print(obs_dict)
    
    try:
        request_ok, error = check_request(obs_dict)
        if not request_ok:
            response = {'error': error}
            return jsonify(response)

        _id = obs_dict['admission_id']
        observation = obs_dict
        del observation['admission_id']

        columns_ok, error = check_valid_column(observation)
        if not columns_ok:
            response = {'error': error}
            return jsonify(response)




        obs = pd.DataFrame([observation], columns=valid_columns)
        
        
        obs = prepare_data(obs, columns).astype(dtypes)
        


        if obs.discharge_disposition_code[0] in ["11.0", "13.0", "14.0", "19.0", "20.0", "21.0"]:  #id patient is dead or in hospice predict False
            proba = 0

        else:
            proba = pipeline.predict_proba(obs)[0, 1]
            # prediction = pipeline.predict(obs)[0]
            

        p = Prediction(
                observation_id=_id,
                proba=proba,
                observation=request.data,
        )

        prediction = "Yes" if  (proba > threshold) else "No"

        response = {'readmitted': prediction}

        try:
            p.save()

        except IntegrityError:
            error_msg = "ERROR: Admission ID: '{}' already exists".format(_id)
            response["error"] = error_msg
            DB.rollback()

        except Exception as e:
            DB.rollback()
            return jsonify({'error': str(e)})
        
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/update', methods=['POST'])
def update():
    obs = request.get_json()
    
    try:
        p = Prediction.get(Prediction.observation_id == obs['admission_id'])
        if obs['readmitted'] in ["Yes", "No"]:
            p.true_class = True if obs['readmitted'] == "Yes" else False
            p.save()
            return jsonify({
                'admission_id': obs['admission_id'],
                'actual_readmitted': obs['readmitted'],
                'predicted_readmitted': "Yes" if  (p.proba > threshold) else "No"
            })
        else: 
            DB.rollback()
            return jsonify({'error': "value of true label can be either Yes or No"})
            
    except Prediction.DoesNotExist:
        error_msg = 'Admission ID: "{}" does not exist'.format(obs['admission_id'])
        DB.rollback()

        return jsonify({'error': error_msg})
    except Exception as e:
        DB.rollback()
        return jsonify({'error': str(e)})


@app.route('/list-db-contents')
def list_db_contents():
    return jsonify([
        model_to_dict(obs) for obs in Prediction.select()
    ])


# End webserver stuff
########################################

if __name__ == "__main__":
    app.run(debug=True, port=5000)
