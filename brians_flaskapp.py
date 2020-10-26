#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Unused imports, wasn't sure if anyone still needs them
# from dateutil.relativedelta import relativedelta
# from sklearn.preprocessing import Imputer, OneHotEncoder
# import datetime
# import math
from flask import Flask, request, render_template, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
import os, pickle, random
import numpy as np
import pandas as pd
app = Flask(__name__)
req = request

#==============================================================================
# Globals
#==============================================================================
hippyStates = ['CA', 'NY', 'MA', 'CO', 'OR']
badTouchStates = ['TX', 'IN', 'AL', 'AK', 'WV', 'MS']

#==============================================================================
# Flask Code
#==============================================================================

@app.route('/check_fraud', methods=['POST', 'GET'])
def login():
    error = None
    return render_template('check_fraud.html', error=error)


@app.route('/results', methods=['POST', 'GET'])
def results():
    if req.method=='POST':
        lead_id = req.form['lead_id']
        lead_created = req.form['lead_created']
        source = req.form['source']
        state = req.form['state']
        switching = req.form['switching']
        company = req.form['company']

    return render_template(
        'results.html',
        lead_id=lead_id,
        lead_created=lead_created,
        source=source, state=state,
        switching=switching,
        company=company)

    # run_sample(createdDate, source, leadStates, switchers, coName)


@app.route('/api', methods=['POST', 'GET'])
def api():
    if req.method == 'POST':
        lead_id = req.form['lead_id']
        lead_created = req.form['lead_created']
        source = req.form['source']
        state = req.form['state']
        switching = req.form['switching']
        company = req.form['company']
    else:
        lead_id = req.args.get('lead_id')
        lead_created = req.args.get('lead_created')
        source = req.args.get('source')
        state = req.args.get('state')
        switching = req.args.get('switching')
        company = req.args.get('company')


    probability, prediction, status = run_sample(lead_created, source, state, switching, company)

    return jsonify(
        lead_id=lead_id,
        probability=probability,
        prediction=prediction,
        status=status)

# @app.route('/', methods=['POST', 'GET'])
# def hello():
#     error = None
#     return "Hello World"

#==============================================================================
# Modeling
#==============================================================================

def test_run():
    createdDate = 'asdf'
    source = 'Direct'
    leadStates = 'CA,NY'
    switchers = 'New Company'
    coName = 'Larson Group'
    return run_sample(createdDate, source, leadStates, switchers, coName)


def run_sample(createdDate, source, leadStates, switchers, coName):
    # From input data, apply same transformations as model
    switch = 1 if str.lower(switchers) == 'switching' \
    else 0 if str.lower(switchers) == 'new company' else random.random()

    multiState = True if len(leadStates.split(',')) > 1 else False
    llc = 1 if ' llc' in coName else 0
    inc = 1 if ' inc' in coName else 0
    threename = 1 if 'and ' in coName else 0
    hiState = True if leadStates in hippyStates else False
    btState = True if leadStates in badTouchStates else False

    sources = ["direct", "inbound referral", "seo", "adwords", "facebook"]
    channel_direct = 1 if str.lower(source) == sources[0] else 0
    channel_inbound = 1 if str.lower(source) == sources[1] else 0
    channel_seo = 1 if str.lower(source) == sources[2] else 0
    channel_adwords = 1 if str.lower(source) == sources[3] else 0
    channel_facebook = 1 if str.lower(source) == sources[4] else 0
    channel_other = 1 if str.lower(source) not in sources else 0

    testSample = [switch, multiState, llc, inc, threename, hiState, btState,
        channel_direct, channel_inbound, channel_seo, channel_adwords,
        channel_facebook, channel_other]

#==============================================================================
#   testing
#        testSample = finalTrain.ix[int(random.random()*999),1:]
#
#==============================================================================
#    testSample = ['switch','multiState','llc','inc',
#        '3name','hiState','btState','Channel=Direct',
#        'Channel=Inbound Referral', 'Channel=SEO', 'Channel=adwords',
#        'Channel=facebook', 'Channel=other']
    #If the model exists, unpickle, otherwise retrain a new model.

    if os.path.isfile('modelDump.p'):
        print("The model does exist")
        with open('modelDump.p', 'rb') as z:
            fstModel = pickle.load(z)
    else:
        fstModel = train_model()

    #testSample = [0.0, False, False, False, True, 0, 1, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    # Take the same decision trees and run it on the test data
    probability = fstModel.predict_proba(testSample)
#==============================================================================
#    testing
#    probability
#==============================================================================

    if probability[0][1] < 0.15:
        fakestatus = 'approve'
        prediction = "In fraud system, you'd be squeaky clean!"
    elif probability[0][1] < 0.35:
        fakestatus = 'pend'
        prediction = "In fraud system, we'd pend you and send you on to an agent"
    else:
        fakestatus = 'decline'
        prediction = "You're probably a fraud... sorry"

    return probability[0][1], prediction, fakestatus


def encode_onehot(df, cols):
    # stolen wholesale from : https://gist.github.com/ramhiser/982ce339d5f8c9a769a0
    vec = DictVectorizer()

    vec_data = pd.DataFrame(vec.fit_transform(df[cols].to_dict(outtype='records')).toarray())
    vec_data.columns = vec.get_feature_names()
    vec_data.index = df.index

    df = df.drop(cols, axis=1)
    df = df.join(vec_data)
    return df


def train_model():
    chooChoo = pd.read_csv('TrainingData/Training Data.csv')
    chooChoo['states'] = chooChoo['states from lead form']
    ## Data Transformations

    ## Impute Switchers
    chooChoo['switch'] = chooChoo.apply(lambda row: \
        1.0 if row['switching'] == 'Switching' else \
        0.0 if row['switching'] == 'New Company' else \
        np.nan, axis=1)

    impute = np.mean(chooChoo['switch'])

    ## Crazy imputing
    chooChoo['switch'] = chooChoo.apply(lambda row: \
        1.0 if np.isnan(row['switch']) and random.random() < impute
        else 0.0 if np.isnan(row['switch']) else row['switch'], axis=1)

    ## All NULLS to 'Unknown'
    chooChoo = chooChoo.fillna('Unknown')

    ## Multiple states
    chooChoo['multiState'] = [',' in x for x in chooChoo['states']]

    ## LLC
    chooChoo['llc'] = \
        [' llc' in str.lower(x) for x in chooChoo['company_name']]

    ## Inc
    chooChoo['inc'] = \
        [' inc' in str.lower(x) for x in chooChoo['company_name']]

    ## 3name
    chooChoo['3name'] = \
        ['and ' in str.lower(x) for x in chooChoo['company_name']]

    ## Other Channel
    chooChoo['Channel'] = chooChoo.apply(lambda row: \
        row['source'] if row['source'] in [\
        'adwords','Direct','facebook','Inbound Referral','SEO']\
    else 'other', axis=1)

    chooChoo['hiState'] = \
        [1 if any(x in asdf for x in hippyStates) else 0 \
        for asdf in chooChoo['states']]

    chooChoo['btState'] = \
        [1 if any(x in asdf for x in badTouchStates) else 0 \
        for asdf in chooChoo['states']]

    chooChoo = encode_onehot(chooChoo, cols=['Channel'])

    ## Train Model & Set up
    fst = RandomForestClassifier(n_estimators=1000)

    trainFields = ['30day_fraud', 'switch', 'multiState', 'llc', 'inc',
        '3name', 'hiState','btState', 'Channel=Direct',
        'Channel=Inbound Referral', 'Channel=SEO', 'Channel=adwords',
        'Channel=facebook', 'Channel=other']

    finalTrain = chooChoo[trainFields]

    trainedForest= fst.fit(finalTrain.ix[:, 1:], finalTrain['30day_fraud'])

    ## Save down a copy of model to pickle
    with open('modelDump.p', 'wb') as z:
        pickle.dump(trainedForest, z)

    ## for when have to reload, figure this shit out
    #trainedModel = pickle.load('modelDump.p')

    #debugging -> fstModel = trainedForest

    #return model
    return trainedForest


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
