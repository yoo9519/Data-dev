# -*- coding: utf-8 -*-
"""
team: Data
* author: Cline
* created: 2021-07-14

"""
import os
airflow_home = os.environ['AIRFLOW_HOME']

import sys
from datetime import datetime, timedelta
from airflow.models import DAG
from airflow.models import Variable
from airflow.operators.postgres_operator import PostgresOperator
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.contrib.hooks.aws_hook import AwsHook
import logging
import importlib
from io import StringIO
import boto3


importlib.reload(sys)

task_default_args = {
    'owner': 'cline',  # your id
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2021, 7, 12),
    'depends_on_past': False,  # does not depend on past dag runs
    'email': 'Your email', # notification to admins
    'email_on_retry': False,
    'email_on_failure': True,
    'execution_timeout': timedelta(hours=1)  # task execution timeout
}


dag_id = 'Your Dag Name version'  # format: <team_name>_<source>_to_<target>


dag = DAG(
    dag_id='Your DAG Name'
    default_args=###,
    schedule_interval='2 0 * * *',  # get clarification crontab
    concurrency=6,
    max_active_runs=1
)

env = Variable.get("environment")
if(env == 'Security'):
    ### = 'Security'
    ### = 'Security'
    ### = 'Security'
    ### = 'Security'
elif(env == 'Security'):
    ### = 'Security'
    ### = 'Security'
    ### = 'Security'
    ### = 'Security'
elif(env == 'Security'):
    ### = 'Security'
    ### = 'Security'
    ### = 'Security'
elif(env == 'Security'):
    ### = 'Security'
    ### = 'Security'
    ### = 'Security'
else:
    raise Exception('Unknown Environment')

Data_Lake_flag_match_string = "d_{{ execution_date.strftime('%Y%m%d') }}"
# Data_Lakeflag_match_string = "h_{{ (execution_date + macros.timedelta(hours=1)).strftime('%Y%m%d_%H') }}"

Data_Lake_key = 'Schema' + dag_id + "/{{ (execution_date + macros.timedelta(days=1)).strftime('%Y%m%d') }}"

# jsonpaths
Data_Lake_params = 'Schema' + dag_id
jsonpaths = airflow_home + """../public_api_exchange_market_pairs.txt"""
jsonpaths_key = '../public_api_exchange_market_pairs.txt'

logging.info('******************* Uploading jsonpaths file *******************')

Data_Lake_hook = AwsHook(aws_conn_id='###')
Data_Lake_client_type = Data_Lake_hook.get_client_type(client_type='S3', region_name='###')
Data_Lake = ###

with open(jsonpaths, 'rb') as data:
    Data_Lake.upload_fileobj(###, jsonpaths_bucket, jsonpaths_key)

####################################################################################################
####################################################################################################
script = airflow_home + "Your DAG Path"

start = DummyOperator(
    task_id='start',
    dag=dag)

bash_task = BashOperator(
    task_id='Security',
    params={"Data_Lake_conn_id": ###},
    bash_command='python %s' %(script),
    dag=dag)

complete = Data_LakeFlagOperator(
    task_id='complete',
    Data_Lake_conn_id=###,
    execution_date=###,
    dag=dag)


start >> bash_task >> complete
