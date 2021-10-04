# -*- coding: utf-8 -*-
"""
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
    'owner': 'jeonghyun',
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2021, 10, 1),
    'depends_on_past': False,
    'email': '[yoo9519@gmail.com]',
    'email_on_retry': False,
    'email_on_failure': True,
    'execution_timeout': timedelta(hours=1)
}


dag_id = 'price_crawler_v0001'


dag = DAG(
    dag_id='price_crawler_v0001',
    default_args=task_default_args,
    schedule_interval='2 0 * * *',
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

s3_flag_match_string = "d_{{ execution_date.strftime('%Y%m%d') }}"
# s3flag_match_string = "h_{{ (execution_date + macros.timedelta(hours=1)).strftime('%Y%m%d_%H') }}"

s3_key = dag_id + "/{{ (execution_date + macros.timedelta(days=1)).strftime('%Y%m%d') }}"

# jsonpaths
s3_params = dag_id
jsonpaths =  dag_id + """../price_crawler_jsonpath.txt"""
jsonpaths_key = '../price_crawler_jsonpath.txt'

logging.info('******************* Uploading jsonpaths file *******************')

s3_hook = AwsHook(aws_conn_id='###')
s3_client_type = s3_hook.get_client_type(client_type='S3', region_name='###')
s3 = ###

with open(jsonpaths, 'rb') as data:
    s3.upload_fileobj(###, jsonpaths_bucket, jsonpaths_key)

####################################################################################################
####################################################################################################
script = airflow_home + "/dags/script/price_cralwer.py"

start = DummyOperator(
    task_id='start',
    dag=dag)

bash_task = BashOperator(
    task_id='cralwer',
    params={"s3_conn_id": },
    bash_command='python %s' %(script),
    dag=dag)

complete = s3FlagOperator(
    task_id='complete',
    s3_conn_id= ,
    execution_date= ,
    dag=dag)


start >> bash_task >> complete