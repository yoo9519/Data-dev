# -*- coding: utf-8 -*-
import os
airflow_home = os.environ['AIRFLOW_HOME']

import sys
from datetime import datetime, timedelta
from airflow.models import DAG
from data.ops.s3_flag_operator import S3FlagOperator
from airflow.models import Variable
from airflow.operators.postgres_operator import PostgresOperator
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator
from data.ops.s3_key_match_sensor import S3KeyMatchSensor
from data.ops.mysql_to_s3_operator import MySQLToS3Operator
from airflow.contrib.hooks.aws_hook import AwsHook
import logging
import importlib
from data.utils.s3_utils import create_object
from io import StringIO
import boto3


importlib.reload(sys)

task_default_args = {
    'owner': 'airflow',  # your id
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2021, 10, 2),
    'depends_on_past': False,  # does not depend on past dag runs
    'email': ['yoo9519@gmail.com'], # notification to admins
    'email_on_retry': False,
    'email_on_failure': True,
    'execution_timeout': timedelta(hours=1)  # task execution timeout
}


dag_id = 'price_crawler_to_s3_v0001'  # format: <team_name>_<source>_to_<target>


dag = DAG(
    dag_id=dag_id,
    default_args=task_default_args,
    schedule_interval='0 15 * * *',  # get clarification
    concurrency=6,
    max_active_runs=1
)

s3_sensorflag_bucket = 'russo-mydata'
# s3_conn_id = 's3_dpds_de'
s3_bucket = 'russo-mydata'
jsonpaths_bucket = 'russo-mydata'


s3flag_match_string = "d_{{ execution_date.strftime('%Y%m%d') }}"
# s3flag_match_string = "h_{{ (execution_date + macros.timedelta(hours=1)).strftime('%Y%m%d_%H') }}"

s3_key = '/airflow/dags' + dag_id + "/{{ (execution_date + macros.timedelta(days=1)).strftime('%Y%m%d') }}"

# jsonpaths
s3_params = '/airflow/dags' + dag_id
# jsonpaths = airflow_home + """/dags/ods/ods_external/jsonpaths/public_api_exchange_market_pairs_jsonpaths.txt"""
# jsonpaths_key = 'staging/ods_external/jsonpaths/public_api_exchange_market_pairs/public_api_exchange_market_pairs_jsonpaths.txt'

logging.info('******************* Uploading jsonpaths file *******************')

s3_hook = AwsHook(aws_conn_id=s3_conn_id)
s3_client_type = s3_hook.get_client_type(client_type='s3', region_name='ap-northeast-1')
s3 = s3_client_type

with open(jsonpaths, 'rb') as data:
    s3.upload_fileobj(data, jsonpaths_bucket, jsonpaths_key)

####################################################################################################
####################################################################################################
# script = airflow_home + "/dags/datateam/script/python_public_api_exchange_market_pairs.py"
script = "/opt/airflow/dags/price_cralwer_v0001.py"


start = DummyOperator(
    task_id='start',
    dag=dag)

bash_task = BashOperator(
    task_id='crawler',
    params={"s3_conn_id": s3_conn_id},
    bash_command='python %s' %(script),
    dag=dag)

complete = S3FlagOperator(
    task_id='complete',
    s3_conn_id=s3_conn_id,
    execution_date=s3flag_match_string,
    dag=dag)


start >> bash_task >> complete
