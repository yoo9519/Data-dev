import pendulum
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator

#######################################
############### Set ENV ###############
#######################################
# local_tz = pendulum.timezone("Asia/Seoul")

default_args = {
    'owner': 'Clyne',
    'email': 'yoo9519@gmail.com',
    'retries': 3,
    'retry_delay': timedelta(minutes=10)
}

dag = DAG(
    dag_id='airflow_log_clear_v0001',
    default_args=default_args,
    start_date=datetime(2022, 6, 23),
    catchup=False,
    schedule_interval='@daily',  # Crontime : min hour day month week / 매일 02시에 삭제
    max_active_runs=3,
    tags=['operation']
)

########################################
############### DAG Task ###############
########################################
start = DummyOperator(
    dag = dag,
    task_id = 'Start'
)


del_log = BashOperator(
    task_id = 'clear_log',
    bash_command = 'find /opt/airflow/logs -type f -mtime +2 -delete',
    dag=dag
)


complete = DummyOperator(
    dag=dag,
    task_id = 'Complete'
)


##############################################
############### Task operation ###############
##############################################
start >> del_log >> complete