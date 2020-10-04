import argparse
import logging
import os
import json
import boto3
import subprocess
import sys
from urllib.parse import urlparse

os.system('pip install autogluon')
from autogluon import TabularPrediction as task
import pandas as pd 

logging.basicConfig(level=logging.DEBUG)
logging.info(subprocess.call('ls -lR /opt/ml/input'.split()))

    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train_dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--filename', type=str, default='train.csv')
    parser.add_argument('--target', type=str, default='churn_yn')
    parser.add_argument('--eval_metric', type=str, default='f1')
    parser.add_argument('--presets', type=str, default='best_quality')    

    parser.add_argument('--debug', type=str, default=False)    

    parser.add_argument('--current_host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))

    parser.add_argument('--s3_output', type=str, default='s3://autogluon-test/results')
    parser.add_argument('--training_job_name', type=str, default=json.loads(os.environ['SM_TRAINING_ENV'])['job_name'])

    return parser.parse_args()


# ------------------------------------------------------------ #
# Hosting methods                                              #
# ------------------------------------------------------------ #

def model_fn(model_dir):
    """
    Load the gluon model. Called once when hosting service starts.
    :param: model_dir The directory where model files are stored.
    :return: a model (in this case an AutoGluon network)
    """
    net = task.load(model_dir)
    return net


def transform_fn(net, data, input_content_type, output_content_type):
    """
    Transform a request using the Gluon model. Called once per request.
    :param net: The AutoGluon model.
    :param data: The request payload.
    :param input_content_type: The request content type.
    :param output_content_type: The (desired) response content type.
    :return: response payload and content type.
    """
    data = json.loads(data)
    df_parsed = pd.DataFrame(data)
    prediction = net.predict(df_parsed)
    response_body = json.dumps(prediction.tolist())
    return response_body, output_content_type


def train(args):
    
    # SageMaker passes num_cpus, num_gpus and other args we can use to tailor training to
    # the current container environment, but here we just use simple cpu context.

    model_dir = args.model_dir    
    train_dir = args.train_dir
    filename = args.filename
    target = args.target    
    debug = args.debug
    eval_metric = args.eval_metric   
    presets = args.presets    
    
    num_gpus = int(os.environ['SM_NUM_GPUS'])
    current_host = args.current_host
    hosts = args.hosts
     
    logging.info(train_dir)
    
    train_data = task.Dataset(file_path=os.path.join(train_dir, filename))
    if debug:
        subsample_size = 500  # subsample subset of data for faster demo, try setting this to much larger values
        train_data = train_data.sample(n=subsample_size, random_state=0)
    predictor = task.fit(train_data = train_data, label=target, 
        output_directory=model_dir, eval_metric=eval_metric, presets=presets)

    return predictor


def evaluate(predictor, args):
    
    train_dir = args.train_dir
    train_file = args.filename
    test_file = train_file.replace('train', 'test', 1)
    target = args.target
    training_job_name = args.training_job_name
    s3_output = args.s3_output

    dataset_name = train_file.split('_')[0]
    logging.info(dataset_name)
    
    test_data = task.Dataset(file_path=os.path.join(train_dir, test_file))   
    
    u = urlparse(s3_output, allow_fragments=False)
    bucket = u.netloc
    logging.info(bucket)
    prefix = u.path.strip('/')
    logging.info(prefix)
    s3 = boto3.client('s3')
    
    y_test = test_data[target]
    test_data_nolab = test_data.drop(labels=[target], axis=1)

    y_pred = predictor.predict(test_data_nolab)
    y_pred_df = pd.DataFrame.from_dict({'True': y_test, 'Predicted': y_pred})
    pred_file = f'{dataset_name}_test_predictions.csv'
    y_pred_df.to_csv(pred_file, index=False, header=True)

    leaderboard = predictor.leaderboard()
    lead_file = f'{dataset_name}_leaderboard.csv'
    leaderboard.to_csv(lead_file)
    
    perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)
    del perf['confusion_matrix']
    perf_file = f'{dataset_name}_model_performance.txt'
    with open(perf_file, 'w') as f:
        print(json.dumps(perf, indent=4), file=f)

    summary = predictor.fit_summary()
    summ_file = f'{dataset_name}_fit_summary.txt'
    with open(summ_file, 'w') as f:
        print(summary, file=f)

    files_to_upload = [pred_file, lead_file, perf_file, summ_file]  
    for file in files_to_upload:
        s3.upload_file(file, bucket, os.path.join(prefix, training_job_name.replace('mxnet-training', 'autogluon', 1), file))
        
        
if __name__ == '__main__':
    args = parse_args()
    predictor = train(args)
    evaluate(predictor, args)