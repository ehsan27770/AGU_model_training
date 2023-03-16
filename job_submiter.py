import yaml
import sys
import subprocess
import pickle

root = '/home/semansou/code/long_range_prediction/'

subprocess.call(f'python {root}configuration_generator.py', shell = True)

with open(root + 'to_do.pickle', 'rb') as f:
    to_do = pickle.load(f)

root_code = '/home/semansou/code/long_range_prediction/configurations/'
root_job = '/home/semansou/jobs'
for name in to_do:
    config_name = root_code + name
    submit_command = (f'sbatch --export=CONFIG={config_name} {root_job}/job_batch.run')
    #print(submit_command)
    exit_status = subprocess.call(submit_command, shell = True)
    if exit_status is 1:  # Check to make sure the job submitted
        print("Job {0} failed to submit".format(submit_command))

print("Done submitting jobs!")
#submit_command
