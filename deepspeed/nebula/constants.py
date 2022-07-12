"""
Copyright (c) Microsoft Corporation
Licensed under the MIT license.
"""

#########################################
# nebula
#########################################
# Nebula. By default, this feature is not enabled.
# Users can configure in ds_config.json as below example:
NEBULA_FORMAT = '''
nebula should be enabled as:
"session_params": {
  "nebula": {
        "enabled": true,
        "persistent_storage_path": "/foo/bar",
        "persistent_time_interval": 100,
        "num_of_version_in_retention": 2
    }
}
'''

NEBULA = "nebula"

NEBULA_ENABLED = "enabled"
NEBULA_ENABLED_DEFAULT = False

NEBULA_LOAD_PATH = "persistent_storage_path"
NEBULA_LOAD_PATH_DEFAULT = None

NEBULA_PERSISTENT_STORAGE_PATH = "persistent_storage_path"
NEBULA_PERSISTENT_STORAGE_PATH_DEFAULT = None

NEBULA_PERSISTENT_TIME_INTERVAL = "persistent_time_interval"
NEBULA_PERSISTENT_TIME_INTERVAL_DEFAULT = 100

NEBULA_NUM_OF_VERSION_IN_RETENTION = "num_of_version_in_retention"
NEBULA_NUM_OF_VERSION_IN_RETENTION_DEFAULT = 2

NEBULA_EXPORT_ENVS = [
    'DLTS_JOB_ID',
    'DLTS_NUM_WORKER',
    'NEBULA_PERSISTENT_STORAGE_PATH',
    'NEBULA_PERSISTENT_TIME_INTERVAL',
    'AML_RUN_ID',
    'AZUREML_RUN_TOKEN',
    'AZUREML_WORKSPACE_SCOPE',
    'AZUREML_EXPERIMENT_SCOPE',
    'AZUREML_RUN_HISTORY_SERVICE_ENDPOINT',
    'AZUREML_RUN_ID',
    'NEBULA_MEMORY_BUFFER_SIZE',
    'AZUREML_PARAMETER_ITPJOB_NAME',
    'FC_TASKROLE_NAME',
    'FC_TASK_INDEX',
    'MASTER_HOST',
    'LOCAL_HOST',
    'AZUREML_BLOB_ACCOUNT_NAME',
    'AZUREML_BLOB_ACCOUNT_KEY'
]
