import json
import pandas as pd
import numpy as np
import requests


def lambda_handler(event, context):
    pandas_version = pd.__version__
    numpy_version = np.__version__

    # Ping nktstudios.com and get the status code
    response =  requests.get("https://nktstudios.com/")
    nktstudios_status = response.status_code

    # Construct the return statement
    return_statement = f"Pandas Version: {pandas_version}, Numpy Version: {numpy_version}, \
NKT Studios Request Status: {nktstudios_status}"

    # Set the return statement expected by AWS Lambda
    return {
        'statusCode': 200,
        'body': json.dumps(return_statement)
    }