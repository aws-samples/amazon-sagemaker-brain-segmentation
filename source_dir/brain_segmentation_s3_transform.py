from __future__ import absolute_import
import boto3
import base64
import json
import io
import os
import mxnet as mx
from mxnet import nd
import numpy as np
mx.test_utils.download("https://s3.amazonaws.com/sagemaker-png/png.py", "png.py")
import png

###############################
###     Hosting Code        ###
###############################

def push_to_s3(img, bucket, prefix):
    """
    A method for encoding an image array as png and pushing to S3

    Parameters
    ----------
    img : np.array
        Integer array representing image to be uploaded.
    bucket : str
        S3 Bucket to upload to.
    prefix : str
        Prefix to upload encoded image to (should be .png).
    """
    s3 = boto3.client('s3')
    png.from_array(img.astype(np.uint8), 'L').save('img.png')
    response = s3.put_object(
        Body=open('img.png', 'rb'),
        Bucket=bucket,
        Key=prefix
    )
    return

def download_from_s3(bucket, prefix):
    """
    A method for downloading object from s3

    Parameters
    ----------
    bucket : str
        S3 Bucket to download from.
    prefix : str
        Prefix to download from.
    """
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=bucket, Key=prefix)
    return response

def decode_response(response):
    """
    A method decoding raw image bytes from S3 call into mx.ndarray.

    Parameters
    ----------
    response : dict
        Dict of S3 get_object response.
    """
    data = response['Body'].read()
    b64_bytes = base64.b64encode(data)
    b64_string = b64_bytes.decode()
    return mx.image.imdecode(base64.b64decode(b64_string)).astype(np.float32)

def transform_fn(net, data, input_content_type, output_content_type):
    try:
        inp = json.loads(json.loads(data)[0])
        bucket = inp['bucket']
        prefix = inp['prefix']
        s3_response = download_from_s3(bucket, prefix)
        img = decode_response(s3_response)
        img = nd.expand_dims(nd.transpose(img, (2, 0, 1)), 0)
        img = nd.sum_axis(nd.array([[[[0.3]], [[0.59]], [[0.11]]]]) * img, 1, keepdims=True)
        batch = mx.io.DataBatch([img])
        net.forward(batch)
        raw_output = net.get_outputs()[0].asnumpy()
        mask = np.argmax(raw_output, axis=(1))[0].astype(np.uint8)
        output_prefix = os.path.join(
            'output', '/'.join(prefix.split('/')[1:]).split('.')[0] + '_MASK_PREDICTION.png')
        push_to_s3(mask, bucket, output_prefix)
        response = {'bucket': bucket, 'prefix': output_prefix}
    except Exception as e:
        response = {'Error': str(e)}
    return json.dumps(response), output_content_type