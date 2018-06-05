from __future__ import absolute_import
import boto3
import base64
import json
import io
import os
import mxnet as mx
from mxnet import ndarray as F
import numpy as np
import urllib
urllib.urlretrieve ("https://raw.githubusercontent.com/drj11/pypng/master/code/png.py", "png.py")
import png

###############################
###     Hosting Code        ###
###############################

def push_to_s3(img, bucket, prefix):

    s3 = boto3.client('s3')
    png.from_array(img.astype(np.uint8), 'L').save('img.png')
    response = s3.put_object(
                         Body=open('img.png', 'rb'),
                         Bucket=bucket,
                         Key=prefix
    )
    return 


def download_from_s3(bucket, prefix):
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=bucket,
                             Key=prefix)
    return response

def decode_response(response):
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
        img = F.expand_dims(F.transpose(img, (2, 0, 1)), 0)
        img = F.sum_axis(F.array([[[[0.3]], [[0.59]],[[0.11]]]]) * img, 1, keepdims=True)
        batch = mx.io.DataBatch([img])
        net.forward(batch)
        raw_output = net.get_outputs()[0].asnumpy()
        mask = np.argmax(raw_output, axis=(1))[0].astype(np.uint8)
        output_prefix = os.path.join('output', '/'.join(prefix.split('/')[1:]).split('.')[0]+'_MASK_PREDICTION.png')
        push_to_s3(mask, bucket, output_prefix)
        response = {'bucket':bucket, 'prefix':output_prefix}
    except Exception as e:
        response = {'Error':str(e)}
    return json.dumps(response), output_content_type