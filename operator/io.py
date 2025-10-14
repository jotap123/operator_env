import re
import numpy as np
import boto3
from botocore.config import Config

DEFAULT_S3_CONFIG = {}
DEFAULT_BOTO3_SESSION_KWARGS = {}
DEFAULT_LIST_KWARGS = {}

def create_s3_client(
    config_kwargs=DEFAULT_S3_CONFIG,
    session_kwargs=DEFAULT_BOTO3_SESSION_KWARGS,
):
    """
    Creates the S3 client connection
    Args:
        uri (str): S3 URL (s3://bucket-name/path/to/file)
        config_kwargs (dict): dict with boto3 Config params (e.g., region_name, signature_version)
        session_kwargs (dict): dict with boto3 Session params (e.g., profile_name, aws_access_key_id)
    Returns:
        object: boto3 S3 client
    """
    session = boto3.Session(**session_kwargs)
    config = Config(**config_kwargs) if config_kwargs else None
    s3_client = session.client('s3', config=config)
    return s3_client


def glob(uri: str, list_kwargs=DEFAULT_LIST_KWARGS, **kwargs):
    """
    Function that allows getting all files in a given S3 directory.
    Args:
        uri (str): S3 URL with regex pattern (s3://bucket-name/path/pattern)
            Supports:
                - '*' gets any string between the last character right before it
                and the first right after.
                - (word1|word2) allows filtering str with either word1 or word2.
        list_kwargs (dict): additional arguments for list_objects_v2
        **kwargs: additional connection kwargs passed to create_s3_client
    Returns:
        list: List with all S3 URIs that match the given pattern.
    """
    s3_client = create_s3_client(**kwargs)
    
    # Parse S3 URI
    uri_parts = uri.replace("s3://", "").split("/", 1)
    bucket_name = uri_parts[0]
    prefix_with_pattern = uri_parts[1] if len(uri_parts) > 1 else ""
    
    # Split the path into existing prefix and pattern suffix
    lista = re.split("^(.*?[\*])", prefix_with_pattern)
    if len(lista) == 1:
        # No wildcard, use full path as prefix
        prefix = lista[0]
        path_suffix = ""
    else:
        # Has wildcard, split into prefix and pattern
        new_split = "/".join(lista[:-1]).split("/")
        prefix = "/".join(new_split[:-1])
        path_suffix = new_split[-1] + lista[-1]
    
    # List all objects with the prefix
    list_objects = []
    paginator = s3_client.get_paginator('list_objects_v2')
    
    page_iterator = paginator.paginate(
        Bucket=bucket_name,
        Prefix=prefix,
        **list_kwargs
    )
    
    for page in page_iterator:
        if 'Contents' in page:
            for obj in page['Contents']:
                # Only include files, not directories (keys ending with /)
                if not obj['Key'].endswith('/'):
                    list_objects.append(f"s3://{bucket_name}/{obj['Key']}")
    
    result_list = []
    if len(list_objects) == 0:
        print("No file match the specified criteria")
        return result_list
    
    if len(path_suffix) == 0:
        return list_objects
    else:
        # Apply regex pattern filtering
        path_suffix = build_re(path_suffix)
        result_list = [
            i for i in np.array(list_objects) if re.search(path_suffix, i) is not None
        ]
    
    return result_list


def build_re(pattern: str) -> str:
    """
    Helper function to build regex pattern from wildcard pattern.
    You'll need to implement this based on your Azure version.
    Basic implementation:
    """
    # Escape special regex characters except * and ()
    pattern = pattern.replace('.', r'\.')
    # Convert * to regex .*
    pattern = pattern.replace('*', '.*')
    return pattern
