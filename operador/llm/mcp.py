"""
AWS S3 MCP Server using FastMCP
Provides tools for reading files and listing objects in S3 buckets
"""
import io
import json
import base64
import boto3
import s3fs
import pandas as pd

from typing import Optional, Any
from mcp.server.fastmcp import FastMCP
from botocore.exceptions import ClientError, NoCredentialsError

# Initialize FastMCP server
mcp = FastMCP("s3-reader")
s3_client = boto3.client('s3')


@mcp.tool()
def list_buckets() -> str:
    """
    List all S3 buckets accessible to the current AWS credentials.
    
    Returns:
        JSON string containing list of bucket names and creation dates
    """
    try:
        response = s3_client.list_buckets()
        buckets = [
            {
                "name": bucket["Name"],
                "creation_date": bucket["CreationDate"].isoformat()
            }
            for bucket in response.get("Buckets", [])
        ]
        return json.dumps({"buckets": buckets, "count": len(buckets)}, indent=2)
    except NoCredentialsError:
        return json.dumps({"error": "AWS credentials not found. Please configure AWS credentials."})
    except ClientError as e:
        return json.dumps({"error": f"AWS error: {str(e)}"})
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"})


@mcp.tool()
def list_objects(bucket: str, prefix: str = "", max_keys: int = 100) -> str:
    """
    List objects in an S3 bucket with optional prefix filtering.
    
    Args:
        bucket: Name of the S3 bucket
        prefix: Optional prefix to filter objects (e.g., "folder/subfolder/")
        max_keys: Maximum number of objects to return (default: 100, max: 1000)
    
    Returns:
        JSON string containing list of objects with key, size, and last modified date
    """
    try:
        max_keys = min(max_keys, 1000)  # AWS limit
        
        params = {
            "Bucket": bucket,
            "MaxKeys": max_keys
        }
        if prefix:
            params["Prefix"] = prefix
        
        response = s3_client.list_objects_v2(**params)
        
        if "Contents" not in response:
            return json.dumps({
                "bucket": bucket,
                "prefix": prefix,
                "objects": [],
                "count": 0,
                "message": "No objects found"
            }, indent=2)
        
        objects = [
            {
                "key": obj["Key"],
                "size": obj["Size"],
                "last_modified": obj["LastModified"].isoformat(),
                "storage_class": obj.get("StorageClass", "STANDARD")
            }
            for obj in response["Contents"]
        ]
        
        result = {
            "bucket": bucket,
            "prefix": prefix,
            "objects": objects,
            "count": len(objects),
            "is_truncated": response.get("IsTruncated", False)
        }
        
        if result["is_truncated"]:
            result["message"] = "Results truncated. Use prefix or pagination to see more."
        
        return json.dumps(result, indent=2)
        
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        if error_code == "NoSuchBucket":
            return json.dumps({"error": f"Bucket '{bucket}' does not exist"})
        elif error_code == "AccessDenied":
            return json.dumps({"error": f"Access denied to bucket '{bucket}'"})
        return json.dumps({"error": f"AWS error: {str(e)}"})
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"})


@mcp.tool()
def read_text_file(bucket: str, key: str, encoding: str = "utf-8") -> str:
    """
    Read a text file from S3 and return its contents as a string.
    
    Args:
        bucket: Name of the S3 bucket
        key: Object key (file path) in the bucket
        encoding: Text encoding (default: utf-8)
    
    Returns:
        File contents as string or error message
    """
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        content = response["Body"].read().decode(encoding)
        
        metadata = {
            "bucket": bucket,
            "key": key,
            "size": response["ContentLength"],
            "content_type": response.get("ContentType", "unknown"),
            "last_modified": response["LastModified"].isoformat()
        }
        
        return json.dumps({
            "metadata": metadata,
            "content": content
        }, indent=2)
        
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        if error_code == "NoSuchKey":
            return json.dumps({"error": f"Object '{key}' not found in bucket '{bucket}'"})
        elif error_code == "NoSuchBucket":
            return json.dumps({"error": f"Bucket '{bucket}' does not exist"})
        elif error_code == "AccessDenied":
            return json.dumps({"error": f"Access denied to '{key}' in bucket '{bucket}'"})
        return json.dumps({"error": f"AWS error: {str(e)}"})
    except UnicodeDecodeError:
        return json.dumps({"error": f"Failed to decode file with encoding '{encoding}'. Try a different encoding or use read_binary_file."})
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"})


@mcp.tool()
def read_binary_file(bucket: str, key: str, return_base64: bool = True) -> str:
    """
    Read a binary file from S3.
    
    Args:
        bucket: Name of the S3 bucket
        key: Object key (file path) in the bucket
        return_base64: If True, return base64-encoded content (default: True)
    
    Returns:
        JSON string with file metadata and content (base64-encoded if return_base64=True)
    """
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        content_bytes = response["Body"].read()
        
        metadata = {
            "bucket": bucket,
            "key": key,
            "size": response["ContentLength"],
            "content_type": response.get("ContentType", "application/octet-stream"),
            "last_modified": response["LastModified"].isoformat()
        }
        
        if return_base64:
            content_b64 = base64.b64encode(content_bytes).decode("ascii")
            return json.dumps({
                "metadata": metadata,
                "content_base64": content_b64,
                "encoding": "base64"
            }, indent=2)
        else:
            return json.dumps({
                "metadata": metadata,
                "content_bytes": list(content_bytes[:1000]),  # First 1000 bytes as list
                "note": "Showing first 1000 bytes only. Use return_base64=True for full content."
            }, indent=2)
        
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        if error_code == "NoSuchKey":
            return json.dumps({"error": f"Object '{key}' not found in bucket '{bucket}'"})
        elif error_code == "NoSuchBucket":
            return json.dumps({"error": f"Bucket '{bucket}' does not exist"})
        elif error_code == "AccessDenied":
            return json.dumps({"error": f"Access denied to '{key}' in bucket '{bucket}'"})
        return json.dumps({"error": f"AWS error: {str(e)}"})
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"})


@mcp.tool()
def get_object_metadata(bucket: str, key: str) -> str:
    """
    Get metadata for an S3 object without downloading its contents.
    
    Args:
        bucket: Name of the S3 bucket
        key: Object key (file path) in the bucket
    
    Returns:
        JSON string with object metadata including size, content type, and custom metadata
    """
    try:
        response = s3_client.head_object(Bucket=bucket, Key=key)
        
        metadata = {
            "bucket": bucket,
            "key": key,
            "size": response["ContentLength"],
            "content_type": response.get("ContentType", "unknown"),
            "last_modified": response["LastModified"].isoformat(),
            "etag": response.get("ETag", "").strip('"'),
            "storage_class": response.get("StorageClass", "STANDARD"),
            "server_side_encryption": response.get("ServerSideEncryption"),
            "custom_metadata": response.get("Metadata", {})
        }
        
        return json.dumps(metadata, indent=2)
        
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        if error_code in ["NoSuchKey", "404"]:
            return json.dumps({"error": f"Object '{key}' not found in bucket '{bucket}'"})
        elif error_code == "NoSuchBucket":
            return json.dumps({"error": f"Bucket '{bucket}' does not exist"})
        elif error_code == "AccessDenied":
            return json.dumps({"error": f"Access denied to '{key}' in bucket '{bucket}'"})
        return json.dumps({"error": f"AWS error: {str(e)}"})
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"})


@mcp.tool()
def read_json_file(bucket: str, key: str) -> str:
    """
    Read and parse a JSON file from S3.
    
    Args:
        bucket: Name of the S3 bucket
        key: Object key (file path) in the bucket
    
    Returns:
        Parsed JSON content as a string
    """
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        content = response["Body"].read().decode("utf-8")
        parsed_json = json.loads(content)
        
        return json.dumps({
            "bucket": bucket,
            "key": key,
            "parsed_content": parsed_json
        }, indent=2)
        
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON file: {str(e)}"})
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        if error_code == "NoSuchKey":
            return json.dumps({"error": f"Object '{key}' not found in bucket '{bucket}'"})
        return json.dumps({"error": f"AWS error: {str(e)}"})
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"})


@mcp.tool()
def read_csv_file(
    bucket: str, 
    key: str, 
    max_rows: int = 1000,
    include_summary: bool = True
) -> str:
    """
    Read a CSV file from S3 and return its contents as JSON.
    
    Args:
        bucket: Name of the S3 bucket
        key: Object key (file path) in the bucket
        max_rows: Maximum number of rows to return (default: 1000)
        include_summary: Include summary statistics (default: True)
    
    Returns:
        JSON string with CSV data, column info, and optional summary statistics
    """
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        content = response["Body"].read()
        
        # Read CSV into pandas DataFrame
        df = pd.read_csv(io.BytesIO(content))
        
        # Limit rows if needed
        total_rows = len(df)
        df_limited = df.head(max_rows)
        
        result = {
            "bucket": bucket,
            "key": key,
            "total_rows": total_rows,
            "returned_rows": len(df_limited),
            "columns": list(df.columns),
            "data": df_limited.to_dict(orient="records")
        }

        if include_summary:
            summary = {
                "column_types": df.dtypes.astype(str).to_dict(),
                "null_counts": df.isnull().sum().to_dict(),
                "numeric_summary": {}
            }

            # Add statistics for numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                summary["numeric_summary"][col] = {
                    "min": float(df[col].min()) if not pd.isna(df[col].min()) else None,
                    "max": float(df[col].max()) if not pd.isna(df[col].max()) else None,
                    "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                    "median": float(df[col].median()) if not pd.isna(df[col].median()) else None
                }

            result["summary"] = summary

        if total_rows > max_rows:
            result["note"] = f"Showing first {max_rows} of {total_rows} rows"
        
        return json.dumps(result, indent=2)

    except pd.errors.EmptyDataError:
        return json.dumps({"error": "CSV file is empty"})
    except pd.errors.ParserError as e:
        return json.dumps({"error": f"Failed to parse CSV: {str(e)}"})
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        if error_code == "NoSuchKey":
            return json.dumps({"error": f"Object '{key}' not found in bucket '{bucket}'"})
        elif error_code == "NoSuchBucket":
            return json.dumps({"error": f"Bucket '{bucket}' does not exist"})
        return json.dumps({"error": f"AWS error: {str(e)}"})
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"})


@mcp.tool()
def read_parquet_file(
    bucket: str,
    cols: Optional[str] = None
) -> Any:
    """
    Read a Parquet file from S3 and return its contents as JSON.
    
    Args:
        bucket: Name of the S3 bucket
        key: Object key (file path) in the bucket
        max_rows: Maximum number of rows to return (default: 1000)
        include_summary: Include summary statistics (default: True)
        cols: Comma-separated list of columns to read (optional, reads all if not specified)
    
    Returns:
        JSON string with Parquet data, column info, and optional summary statistics
    """
    s3 = s3fs.S3FileSystem(anon=True)
    directory = "/".join(bucket.split("//")[1].split("/")[:-1])
    pattern = directory + "/*.parquet"
    files = s3.glob(pattern)
    
    try:
        if len(files) == 1:
                df = pd.read_parquet(bucket, columns=cols, storage_options={'anon': True})
        elif len(files) > 1:
                df = pd.concat([pd.read_parquet(f's3://{f}', storage_options={'anon': True}) for f in files], ignore_index=True)

        return df
    except ValueError as e:
        if "columns" in str(e).lower():
            return json.dumps({"error": f"Invalid column names: {str(e)}"})
        return json.dumps({"error": f"Failed to read Parquet file: {str(e)}"})
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        if error_code == "NoSuchBucket":
            return json.dumps({"error": f"Bucket '{bucket}' does not exist"})
        return json.dumps({"error": f"AWS error: {str(e)}"})
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"})


if __name__ == "__main__":
    # Run the MCP server
    mcp.run()