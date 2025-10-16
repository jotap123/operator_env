import json
import boto3

from botocore.exceptions import ClientError
from langchain_core.messages import HumanMessage, AIMessage


def call_model(version: str, messages: str):
    """
    Synchronously call Bedrock runtime with a concatenated prompt built from LangChain messages.
    Returns the assistant text output.
    """
    runtime = boto3.client('bedrock-runtime', region_name='us-east-2')
    
    models = {
        "lightweight": "qwen.qwen3-32b-v1:0",
        "medium": "meta.llama3-1-70b-instruct-v1:0",
        "heavy": "qwen.qwen3-235b-a22b-2507-v1:0",
    }
    model_id = models.get(version, "meta.llama3-1-70b-instruct-v1:0")

    bedrock_messages = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            bedrock_messages.append({
                "role": "user",
                "content": [{"type": "text", "text": msg.content}]
            })
        elif isinstance(msg, AIMessage):
            bedrock_messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": msg.content}]
            })

    native_request = {
        "prompt": bedrock_messages,
        "max_gen_len": 1024,
        "temperature": 0.1,
        "top_p": 0.5
    }

    request_body = json.dumps(native_request)

    try:
        # Invoke the model with the request.
        response = runtime.invoke_model(modelId=model_id, body=request_body)

    except (ClientError, Exception) as e:
        # Keep behavior visible to caller
        raise RuntimeError(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")

    # Decode the response body
    try:
        model_response = json.loads(response["body"].read())
    except Exception:
        # Fallback if body is already a string
        try:
            model_response = json.loads(response["body"])
        except Exception:
            model_response = response

    assistant_response = model_response["generation"].strip()

    return assistant_response
