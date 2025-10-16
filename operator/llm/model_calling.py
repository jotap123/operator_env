import json
import boto3

from botocore.exceptions import ClientError
from langchain_core.messages import HumanMessage, AIMessage


def call_model(version: str, messages: str):
    runtime = boto3.client('bedrock-runtime', region_name='us-east-2')
    
    models = {
        "lightweight": "anthropic.claude-sonnet-4-5-20250929-v1:0",
        "medium": "anthropic.claude-sonnet-4-5-20250929-v1:0",
        "heavy": "anthropic.claude-sonnet-4-5-20250929-v1:0",
    }
    model_id = models.get(version, "anthropic.claude-sonnet-4-5-20250929-v1:0")

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
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "temperature": 0.1,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": bedrock_messages}],
            }
        ],
    }

    request = json.dumps(native_request)

    try:
        # Invoke the model with the request.
        response = runtime.invoke_model(modelId=model_id, body=request)

    except (ClientError, Exception) as e:
        print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
        exit(1)

    # Decode the response body.
    model_response = json.loads(response["body"].read())

    # Extract and print the response text.
    response_text = model_response["content"][0]["text"]
    
    return response_text
