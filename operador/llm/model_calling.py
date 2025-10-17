import json
import boto3

from botocore.exceptions import ClientError
from langchain_core.messages import HumanMessage, AIMessage


def call_model(version: str, system_prompt: str, messages: str):
    """
    Synchronously call Bedrock runtime with a concatenated prompt built from LangChain messages.
    Returns the assistant text output.
    """
    runtime = boto3.client('bedrock-runtime', region_name='us-east-2')
    
    models = {
        "lightweight": "qwen.qwen3-32b-v1:0",
        "medium": "qwen.qwen3-235b-a22b-2507-v1:0",
        "heavy": "qwen.qwen3-coder-480b-a35b-v1:0",
    }
    model_id = models.get(version, "qwen.qwen3-235b-a22b-2507-v1:0")

    bedrock_messages = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            bedrock_messages.append({
                "role": "user",
                "content": [{"text": msg.content}]
            })
        elif isinstance(msg, AIMessage):
            bedrock_messages.append({
                "role": "assistant",
                "content": [{"text": msg.content}]
            })

    try:
        response = runtime.converse_stream(
            modelId=model_id,
            system=[{"text": system_prompt}],
            messages=bedrock_messages,
            inferenceConfig={
                "maxTokens": 1024,
                "temperature": 0.1,
                "topP": 0.5
            }
        )
    
        # Process streaming response
        full_response = ""
        for event in response['stream']:
            if 'contentBlockDelta' in event:
                delta = event['contentBlockDelta']['delta']
                if 'text' in delta:
                    chunk = delta['text']
                    print(chunk, end='', flush=True)
                    full_response += chunk

        return full_response

    except (ClientError, Exception) as e:
        # Keep behavior visible to caller
        raise RuntimeError(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
