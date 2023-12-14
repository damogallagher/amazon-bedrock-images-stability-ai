import json
import logging
import boto3
from botocore.exceptions import ClientError
import base64

logger = logging.getLogger(__name__)

bedrock_runtime_client = boto3.client('bedrock-runtime', region_name='us-east-1' )

 
def lambda_handler(event, context):

    if "queryStringParameters" not in event:
        return  {
        'statusCode': 400,
        'body': 'No query string parameters passed in'
        }

    if "prompt" not in event["queryStringParameters"]:
        return  {
        'statusCode': 400,
        'body': 'Prompt needs to be passed in the query string parameters'
        }

    query_string_parameters = event["queryStringParameters"]
    prompt = query_string_parameters["prompt"]

    seed = 0
    if "seed"  in query_string_parameters:
        seed = query_string_parameters["seed"]


    image_data = invoke_stable_diffusion(prompt, seed)

    print(f"image_data: {image_data}")
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Amazon Bedrock - Stability AI</title>
    </head>
    <body>
        <img src="{}" alt="Base64 Image">
    </body>
    </html>
    """.format(f"data:image/png;base64,{image_data}")

    # Return HTML content as the response
    return {
        'statusCode': 200,
        'body': html_content,
        'headers': {
            'Content-Type': 'text/html',
        },
    }

def invoke_stable_diffusion(prompt, seed, style_preset=None):
        """
        Invokes the Stability.ai Stable Diffusion XL model to create an image using
        the input provided in the request body.

        :param prompt: The prompt that you want Stable Diffusion  to use for image generation.
        :param seed: Random noise seed (omit this option or use 0 for a random seed)
        :param style_preset: Pass in a style preset to guide the image model towards a particular style.
        :return: Base64-encoded inference response from the model.
        """

        try:
            # The different model providers have individual request and response formats.
            # For the format, ranges, and available style_presets of Stable Diffusion models refer to:
            # https://platform.stability.ai/docs/api-reference#tag/v1generation

            body = {
                "text_prompts": [{"text": prompt}],
                "seed": seed,
                "cfg_scale": 10,
                "steps": 30,
            }

            if style_preset:
                body["style_preset"] = style_preset

            response = bedrock_runtime_client.invoke_model(
                modelId="stability.stable-diffusion-xl", body=json.dumps(body)
            )

            response_body = json.loads(response["body"].read())
            base64_image_data = response_body["artifacts"][0]["base64"]

            return base64_image_data

        except ClientError:
            logger.error("Couldn't invoke Stable Diffusion XL")
            raise