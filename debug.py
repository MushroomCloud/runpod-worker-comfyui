import os
import time
import requests
import json
import base64
import uuid
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configuration
BASE_URI = "https://c04a8lr5o44v67-3000.proxy.runpod.net"
TIMEOUT = 600
LOG_LEVEL = 'INFO'
PAYLOAD_PATH = 'comfyui-20250424b.json'
VOLUME_MOUNT_PATH = '/workspace'

# Set up logging
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s : %(levelname)s : %(message)s',
    handlers=[logging.StreamHandler()]
)

def wait_for_service(url):
    """Wait for the ComfyUI service to be available"""
    retries = 0
    while True:
        try:
            requests.get(url)
            return
        except requests.exceptions.RequestException:
            retries += 1
            if retries % 15 == 0:
                logging.info('Service not ready yet. Retrying...')
        except Exception as err:
            logging.error(f'Error: {err}')
        time.sleep(0.2)

def send_get_request(endpoint):
    """Send a GET request to the API"""
    return session.get(
        url=f'{BASE_URI}/{endpoint}',
        timeout=TIMEOUT
    )

def send_post_request(endpoint, payload):
    """Send a POST request to the API"""
    return session.post(
        url=f'{BASE_URI}/{endpoint}',
        json=payload,
        timeout=TIMEOUT
    )

def create_unique_filename_prefix(payload):
    """
    Create a unique filename prefix for each request to avoid race conditions
    """
    for key, value in payload.items():
        class_type = value.get('class_type')
        if class_type == 'SaveImage':
            payload[key]['inputs']['filename_prefix'] = str(uuid.uuid4())

def get_output_images(output):
    """
    Get the output images
    """
    images = []

    for key, value in output.items():
        if 'images' in value and isinstance(value['images'], list):
            images.append(value['images'][0])

    return images

def main():
    # Initialize session with retries
    global session
    session = requests.Session()
    retries = Retry(total=10, backoff_factor=0.1, status_forcelist=[502, 503, 504])
    session.mount('http://', HTTPAdapter(max_retries=retries))
    session.mount('https://', HTTPAdapter(max_retries=retries))

    # Wait for the ComfyUI service to be available
    logging.info(f"Checking if ComfyUI API is available at {BASE_URI}")
    wait_for_service(url=f'{BASE_URI}/system_stats')
    logging.info('ComfyUI API is ready')

    # Load the payload from the file
    try:
        with open(PAYLOAD_PATH, 'r') as f:
            payload = json.load(f)
        logging.info(f'Successfully loaded payload from {PAYLOAD_PATH}')
    except Exception as e:
        logging.error(f'Failed to load payload: {e}')
        return

    # Add unique filename prefix to avoid collisions
    create_unique_filename_prefix(payload)
    logging.info('Queuing prompt')

    # Send the prompt to the API
    queue_response = send_post_request(
        'prompt',
        {
            'prompt': payload
        }
    )

    if queue_response.status_code == 200:
        resp_json = queue_response.json()
        prompt_id = resp_json['prompt_id']
        logging.info(f'Prompt queued successfully: {prompt_id}')
        retries = 0

        # Poll for completion
        while True:
            if retries == 0 or retries % 15 == 0:
                logging.info(f'Getting status of prompt: {prompt_id}')

            r = send_get_request(f'history/{prompt_id}')

            # Check if response has content before trying to parse JSON
            if r.status_code == 200 and r.content:
                try:
                    resp_json = r.json()
                    if resp_json and prompt_id in resp_json:
                        break
                except requests.exceptions.JSONDecodeError as e:
                    logging.warning(f"Failed to decode JSON response: {e}")

            time.sleep(0.2)
            retries += 1

            # Add a timeout to avoid infinite loops
            if retries > 1500:  # ~5 minutes
                logging.error("Timed out waiting for prompt completion")
                return

        status = resp_json[prompt_id]['status']

        if status['status_str'] == 'success' and status['completed']:
            print(json.dumps(resp_json, indent=4, default=str))
            # Job was processed successfully
            outputs = resp_json[prompt_id]['outputs']

            if len(outputs):
                logging.info(f'Images generated successfully for prompt: {prompt_id}')
                output_images = get_output_images(outputs)
                images = []

                for output_image in output_images:
                    print(output_image)
                    filename = output_image.get('filename')

                    if output_image['type'] == 'output':
                        image_path = f'{VOLUME_MOUNT_PATH}/ComfyUI/output/{filename}'
                        logging.info(f"image_path (output): {image_path}")

                        if os.path.exists(image_path):
                            with open(image_path, 'rb') as image_file:
                                image_data = base64.b64encode(image_file.read()).decode('utf-8')
                                images.append(image_data)
                                logging.info(f'Processed output file: {image_path}')
                                # Uncomment if you want to delete files after processing
                                # logging.info(f'Deleting output file: {image_path}')
                                # os.remove(image_path)
                    elif output_image['type'] == 'temp':
                        image_path = f'{VOLUME_MOUNT_PATH}/ComfyUI/temp/{filename}'
                        logging.info(f"image_path (temp): {image_path}")

                        # Remove temp images
                        if os.path.exists(image_path):
                            os.remove(image_path)

                logging.info(f'Successfully processed {len(images)} images')
                # Here you could save the base64 images or do something else with them
                return images
            else:
                logging.error(f'No output found for prompt id: {prompt_id}')
        else:
            # Job did not process successfully
            for message in status.get('messages', []):
                if len(message) >= 2:
                    key, value = message

                    if key == 'execution_error':
                        if isinstance(value, dict) and 'node_type' in value and 'exception_message' in value:
                            node_type = value['node_type']
                            exception_message = value['exception_message']
                            logging.error(f'{node_type}: {exception_message}')
                        else:
                            logging.error(f'Job failed for prompt_id: {prompt_id}')
                            logging.error(f'Response JSON: {resp_json}')
    else:
        try:
            queue_response_content = queue_response.json()
        except Exception:
            queue_response_content = str(queue_response.content)

        logging.error(f'HTTP Status code: {queue_response.status_code}')
        logging.error(queue_response_content)

if __name__ == '__main__':
    main()