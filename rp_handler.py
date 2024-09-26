import os
import time
import requests
import traceback
import json
import base64
import uuid
import logging
import logging.handlers
import runpod
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.modules.rp_logger import RunPodLogger
from requests.adapters import HTTPAdapter, Retry
from schemas.input import INPUT_SCHEMA


APP_NAME = 'runpod-worker-comfyui'
BASE_URI = 'http://127.0.0.1:3000'
VOLUME_MOUNT_PATH = '/runpod-volume'
LOG_FILE = 'comfyui-worker.log'
TIMEOUT = 600
LOG_LEVEL = 'INFO'


class SnapLogHandler(logging.Handler):
    def __init__(self, app_name: str):
        super().__init__()
        self.app_name = app_name
        self.rp_logger = RunPodLogger()
        self.rp_logger.set_level(LOG_LEVEL)
        self.runpod_endpoint_id = os.getenv('RUNPOD_ENDPOINT_ID')
        self.runpod_cpu_count = os.getenv('RUNPOD_CPU_COUNT')
        self.runpod_pod_id = os.getenv('RUNPOD_POD_ID')
        self.runpod_gpu_size = os.getenv('RUNPOD_GPU_SIZE')
        self.runpod_mem_gb = os.getenv('RUNPOD_MEM_GB')
        self.runpod_gpu_count = os.getenv('RUNPOD_GPU_COUNT')
        self.runpod_volume_id = os.getenv('RUNPOD_VOLUME_ID')
        self.runpod_pod_hostname = os.getenv('RUNPOD_POD_HOSTNAME')
        self.runpod_debug_level = os.getenv('RUNPOD_DEBUG_LEVEL')
        self.runpod_dc_id = os.getenv('RUNPOD_DC_ID')
        self.runpod_gpu_name = os.getenv('RUNPOD_GPU_NAME')
        self.log_api_endpoint = os.getenv('LOG_API_ENDPOINT')
        self.log_api_timeout = os.getenv('LOG_API_TIMEOUT', 5)
        self.log_api_timeout = int(self.log_api_timeout)
        self.log_token = os.getenv('LOG_API_TOKEN')

    def emit(self, record):
        # Only log to RunPod logger if the length of the log entry is >= 1000 characters
        if len(record.message) <= 1000:
            level_mapping = {
                logging.DEBUG: self.rp_logger.debug,
                logging.INFO: self.rp_logger.info,
                logging.WARNING: self.rp_logger.warn,
                logging.ERROR: self.rp_logger.error,
                logging.CRITICAL: self.rp_logger.error
            }

            # Wrapper to invoke RunPodLogger logging
            rp_logger = level_mapping.get(record.levelno, self.rp_logger.info)
            rp_logger(record.message)

        log_payload = {
            'app_name': self.app_name,
            'log_asctime': record.asctime,
            'log_levelname': record.levelname,
            'log_message': record.message,
            'runpod_endpoint_id': self.runpod_endpoint_id,
            'runpod_cpu_count': self.runpod_cpu_count,
            'runpod_pod_id': self.runpod_pod_id,
            'runpod_gpu_size': self.runpod_gpu_size,
            'runpod_mem_gb': self.runpod_mem_gb,
            'runpod_gpu_count': self.runpod_gpu_count,
            'runpod_volume_id': self.runpod_volume_id,
            'runpod_pod_hostname': self.runpod_pod_hostname,
            'runpod_debug_level': self.runpod_debug_level,
            'runpod_dc_id': self.runpod_dc_id,
            'runpod_gpu_name': self.runpod_gpu_name,
            'runpod_job_id': os.getenv('RUNPOD_JOB_ID')
        }

        if self.log_api_endpoint:
            try:
                headers = {'Authorization': f'Bearer {self.log_token}'}
                response = session.post(
                    self.log_api_endpoint,
                    json=log_payload,
                    headers=headers,
                    timeout=self.log_api_timeout
                )
                if response.status_code != 200:
                    self.rp_logger.error(f'Failed to send log to API. Status code: {response.status_code}')
            except requests.Timeout:
                self.rp_logger.error(f'Timeout error sending log to API (timeout={self.log_api_timeout}s)')
            except Exception as e:
                self.rp_logger.error(f'Error sending log to API: {str(e)}')
        else:
            self.rp_logger.warn('LOG_API_ENDPOINT environment variable is not set, not logging to API')


# ---------------------------------------------------------------------------- #
#                               ComfyUI Functions                              #
# ---------------------------------------------------------------------------- #

def wait_for_service(url):
    retries = 0

    while True:
        try:
            session.get(url)
            return
        except requests.exceptions.RequestException:
            retries += 1

            # Only log every 15 retries so the logs don't get spammed
            if retries % 15 == 0:
                logging.info('Service not ready yet. Retrying...')
        except Exception as err:
            logging.error(f'Error: {err}')

        time.sleep(0.2)


def send_get_request(endpoint):
    return session.get(
        url=f'{BASE_URI}/{endpoint}',
        timeout=TIMEOUT
    )


def send_post_request(endpoint, payload):
    return session.post(
        url=f'{BASE_URI}/{endpoint}',
        json=payload,
        timeout=TIMEOUT
    )


def get_txt2img_payload(workflow, payload):
    workflow["3"]["inputs"]["seed"] = payload["seed"]
    workflow["3"]["inputs"]["steps"] = payload["steps"]
    workflow["3"]["inputs"]["cfg"] = payload["cfg_scale"]
    workflow["3"]["inputs"]["sampler_name"] = payload["sampler_name"]
    workflow["4"]["inputs"]["ckpt_name"] = payload["ckpt_name"]
    workflow["5"]["inputs"]["batch_size"] = payload["batch_size"]
    workflow["5"]["inputs"]["width"] = payload["width"]
    workflow["5"]["inputs"]["height"] = payload["height"]
    workflow["6"]["inputs"]["text"] = payload["prompt"]
    workflow["7"]["inputs"]["text"] = payload["negative_prompt"]
    return workflow


def get_img2img_payload(workflow, payload):
    workflow["13"]["inputs"]["seed"] = payload["seed"]
    workflow["13"]["inputs"]["steps"] = payload["steps"]
    workflow["13"]["inputs"]["cfg"] = payload["cfg_scale"]
    workflow["13"]["inputs"]["sampler_name"] = payload["sampler_name"]
    workflow["13"]["inputs"]["scheduler"] = payload["scheduler"]
    workflow["13"]["inputs"]["denoise"] = payload["denoise"]
    workflow["1"]["inputs"]["ckpt_name"] = payload["ckpt_name"]
    workflow["2"]["inputs"]["width"] = payload["width"]
    workflow["2"]["inputs"]["height"] = payload["height"]
    workflow["2"]["inputs"]["target_width"] = payload["width"]
    workflow["2"]["inputs"]["target_height"] = payload["height"]
    workflow["4"]["inputs"]["width"] = payload["width"]
    workflow["4"]["inputs"]["height"] = payload["height"]
    workflow["4"]["inputs"]["target_width"] = payload["width"]
    workflow["4"]["inputs"]["target_height"] = payload["height"]
    workflow["6"]["inputs"]["text"] = payload["prompt"]
    workflow["7"]["inputs"]["text"] = payload["negative_prompt"]
    return workflow


def get_workflow_payload(workflow_name, payload):
    with open(f'/workflows/{workflow_name}.json', 'r') as json_file:
        workflow = json.load(json_file)

    if workflow_name == 'txt2img':
        workflow = get_txt2img_payload(workflow, payload)

    return workflow


"""
Get the filenames of the output images
"""
def get_filenames(output):
    for key, value in output.items():
        if 'images' in value and isinstance(value['images'], list):
            return value['images']


"""
Create a unique filename prefix for each request to avoid a race condition where
more than one request completes at the same time, which can either result in the
incorrect output being returned, or the output image not being found.
"""
def create_unique_filename_prefix(payload):
    for key, value in payload.items():
        class_type = value.get('class_type')

        if class_type == 'SaveImage':
            payload[key]['inputs']['filename_prefix'] = str(uuid.uuid4())


# ---------------------------------------------------------------------------- #
#                                RunPod Handler                                #
# ---------------------------------------------------------------------------- #
def handler(event):
    job_id = event['id']
    os.environ['RUNPOD_JOB_ID'] = job_id

    try:
        validated_input = validate(event['input'], INPUT_SCHEMA)

        if 'errors' in validated_input:
            return {
                'error': '\n'.join(validated_input['errors'])
            }

        payload = validated_input['validated_input']
        workflow_name = payload['workflow']
        payload = payload['payload']

        if workflow_name == 'default':
            workflow_name = 'txt2img'

        logging.info(f'Workflow: {workflow_name}', job_id)

        if workflow_name != 'custom':
            try:
                payload = get_workflow_payload(workflow_name, payload)
            except Exception as e:
                logging.error(f'Unable to load workflow payload for: {workflow_name}', job_id)
                raise

        create_unique_filename_prefix(payload)
        logging.debug('Queuing prompt', job_id)

        queue_response = send_post_request(
            'prompt',
            {
                'prompt': payload
            }
        )

        if queue_response.status_code == 200:
            resp_json = queue_response.json()
            prompt_id = resp_json['prompt_id']
            logging.info(f'Prompt queued successfully: {prompt_id}', job_id)
            retries = 0

            while True:
                # Only log every 15 retries so the logs don't get spammed
                if retries == 0 or retries % 15 == 0:
                    logging.info(f'Getting status of prompt: {prompt_id}', job_id)

                r = send_get_request(f'history/{prompt_id}')
                resp_json = r.json()

                if r.status_code == 200 and len(resp_json):
                    break

                time.sleep(0.2)
                retries += 1

            status = resp_json[prompt_id]['status']

            if status['status_str'] == 'success' and status['completed']:
                # Job was processed successfully
                outputs = resp_json[prompt_id]['outputs']

                if len(outputs):
                    logging.info(f'Images generated successfully for prompt: {prompt_id}', job_id)
                    image_filenames = get_filenames(outputs)
                    images = []

                    for image_filename in image_filenames:
                        filename = image_filename['filename']
                        image_path = f'{VOLUME_MOUNT_PATH}/ComfyUI/output/{filename}'

                        with open(image_path, 'rb') as image_file:
                            images.append(base64.b64encode(image_file.read()).decode('utf-8'))

                        logging.info(f'Deleting output file: {image_path}', job_id)
                        os.remove(image_path)

                    return {
                        'images': images
                    }
                else:
                    raise RuntimeError(f'No output found for prompt id: {prompt_id}')
            else:
                # Job did not process successfully
                for message in status['messages']:
                    key, value = message

                    if key == 'execution_error':
                        if 'node_type' in value and 'exception_message' in value:
                            node_type = value['node_type']
                            exception_message = value['exception_message']
                            raise RuntimeError(f'{node_type}: {exception_message}')
                        else:
                            # Log to file instead of RunPod because the output tends to be too verbose
                            # and gets dropped by RunPod logging
                            error_msg = f'Job did not process successfully for prompt_id: {prompt_id}'
                            logging.error(error_msg)
                            logging.info(f'{job_id}: Response JSON: {resp_json}')
                            raise RuntimeError(error_msg)

        else:
            try:
                queue_response_content = queue_response.json()
            except Exception as e:
                queue_response_content = str(queue_response.content)

            logging.error(f'HTTP Status code: {queue_response.status_code}', job_id)
            logging.error(queue_response_content, job_id)

            return {
                'error': f'HTTP status code: {queue_response.status_code}',
                'output': queue_response_content
            }
    except Exception as e:
        logging.error(f'An exception was raised: {e}', job_id)

        return {
            'error': traceback.format_exc(),
            'refresh_worker': True
        }


def setup_logging():
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
    log_handler = SnapLogHandler(APP_NAME)
    log_handler.setFormatter(formatter)
    logging.getLogger().addHandler(log_handler)


if __name__ == '__main__':
    session = requests.Session()
    retries = Retry(total=10, backoff_factor=0.1, status_forcelist=[502, 503, 504])
    session.mount('http://', HTTPAdapter(max_retries=retries))
    setup_logging()
    wait_for_service(url=f'{BASE_URI}/system_stats')
    logging.info('ComfyUI API is ready')
    logging.info('Starting RunPod Serverless...')
    runpod.serverless.start(
        {
            'handler': handler
        }
    )
