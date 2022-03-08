# Copyright 2019 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Runs AutoML NER on the text and writes results to BigQuery."""

import logging
import os
import utils
import requests

from google.cloud import storage, bigquery
from google.cloud import automl_v1beta1 as automl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_field_from_payload(text, payload, field_name, default_value='None'):
  """Parses a payload to extract the value of the given field.

  Args
    text: text analyzed by AutoML NER.
    payload: payload returned by AutoML NER.
    field_name: Name of the field to extract.
    default_value: Value to return if the field can not be found.

  Returns:
    extracted value.

  In case the payload contains several times the given field, we take the occurence
    with the highest score.
  """
  value_found = default_value
  score_found = -1
  for result in payload:
    extracted_field_name = result["displayName"]
    extracted_value_start = int(result["textExtraction"]["textSegment"]["startOffset"])
    extracted_value_end = int(result["textExtraction"]["textSegment"]["endOffset"])
    
    extracted_value = text[extracted_value_start:extracted_value_end]

    score = result["textExtraction"]["score"]
    
    if (extracted_field_name == field_name) and (score > score_found):
      score_found = score
      value_found = extracted_value
  return value_found


def run_automl_single(ocr_path,
                      list_fields,
                      service_acct,
                      model_id,
                      main_project_id,
                      compute_region,
                      id_token,
                      content):
  """Runs AutoML NER on a single document and returns the dictionary of results."""
  
  #[START] GSP666-API REQUEST
  header = {'Authorization': 'Bearer ' + id_token}
  data = {"compute_region": compute_region,
      "model_id": model_id,
      "text_url": content
      }
  url = 'https://gsp666-api-kjyo252taq-uc.a.run.app/text'
  response = requests.post(url, json=data, headers=header)
  response_data = response.json()
  logger.info("response data from API")
  logger.info(response_data)
  #[END] GSP666-API REQUEST

  # Load text
  text = utils.download_string(ocr_path, service_acct).read().decode('utf-8')

  # Parse results
  results = {'file': os.path.basename(ocr_path).replace('.txt', '.pdf')}
  if "payload" in  response_data:
    for field in list_fields:      
        value_field = extract_field_from_payload(text, response_data["payload"], field)
        results[field] = value_field
    return results

def predict(main_project_id,
            input_path,
            demo_dataset,
            demo_table,
            model_id,
            service_acct,
            compute_region,
            id_token,
            config):
  """Runs AutoML NER on a folder and writes results to BigQuery.

  Args:
    gcs_ocr_text_folder: JSON folder (outputs of OCR).    
    dataset_bq: BiqQuery dataset name.
    table_bq_output: BigQuery table where the ner results are written to.
    project_id_ner: Project ID for AutoML Ner.
    project_id_bq: Project ID for BigQuery Table.
    ner_model_id: AutoML Model ID (NER).
    list_fields: List of field_names to extract (list of string).
    service_account_ner: Location of service account key to access the NER model.
    service_account_gcs_bq: Location of service account key to access BQ and Storage.
    compute_region: Compute Region for NER model.
  """
  logger.info('Starting entity extraction.')

  input_bucket_name = input_path.replace('gs://', '').split('/')[0]
  input_txt_folder = f"gs://{input_bucket_name}/{demo_dataset}/txt"

  list_fields = [x['field_name'] for x in config["model_ner"]["fields_to_extract"]]
  list_fields.remove('gcs_path')

  storage_client = storage.Client.from_service_account_json(service_acct)
  bucket_name, path = utils.get_bucket_blob(input_txt_folder)
  bucket = storage_client.get_bucket(bucket_name)

  list_results = []
  for file in bucket.list_blobs(prefix=path):
    full_filename = os.path.join(input_txt_folder, os.path.basename(file.name))
    logger.info(full_filename)
    url_lifetime = 3600  # Seconds in an hour
    content = file.generate_signed_url(expiration=url_lifetime, version='v4')
    result = run_automl_single(ocr_path=full_filename,
                               list_fields=list_fields,
                               service_acct=service_acct,
                               model_id=model_id,
                               main_project_id=main_project_id,
                               compute_region=compute_region,
                               id_token=id_token,
                               content=content)
    list_results.append(result)

  schema = [bigquery.SchemaField('file', 'STRING', mode='NULLABLE')]
  for field in list_fields:
      schema.append(bigquery.SchemaField(field, 'STRING', mode='NULLABLE'))
  
  utils.save_to_bq(
    demo_dataset,
    demo_table,
    list_results,
    service_acct,
    _create_table=True,
    schema=schema)

  logger.info('Entity extraction finished.\n')