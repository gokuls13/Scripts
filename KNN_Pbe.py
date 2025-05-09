import os
import sys
import shutil
import re
import numpy as np
import pandas as pd
import polars as pl
from datetime import datetime, timezone
import hashlib
import itertools
import uuid
import json
import math
from sqlalchemy import create_engine
from io import StringIO
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from memory_profiler import memory_usage
import psutil
import time


# make the dataset the same as the KNN to handle any or further steps
input_config = {
"source_type": "ftr",
"source_address": "/commonshare/pine/digital/perf_cep_eig/fuzzy_extracts/data_file/pbe_input_chunk_#.ftr"
}
reference_config= {
"source_type": "ftr",
"source_address": "/commonshare/pine/digital/perf_cep_eig/fuzzy_extracts/data_file/pbe_reference_chunk_#.ftr"
}
prematching_config = {
"distance_threshold": 3,
"max_matches": 5
}
matching_config ={
"confidence_threshold": 0.75,
"number_of_matches": 1,
"score_cutoff": 0.65
}
evaluation_config={
    "ip": {
      "match_mode": "any",
      "match_items_min": 1,
      "match_score_cutoff": 1,
      "match_weight": 40,
      "match_mandatory": True
    },
    "ua_tokenized": {
      "match_mode": "fuzzy",
      "match_items_min": 4,
      "match_score_cutoff": 0.5,
      "match_weight": 30
    },
    "all_dellid": {
      "match_mode": "any",
      "match_items_min": 0,
      "match_score_cutoff": 0,
      "match_weight": 30,
      "match_mandatory": False
    },
    "language_info": {
      "match_mode": "fuzzy",
      "match_ngrams_length":3,
      "match_items_min": 1,
      "match_score_cutoff": 0.5,
      "match_weight": 10
    },
    "javascript_info": {
      "match_mode": "precise",
      "match_items_min": 2,
      "match_score_cutoff": 0.8,
      "match_weight": 5 #10 for audience name
    },
    "device_info": {
      "match_mode": "fuzzy",
      "match_ngrams_length":3,
      "match_items_min": 4,
      "match_score_cutoff": 0.4,
      "match_weight": 20
    },
	"browser_info": {
      "match_mode": "fuzzy",
      "match_ngrams_length":3,
      "match_items_min": 4,
      "match_score_cutoff": 0.4,
      "match_weight": 10
    },
	"location_info": {
      "match_mode": "fuzzy",
	  "match_ngrams_length":3,
      "match_items_min": 2,
      "match_score_cutoff": 0.8,
      "match_weight": 10
    }
}
output_config = {
 "db_output_config": {
    "match_table_address": "dgtl_pbe_cookie_matches",
    "process_table_address": "dgtl_pbe_cookie_processes"
 }
}

pbe_input_chunk_location = 'C:/Users/G_Subramanian/OneDrive - Dell Technologies/Documents/probablistic matching/test_files/pbe_input_chunk_88.ftr'
pbe_reference_chunk_location ='C:/Users/G_Subramanian/OneDrive - Dell Technologies/Documents/probablistic matching/test_files/pbe_reference_chunk_88.ftr'

input_data = {
    'mcvisid': ['28500858651727346285020350108621069461', '24620052857140712789123942953428383601', '13586664542912325873598095252796534704', '34331333902054495668202153208490115857'],
    'record_id': ['3705e458d52a21208c9318eb14d64635', 'ba5a0edfb085a930e6428888b4dabd39', '9fe55ae306959995752ea3f73b0890f0', '01b6993605b3a32188a7f6df2cfdd8ef'],
    'ip': ['202.185.182.208', '223.238.109.185', '192.178.10.1', '192.178.10.1'],
    'all_dellid': ['16b4116087920f004e414866c1030000ce980100', '', '', 'd33a2f175342000030a16166d2000000de940400']
}

reference_data = {
    'mcvisid': ['66778287741467308500551369026521589689', '16767050189469603081376805249713206768', '12149813982851293790674001261693009489', '76817602842982262313383500064451056181', '92202794005391452041687519518602851762','77202794005391452041687519518602851762','88202794005391452041687519518602851762','99202794005391452041687519518602851762'],
    'record_id': ['f3f22ed0b4b93e32d50d819ed42b4757', '26703e0771e4559dba122c8d5c2935bc', '579e54f1b5f5dc20abb3329cfc2aed30', '8081f32fdf5667844c231bce7ae5be64', 'a0d4bed6510d1f425af881ccbe006a77','a3d4bed6510d1f425af881ccbe006a77','a2d4bed6510d1f425af881ccbe006a77','a1d4bed6510d1f425af881ccbe006a77'],
    'ip': ['192.178.10.1', '202.185.182.208', '202.556.182.208', '202.185.182.208', '223.238.109.185', '103.185.182.208', '202.185.182.208', '202.287.182.208'],
    'all_dellid': ['e8382f17aed33d00de916c66080000008f530100', '21f8cd1744451e0042697f663d000000f2350000', '35ab34176ca308008ef5ab650802000077040000', '16b41160ef4214003cd97066b5030000ae040000', '7c5d3a17ed3c25007d7c86664c000000ad220000','1c5d3a17ed3c25007d7c86664c000000ad220000','9c5d3a17ed3c25007d7c86664c000000ad220000','8c5d3a17ed3c25007d7c86664c000000ad220000']
}



# pbe_input = pd.DataFrame(input_data)
# pbe_reference = pd.DataFrame(reference_data)

# pbe_input = pd.read_feather(pbe_input_chunk_location)
# pbe_reference = pd.read_feather(pbe_reference_chunk_location)

input_sample_data = {
    'record_id': [
        '57803898799474217141063769422309591632', '64788255274159110758554466918210235528', 
        '52956921864344840083167122687203035508', '29724420621880013701197680127590713444', 
        '39685178725791382409215285946862277618', '35613802664672351018336591952431218295'
    ],
    'mcvisid': [
        'a529de88c0bd83e69adbc3ce43974ce6', '7b0a85908e309900a8adbd3525816a29', 
        'b96f8a550ae37e2b25fd29a675bf61db', '0d8e3e1926cea282128951b5440b9a73',
        '003df3b306b67c0a53fd94d936aa634a', '95a741daf3c7685c255989c57c1672e2'
    ],
    'ip': [
        '197.250.100.14', '73.43.206.182', '74.254.113.126', '191.58.68.203', 
        '209.17.40.36','197.250.100.14'
    ],
    'ua_tokenized': [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML like Gecko) Chrome/125.0.0.0 Safari/537.36', 
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML like Gecko) Chrome/125.0.0.0 Safari/537.36 Edg/125.0.0.0', 
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0', 
        'Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML like Gecko) Chrome/123.0.0.0 Mobile Safari/537.36', 
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0', 
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML like Gecko) Chrome/125.0.0.0 Safari/537.36 Edg/125.0.0.0', 
    ],
    'all_dellid': [
        '2a6d100251bf000074066766930000009d2b0000', 'bc5dda17a1fb1f00e36e4a6630030000e27f0100', 
        '341a7b5cac343900d98469663b0000009b000000', 'b4cd586879ba2f0030c731654a0200009d420000', 
        'e5c8301735d43300402324666f0300007c6d0100', 'af332c17a16835000a04f065c502000067030000'
    ],
    'language_info': [
        'en|gb|us|english|united|kingdom', 'en|us|fi|english|united|states', 
        'en|us|english|united|states', 'pt|br|en|us|portuguese|brazil', 
        'en|us|es|english|united|states', 'ja|en|gb|japanese'
    ],
    'javascript_info': [
        '7~1.6~N', '7~1.6~N', '7~1.6~N', '7~1.6~N', '7~1.6~N', 
        '7~1.6~N'
    ],
    'device_info': [
        'Microsoft Windows~2~{1920x1080}~Mobile Carrier~vodacom', 
        'Microsoft Windows~2~{1920x1080 2560x1440}~LAN/wifi', 
        'Microsoft Windows~2~{1536x864}~LAN/wifi', 
        'Google Android~2~Unknown~M3~Mobile Phone~Android~10.1~800~1200~{384x857}~Mobile Carrier|LAN/wifi~claro', 
        'Microsoft Windows~2~{1280x800 1920x1080}~LAN/wifi', 
        'Microsoft Windows~2~{2560x1440}~LAN/wifi'
    ],
    'browser_info': [
        'Chrome 125.0~Google~{1920x945 929x917}', 'Edge 125.0~Microsoft~{1489x887 2512x1284}', 
        'Firefox 126.0~Mozilla~{1504x978 1628x732 1236x768}', 'Chrome 123.0~Google~{360x592}', 
        'Firefox 115.0~Mozilla~{1905x973 1931x1127}', 'Edge 125.0~Microsoft~{2392x1345 2392x1262}'
    ],
    'location_info': [
        'ZA~South Africa~Gauteng~Johannesburg~Sandton~26.0558~28.0614~50252~264755~6~yes~~15.0~~Sat~Feb~04~2023~16:59:57~+02:00~SAST', 
        'US~United States~Minnesota~Minneapolis~Saint Louis Park~44.9483~-93.3294~23442~612~6~yes~~5.0~~Tue~Feb~07~2023~13:47:12~-06:00~CST', 
        'US~United States~Washington~Seattle~Seattle~47.6092~-122.3311~14321~206~6~yes~~5.0~~Tue~Feb~07~2023~13:50:13~-08:00~PST', 
        'BR~Brazil~Sao Paulo~Sao Paulo~Sao Paulo~-23.5475~-46.6361~11887~011~6~yes~~15.0~~Thu~Feb~09~2023~19:01:35~-03:00~BRT', 
        'US~United States~California~Los Angeles~Los Angeles~34.0522~-118.2437~39947~213~6~yes~~5.0~~Thu~Feb~09~2023~18:05:21~-08:00~PST', 
        'JP~Japan~Tokyo~Tokyo~Tokyo~35.6895~139.6917~37974~03~6~yes~~15.0~~Sat~Feb~11~2023~15:15:32~+09:00~JST'
    ],
    'timezone': [
        'GMT+2', 'GMT-6', 'GMT-8', 'GMT-3', 'GMT-8', 'GMT+9'
    ],
    'country': [
        'ZA', 'US', 'US', 'BR', 'US', 'JP'
    ],
    'date_time': [
        '2023-02-04 14:59:57', '2023-02-07 19:47:12', '2023-02-07 21:50:13', 
        '2023-02-09 22:01:35', '2023-02-10 10:05:21', '2023-02-11 19:15:32'
    ]
}

reference_sample_data = {
    'record_id': [
        '39685178725791382409215285946862277618', '35613802664672351018336591952431218295', 
        '25863761313115636116923893132250573350', '62800074902857353343071935425533869222', 
        '90259717970294112050067860578204134693', '23170748953986358943531800052729532061'
    ],
    'mcvisid': [
        '003df3b306b67c0a53fd94d936aa634a', 'b5cb77c5280e5c5d512452b4542e511d', 
        '95a741daf3c7685c255989c57c1672e2', 'a8cc6b8973ec508b7940f310adbb4477', 
        '24cd1d689f308072cb33fcf75966483f', '58d14564540fc52b46d27967b60e5aa2'
    ],
    'ip': [
        '209.17.40.36|202.212.64.201', '202.212.64.201', '136.226.48.118', '128.119.202.127', 
        '73.213.248.187', '70.140.132.23'
    ],
    'ua_tokenized': [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0', 
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML like Gecko) Chrome/125.0.0.0 Safari/537.36 Edg/125.0.0.0', 
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML like Gecko) Chrome/123.0.0.0 Safari/537.36', 
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML like Gecko) Chrome/124.0.0.0 Safari/537.36', 
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0', 
        'Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML like Gecko) Chrome/124.0.0.0 Mobile Safari/537.36'
    ],
    'all_dellid': [
        'e5c8301735d43300402324666f0300007c6d0100', 'af332c17a16835000a04f065c502000067030000', 
        'c769dc175c0f2700f2ea5566860200000cd00100', '341a7b5cac343900d98469663b0000009b000000|af332c17a16835000a04f065c502000067030000', '#', '#'
    ],
    'language_info': [
        'en|us|es|english|united|states', 'ja|en|gb|japanese', 
        'en|us|es|english|united|states', '#', 'en|us|english|united|states', 
        'en|us|english|united|states'
    ],
    'javascript_info': [
         '7~1.6~N', 
        '7~1.6~N', '7~1.6~N', '#', '7~1.6~N', '7~1.6~N'
    ],
    'device_info': [
        'Microsoft Windows~2~{1280x800 1920x1080}~LAN/wifi', 
        'Microsoft Windows~2~{2560x1440}~LAN/wifi', 
        'Apple Macintosh~2~{1440x900}~LAN/wifi', '#', 
        'Microsoft Windows~2~{1191x745}~LAN/wifi', 'Google Android~2~Unknown~M3~Mobile Phone~Android~10.1~800~1200~{377x843}~LAN/wifi'
    ],
    'browser_info': [
        'Firefox 115.0~Mozilla~{1150x564 1250x560 1586x526}', 
        'Edge 125.0|Edge 126.0|Edge 123.0|Edge 124.0~Microsoft~{2512x1326}|{1701x789 1701x806 2512x1326}', 
        'Chrome 123.0~Google~{1440x779}', '#', 'Edge 123.0~Microsoft~{871x571}', 
        'Chrome 124.0~Google~{377x711}'
    ],
    'location_info': [
        'usa|col~burlington|bogota|orlando~ma|dc|fl~01803|110861|32822~300', 
        'jpn~aikawa|shinchiba|tanabe~14|12|26~243-0301|260-0031|610-0331~-540', 
        'usa~washington~dc~20011~240', 'usa~amherst~ma~01003~240', 
        'usa~hyattsville~md~20785~240', 'usa~channelview~tx~77530~300'
    ],
    'timezone': [
         '300', 
        '-540', '240', '240', '240', '300'
    ],
    'country': [
         'usa|col', 
        'jpn', 'usa', 'usa', 'usa', 'usa'
    ],
    'date_time': [
        '06/09/2024 8:12:26.000000', '07/01/2024 6:23:09.000000', 
        '05/28/2024 2:37:40.000000', '05/13/2024 12:00:59.000000', 
        '04/26/2024 2:20:15.000000', '05/14/2024 3:27:09.000000'
    ]
}


def pl_pbe_dataset_initialization(pl_input_dataset: pl.DataFrame=None):
    schema = {
        "record_id": pl.Utf8,
        "mcvisid": pl.Utf8,
        "ip": pl.Utf8,
        "ua_tokenized": pl.Utf8,
        "all_dellid": pl.Utf8,
        "language_info": pl.Utf8,
        "javascript_info": pl.Utf8,
        "device_info": pl.Utf8,
        "browser_info": pl.Utf8,
        "location_info": pl.Utf8,
        "timezone": pl.Utf8,
        "country": pl.Utf8,
        "date_time": pl.Utf8,
        "pbe_ip": pl.List(pl.Utf8),
        "pbe_ua_tokenized": pl.List(pl.Utf8),
        "pbe_all_dellid": pl.List(pl.Utf8),
        "pbe_language_info": pl.List(pl.Utf8),
        "pbe_javascript_info": pl.List(pl.Utf8),
        "pbe_device_info": pl.List(pl.Utf8),
        "pbe_browser_info": pl.List(pl.Utf8),
        "pbe_location_info": pl.List(pl.Utf8)
    }

    pl_pbe_dataset = pl.DataFrame(schema=schema)

    if pl_input_dataset is not None and len(pl_input_dataset) > 0:
        pl_df = pl_input_dataset.with_columns([pl.all().cast(pl.Utf8).fill_null('')])
        pl_pbe_dataset = pl.concat([pl_pbe_dataset, pl_df], how="diagonal")
        #print(pl_pbe_dataset)

    return pl_pbe_dataset

def pl_generate_pbe_dataset(df: pd.DataFrame):
    # Initialize and copy input data
    pl_dataset = pl_pbe_dataset_initialization(pl.from_pandas(df)).lazy()
    pl_dataset_columns = pl_dataset.collect_schema().names()

    # Convert all fields to string and populate missing values with ''
    pl_dataset = pl_dataset.with_columns([pl.all().cast(pl.Utf8).fill_null('')])

    # Check if mandatory columns are populated
    pl_dataset = pl_dataset.with_columns([
        ((pl.col('record_id') != '') | (pl.col('mcvisid') != '')).cast(int).alias('valid_record')
    ])
    
    invalid_records = pl_dataset.filter(pl.col('valid_record') == 0).select(pl.len()).collect()[0, 0]

    if invalid_records > 0:
        print(f"{invalid_records} records do not have valid record_id/mcvisid and will be removed from PBE dataset")
        pl_dataset = pl_dataset.filter(pl.col('valid_record') == 1)

    # Pretreatment: split and transform columns
    pl_dataset = pl_dataset.with_columns([
        pl.col("ip").str.split('|').alias('pbe_ip'),
        pl.col("ua_tokenized").str.split('|').alias('pbe_ua_tokenized'),
        pl.col("all_dellid").str.split('|').alias('pbe_all_dellid'),
        pl.col("language_info").str.split('|').alias('pbe_language_info'),
        pl.col("javascript_info").str.to_uppercase().str.split('~').alias('pbe_javascript_info'),
        pl.col("device_info").str.split('~').alias('pbe_device_info'),
        pl.col("browser_info").str.to_uppercase().str.split('~').alias('pbe_browser_info'),
        pl.col("location_info").str.to_uppercase().str.split('~').alias('pbe_location_info')
    ])

    # Drop validation and preprocessing columns
    columns_to_drop = [col for col in pl_dataset.collect_schema().names() if col not in pl_dataset_columns]
    pl_dataset = pl_dataset.drop(columns_to_drop).collect()

    return pl_dataset


# Convert data to a Pandas DataFrame
# input_df= pd.DataFrame(input_sample_data)
# reference_df = pd.DataFrame(reference_sample_data)

input_df = pd.read_feather(pbe_input_chunk_location)
reference_df = pd.read_feather(pbe_reference_chunk_location)

# Generate the processed dataset using the pl_generate_pbe_dataset function
pre_pbe_input = pl_generate_pbe_dataset(input_df)
pre_pbe_reference = pl_generate_pbe_dataset(reference_df)

# Convert the Polars DataFrame to a Pandas DataFrame
pbe_input = pre_pbe_input.to_pandas()
pbe_reference = pre_pbe_reference.to_pandas()


class ProfileFunction:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        start_time = time.time()
        mem_usage_before = memory_usage(-1, interval=0.1, timeout=1)
        cpu_usage_before = psutil.cpu_percent(interval=1)

        result = self.func(*args, **kwargs)

        mem_usage_after = memory_usage(-1, interval=0.1, timeout=1)
        cpu_usage_after = psutil.cpu_percent(interval=1)
        end_time = time.time()
        time_taken = end_time - start_time
        mem_usage_diff = mem_usage_after[0] - mem_usage_before[0]
        
        print(f"Function: {self.func.__name__}")
        print(f"Time taken: {time_taken:.4f} seconds")
        print(f"Memory usage: {mem_usage_diff:.4f} MiB")
        print(f"CPU usage before: {cpu_usage_before}%")
        print(f"CPU usage after: {cpu_usage_after}%")
        
        return result
    
def pbe_match_dataset_initialization(
    input_dataset: pd.DataFrame=None
    ):
    """
    function to initialize empty PBE MATCH dataframe
    """
    output_columns = [ 
         'in_record_id'
        ,'in_mcvisid'
        ,'in_ip'
        ,'in_ua_tokenized'
        ,'in_all_dellid'
        ,'in_language_info'
        ,'in_javascript_info'
        ,'in_device_info'
        ,'in_browser_info'
        ,'in_location_info'
        ,'in_timezone'
        ,'in_country'
        ,'in_date_time'
        ,'ref_record_id'
        ,'ref_mcvisid'
        ,'ref_ip'
        ,'ref_ua_tokenized'
        ,'ref_all_dellid'
        ,'ref_language_info'
        ,'ref_javascript_info'
        ,'ref_device_info'
        ,'ref_browser_info'
        ,'ref_location_info'
        ,'ref_timezone'
        ,'ref_country'
        ,'ref_date_time'
        ,'pbe_ip_score'
        ,'pbe_ua_tokenized_score'
        ,'pbe_all_dellid_score'
        ,'pbe_language_info_score'
        ,'pbe_javascript_info_score'
        ,'pbe_device_info_score'
        ,'pbe_browser_info_score'
        ,'pbe_location_info_score'
        ,'pbe_match_score'
        ,'pbe_match_rank'
        ,'pbe_match_confidence_flag'
        ]

    pbe_matches_dataset = pd.DataFrame(columns=output_columns)

    if input_dataset is not None and len(input_dataset) > 0:
        pbe_matches_dataset = pd.concat([pbe_matches_dataset, input_dataset], axis=0, join='inner', ignore_index=True)

    return pbe_matches_dataset

def md5_hash(
    tokens: list=[], 
    separator: str='|', 
    null_replacement: str='NULL', 
    null_values: list=['',None,np.NaN],
    case_sensitivity: bool=False
    ):
    """
    - function to generate MD5 hash key from list of values
    """

    # replace NULLs
    tokens_cleaned = separator.join([null_replacement if i in null_values else i for i in tokens])
    tokens_cleaned = tokens_cleaned if case_sensitivity==True else tokens_cleaned.lower()

    return hashlib.md5(tokens_cleaned.encode('utf-8')).hexdigest()

def generate_fcm_process_id():
    """
    - function to generate unique FCM process id
    """
    process_id = str(uuid.uuid4())
    timestamp = datetime.now().astimezone(timezone.utc).strftime('%Y-%m-%d %H:%M:%S.%f')
    fcm_process_id = md5_hash([process_id, timestamp])

    return fcm_process_id


def generate_prematch_key(fcm_dataset: pd.DataFrame, key_name: str='fcm_prematch_key', prematch_key_columns: list=None):
    if prematch_key_columns is None:
        raise Exception("prematch_key_columns must be specified")

    tmp_fcm_dataset = fcm_dataset.copy()
    tmp_fcm_dataset[key_name] = ''
    tmp_fcm_dataset[key_name] = tmp_fcm_dataset[sorted(prematch_key_columns)].values.tolist()
    tmp_fcm_dataset[key_name] = tmp_fcm_dataset[key_name].apply(lambda x: ' '.join(list(set([item for sublist in x for item in sublist]))))

    return tmp_fcm_dataset

def get_ngrams(text: str=None, n: int=None, min_n: int=2):

    if text!=None:   
        if text!="" and (text.lower()!="nan" and text.lower()!='na') and text!=None:
   
            text = re.split(r'[|~{,]+', text.upper().strip())
    
    
            text = ' '.join(text)  # Join the list into a string
            text = re.sub(r'\.', '', text)
            text=re.sub(r'}','',text)
            #print(text)
    
            if n is not None:
                if len(text) <n:
                    ngram_list = [text]
    
                else:
                    ngrams = (text[i:i+n] for i in range(len(text) - n + 1) if ' ' not in text[i:i+n] and len((text[i:i+n])) == n)
                    ngram_list=[i for i in list(set(ngrams)) if  i!='']
                    short_words = [word for word in text.split() if len(word) <n]
                    ngram_list.extend(i for i in set(short_words) if i!='')
    
            # if n is not specified, get list of words
            else:
                ngrams = re.split('\\s+', text)
                ngram_list = [i for i in list(set(ngrams)) if i != '']
            return ngram_list
        else:
            return ['derp']
    else:
        return ['derp']


def pbe_matching(
    input_pbe_dataset: pd.DataFrame=None, 
    reference_pbe_dataset: pd.DataFrame=None, 
    prematch_mapping: pd.DataFrame=None, 
    evaluation_configuration: dict=None,
    matching_configuration: dict=None
    ):
    """
    - procedure evaluating matches between input and reference PBE datasets
    """

    ##################################################################################
    # PREPARE MATCHING DATASET
    ##################################################################################

    # prepare temporary copy of input datasets
    in_df = input_pbe_dataset.rename(columns={f"{col}":f"in_{col}" for col in input_pbe_dataset.columns}).copy()
    ref_df = reference_pbe_dataset.rename(columns={f"{col}":f"ref_{col}" for col in reference_pbe_dataset.columns}).copy()

    in_df['input_index'] = in_df.index.astype(str)
    ref_df['reference_index'] = ref_df.index.astype(str)
    
    # Ensure 'input_index' is of the same type in both DataFrames
    prematch_mapping['input_index'] = prematch_mapping['input_index'].astype(str)
    
    df_matched = pd.merge(in_df, prematch_mapping, on='input_index', how="inner")
    df_matched = pd.merge(df_matched, ref_df, on='reference_index', how="inner")
    
    # # Ensure 'input_index' is of the same type in both DataFrames
    # prematch_mapping['input_index'] = prematch_mapping['input_index'].astype(str)

    print(f"PBE MATCHING DATASET created with {len(df_matched)} records")

    # drop temporary input datasets
    del in_df
    del ref_df

    # add column to flag disqualified records due to match_mandatory constaints
    df_matched['pbe_match_confidence_flag'] = ''

    ##################################################################################
    # EVALUATE MATCH QUALITY
    ##################################################################################

    pbe_score_columns = []
    pbe_score_max = 0
    for attribute_name, attribute_config in evaluation_configuration.items():
    
        matching_mode = attribute_config.get("match_mode", "precise").upper()

        # skip evaluation if attribute weight is 0
        if attribute_config.get('match_weight', 0) == 0: continue

        # print(attribute_name, attribute_config)
        print(f"processing attribute '{attribute_name.upper()}' using '{matching_mode}' matching logic...")

        try:

            # prepare token intersection
            df_matched[f'pbe_{attribute_name}_intersection'] = df_matched.apply(lambda x: np.intersect1d(x[f'ref_pbe_{attribute_name}'],x[f'in_pbe_{attribute_name}']), axis = 1)
            #print(f'pbe_{attribute_name}_intersection')

            if matching_mode in ["PRECISE", "FUZZY"]:

                # prepare list of unique tokens
                df_matched[f'pbe_{attribute_name}_union'] = df_matched.apply(lambda x: np.union1d(x[f'ref_pbe_{attribute_name}'],x[f'in_pbe_{attribute_name}']), axis = 1)
                #print(f'pbe_{attribute_name}_union')

                # calculate matching score if number of overlaping items is higher than threshold
                df_matched[f'pbe_{attribute_name}_score'] = df_matched[f'pbe_{attribute_name}_intersection'].apply(lambda x: len(x) if len(x) >= attribute_config.get('match_items_min', 0) else 0) \
                    / df_matched[f'pbe_{attribute_name}_union'].str.len()
                #print(f'pbe_{attribute_name}_score')

                # fill blanks caused by 0-division with 0
                df_matched[f'pbe_{attribute_name}_score'] = df_matched[f'pbe_{attribute_name}_score'].fillna(0)

            elif matching_mode == "ANY":

                df_matched[f'pbe_{attribute_name}_score'] = df_matched[f'pbe_{attribute_name}_intersection'].apply(lambda x: 1 if len(x) > 0 else 0)
                #print(df_matched[f'pbe_{attribute_name}_score'])
            df_matched[f'pbe_{attribute_name}_score']=df_matched.apply(lambda row: 0 if row[f'ref_pbe_{attribute_name}'].tolist() == ['derp'] and row[f'in_pbe_{attribute_name}'].tolist() == ['derp'] else row[f'pbe_{attribute_name}_score'], axis=1)

            # calculate weighted matching score if matching score is higher than threshold
            df_matched[f'pbe_{attribute_name}_score_weighted'] = df_matched[f'pbe_{attribute_name}_score'].apply(lambda x: 0 if x < attribute_config.get('match_score_cutoff', 1) else x) \
                * attribute_config.get('match_weight', 1)
            #print(df_matched[f'pbe_{attribute_name}_score_weighted'])

            # include column in the output
            pbe_score_columns.append(f'pbe_{attribute_name}_score_weighted')

            # include attribute weight in match score calculation
            pbe_score_max = pbe_score_max + attribute_config.get('match_weight',0)

            # if match is mandatory, set pbe_match_confidence_flag to N 
            if attribute_config.get('match_mandatory', False) == True: 
                df_matched['pbe_match_confidence_flag'] = np.where(
                    (df_matched['pbe_match_confidence_flag'] == '') & (df_matched[f'pbe_{attribute_name}_score_weighted'] == 0),
                    'N', 
                    df_matched['pbe_match_confidence_flag']
                )
                

        except Exception as e:
            print(f"unable to evaluate '{attribute_name}'")
            print(e)

    # calculate overall match score 
    df_matched['pbe_match_score'] = df_matched[pbe_score_columns].sum(axis=1) / pbe_score_max
    #print(pbe_score_max)
    #print(df_matched[pbe_score_columns])
    
    # rank matches per in_record_id by match score (disqualifying matches with pbe_match_confidence_flag = 'N') #New Version 
    df_matched['pbe_match_rank'] = np.where(
                    (df_matched['pbe_match_confidence_flag'] == 'N'),
                    df_matched['pbe_match_score'] * 0.001, 
                    df_matched['pbe_match_score']
                ).astype(float)
    df_matched['pbe_match_rank'] = df_matched.groupby(['in_record_id'])['pbe_match_rank'].rank(method="first", ascending=False).astype(int)

    # filter valid matches
    #df_matched['pbe_match_rank'] = df_matched.groupby(['in_record_id'])['pbe_match_score'].rank(method="first", ascending=False).astype(int) #Previous Version


    ##################################################################################
    # PREPARE OUTPUT
    ##################################################################################

    # remove intermediary calculation columns
    df_output = pbe_match_dataset_initialization(df_matched)

    # filter match quality
    df_output = df_output[df_output['pbe_match_score'] >= matching_configuration.get('score_cutoff', 0)]

    # filter number of matches
    df_output = df_output[df_output['pbe_match_rank'] <= matching_configuration.get('number_of_matches', 10)]

    # evaluate match confidence if record was not already invalidated due to attribute with match_mandatory=True
    df_output['pbe_match_confidence_flag'] = np.where(
        df_output['pbe_match_confidence_flag'] == '',
        df_output['pbe_match_score'].apply(lambda x: 'Y' if x >= matching_configuration.get('confidence_threshold', 0) else 'N'),  
        df_output['pbe_match_confidence_flag']
    )


    #df_output['pbe_region'] = df_matched['ref_pbe_region'].apply(lambda x: ','.join(x).upper())
    #df_output['pbe_country'] = df_matched['ref_pbe_country'].apply(lambda x: ','.join(x).upper())

    # define unique key for each match
    pbe_match_id_cols =	sorted([
         'in_record_id'
        ,'in_mcvisid'
        ,'in_ip'
        ,'in_ua_tokenized'
        ,'in_all_dellid'
        ,'in_language_info'
        ,'in_javascript_info'
        ,'in_device_info'
        ,'in_browser_info'
        ,'in_location_info'
        ,'in_timezone'
        ,'in_country'
        ,'in_date_time'
        ,'ref_record_id'
        ,'ref_mcvisid'
        ,'ref_ip'
        ,'ref_ua_tokenized'
        ,'ref_all_dellid'
        ,'ref_language_info'
        ,'ref_javascript_info'
        ,'ref_device_info'
        ,'ref_browser_info'
        ,'ref_location_info'
        ,'ref_timezone'
        ,'ref_country'
        ,'ref_date_time' 
    ])

    # encode key to md5
    df_output['pbe_match_id'] = df_output[pbe_match_id_cols].values.tolist()
    df_output['pbe_match_id'] = df_output['pbe_match_id'].apply(lambda x: md5_hash(map(str, x)))

    # convert timestamps to string
    timestamp_cols = [col for col in df_output.columns if col.endswith(('_time'))]
    df_output[timestamp_cols] = df_output[timestamp_cols].apply(pd.to_datetime, errors='coerce', utc=True).astype(str)

    print(f"FCM MATCH DATASET created with {len(df_output)} records")
    print(f"FCM MATCH DATASET contains {len(df_output[df_output['pbe_match_confidence_flag']=='Y'])} confident matches")

    return df_output.reset_index(drop=True)

def generate_fcm_process_timestamp():
    """
    - function to generate unique FCM process timestamp
    """
    return datetime.now().astimezone(timezone.utc).strftime('%Y-%m-%d %H:%M:%S.%f')

@ProfileFunction
def process_data(pbe_input, pbe_reference):
    
    # generate pbe process identifiers
    pbe_process_id = generate_fcm_process_id()
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S")  +'::' + f'pbe_process_id: {pbe_process_id}')
    
     # capture process start
    pbe_process_start = generate_fcm_process_timestamp()
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S")  +'::' + f'pbe_process_start: {pbe_process_start}')
    
    print('splitting the column values into ngrams')
    
    # pbe_input['pbe_ip'] = pbe_input['pbe_ip'].apply(lambda x: [get_ngrams(token, None) for token in x])
    # pbe_input['pbe_ip'] = pbe_input['pbe_ip'].apply(lambda x: np.array([item for sublist in x for item in sublist], dtype=str))

    # pbe_reference['pbe_ip'] = pbe_reference['pbe_ip'].apply(lambda x: [get_ngrams(token, None) for token in x])
    # pbe_reference['pbe_ip'] = pbe_reference['pbe_ip'].apply(lambda x: np.array([item for sublist in x for item in sublist], dtype=str))
    
    #print(pbe_reference['pbe_ip'])
    #print(pbe_input['pbe_ip'])
    
    for pbe_attribute in evaluation_config.keys():

        # if fuzzy mode to be applied, use trigrams
        if evaluation_config[pbe_attribute].get('match_mode','').lower() == 'fuzzy':
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S") +'::' + f'fuzzyfying tokens')
            for dataset in [pbe_input, pbe_reference]:
                dataset[f"pbe_{pbe_attribute}"] = dataset[f"pbe_{pbe_attribute}"].apply(lambda x: [get_ngrams(token, evaluation_config[pbe_attribute].get('match_ngrams_length',None)) for token in x])
                dataset[f"pbe_{pbe_attribute}"] = dataset[f"pbe_{pbe_attribute}"].apply(lambda x: np.array([item for sublist in x for item in sublist], dtype=str))

    # instantiate output
    pbe_match_dataset = pbe_match_dataset_initialization()
    reg_pbe_match_dataset=pbe_match_dataset_initialization()
    
    #evaluation_attributes  = ['ip', 'ua_tokenized','device_info']
    evaluation_attributes  = ['ip','ua_tokenized','device_info','browser_info','location_info']
    #evaluation_attributes  = ['ip']
    #evaluation_attributes  = ['ip','ua_tokenized']
    
    print('evaluation_attributes with keys:-->',evaluation_attributes)
    prematch_key_columns=sorted(list(map(lambda x: "pbe_" + x, evaluation_attributes)))
    #print (prematch_key_columns)
    # generate prematch keys
    print(f"generating INPUT prematch key...")
    pbe_input = generate_prematch_key(pbe_input, prematch_key_columns=prematch_key_columns, key_name='pbe_prematch_key')
    #print(pbe_input['pbe_prematch_key'])
    print(f"generating REFERENCE prematch key...")
    pbe_reference = generate_prematch_key(pbe_reference, prematch_key_columns=prematch_key_columns, key_name='pbe_prematch_key')
    
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S")  +'::' + f'vectorizing REFERENCE data...')
    reg_vectorizer = TfidfVectorizer(min_df=1, analyzer=get_ngrams)
    reg_ref_tfidf = reg_vectorizer.fit_transform(pbe_reference['pbe_prematch_key'])
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S")  +'::' + f'REFERENCE data vecorization completed')

    batch_size = 250000 
    # calculate number of batches
    batches_total = math.ceil((len(pbe_input) / batch_size))
    print(f"using batch size of {batch_size} -> total number of batches {batches_total}")
    batch_index_min = 0
    for batch in range(batches_total):
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S")  +'::' + f"processing batch {batch + 1}/{batches_total}")
        # extract indices to be used
        batch_index_min = batch_index_min
        batch_index_max = batch_index_min + batch_size
        
        # subset iput data
        tmp_pbe_input = pbe_input[batch_index_min:batch_index_max].reset_index(drop=True).copy()        
        #tmp_pbe_input = pbe_input.reset_index(drop=True).copy()
        tmp_pbe_reference = pbe_reference.reset_index(drop=True).copy()

        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S")  +'::' + f'INPUT data vecorization started')
        tmp_in_tfidf = reg_vectorizer.transform(tmp_pbe_input['pbe_prematch_key'])
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S")  +'::' + f'INPUT data vecorization completed')
        # apply KNN search
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S")  +'::' + f'KNN search apply started')
        nbrs = NearestNeighbors(
            n_neighbors=prematching_config.get('max_matches', 5), 
            radius=prematching_config.get('distance_threshold', 3), 
            n_jobs=-1, 
            algorithm=prematching_config.get('algorithm', 'auto'),
            metric=prematching_config.get('metric', 'euclidean')
            ).fit(reg_ref_tfidf)

        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S")  +'::' + f'evaluating pre-matches...')
        try:
            _, indices_raw = nbrs.kneighbors(tmp_in_tfidf)

            # extract pre-match indices
            in_indices = np.arange(len(indices_raw), dtype=int)
            ref_indices = indices_raw

            # generate prematch dataset
            tmp_pbe_prematch_dataset = pd.DataFrame({'input_index': in_indices, 'reference_index':[x for x in ref_indices]})
            tmp_pbe_prematch_dataset = tmp_pbe_prematch_dataset.explode('reference_index', ignore_index=True).astype(str)

        except ValueError:
        # remove max_df if ValueError is raised due to not enough terms
            print(f"skipping iteration -> not enough data")
            tmp_pbe_prematch_dataset = None
            tmp_pbe_match_dataset = None

        if tmp_pbe_prematch_dataset is not None and len(tmp_pbe_prematch_dataset) > 0:
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S")  +'::' + f'apply main matching if any prematch...')
    
            tmp_pbe_match_dataset = pbe_matching(
                input_pbe_dataset = tmp_pbe_input, 
                reference_pbe_dataset = tmp_pbe_reference, 
                prematch_mapping = tmp_pbe_prematch_dataset, 
                evaluation_configuration = evaluation_config,
                matching_configuration = matching_config
                )

            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S")  +'::' + f'apply main matching if any prematch...completed')

        if tmp_pbe_match_dataset is not None and len(tmp_pbe_match_dataset) > 0:
            reg_pbe_match_dataset = pd.concat([reg_pbe_match_dataset, tmp_pbe_match_dataset], axis=0, ignore_index=True)
            
        # update min index
        batch_index_min = batch_index_max

        del tmp_pbe_prematch_dataset
        del tmp_pbe_match_dataset

        print(f'batch {batch + 1} matching procedure completed')
        print(f"PBE MATCH DATASET  prepared with {len(reg_pbe_match_dataset)} total records")
        print(f"PBE MATCH DATASET  contains {len(reg_pbe_match_dataset[reg_pbe_match_dataset['pbe_match_confidence_flag']=='Y'])} confident matches")
    
    reg_pbe_match_dataset.dropna(subset=['in_record_id'], inplace=True)
    if reg_pbe_match_dataset is not None and len(reg_pbe_match_dataset) > 0:
        pbe_match_dataset = pd.concat([pbe_match_dataset, reg_pbe_match_dataset], axis=0, ignore_index=True)
            
    print(f"PBE MATCH DATASET prepared with {len(pbe_match_dataset)} total records in total")
    print(f"PBE MATCH DATASET contains {len(pbe_match_dataset[pbe_match_dataset['pbe_match_confidence_flag']=='Y'])} confident matches")

    pbe_match_dataset['pbe_process_id'] = pbe_process_id
    
    del reg_pbe_match_dataset
    
    ######################################################################
    # OUTPUT DATA CLEANING
    ######################################################################
    pbe_process_end = generate_fcm_process_timestamp()
    
    pbe_configuration = {
            'prematching_config': prematching_config,
            'evaluation_config': evaluation_config, 
            'matching_config': matching_config
            }


    # consolidate descriptive process data
    pbe_process_data = {
        'pbe_process_id': pbe_process_id,
        'pbe_process_start': pbe_process_start,
        'pbe_process_end': pbe_process_end,
        'pbe_configuration': pbe_configuration
    }

    return pbe_match_dataset#, pbe_process_data

if __name__ == "__main__":
    # Profile the function
    pbe_match_dataset = process_data(pbe_input, pbe_reference)
    print (pbe_match_dataset)
    pbe_match_dataset.to_csv('C:/Users/G_Subramanian/OneDrive - Dell Technologies/Documents/probablistic matching/test_files/KNN_PBE_result_88_02_09_ip_ua.csv')

    # # Merge the pre-matches with input and reference data to get corresponding IPs
    # pre_match_dataset = pre_match_dataset.astype({'input_index': int, 'reference_index': int})
    # merged_df = pre_match_dataset.merge(pbe_input.reset_index(), left_on='input_index', right_index=True, suffixes=('_input', '_reference'))
    # merged_df = merged_df.merge(pbe_reference.reset_index(), left_on='reference_index', right_index=True, suffixes=('_input', '_reference'))
    
    # # Print the corresponding IPs and whole reference data
    # print(merged_df[['ip_input', 'ip_reference', 'mcvisid_reference', 'record_id_reference', 'all_dellid_reference']])
    