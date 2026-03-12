import pandas as pd
import csv
import json
import xml.etree.ElementTree as ET
import chardet



def detect_encoding(file_path):

    try:
        with open(file_path,'rb') as fp:
            
            raw_data = fp.read(100000)
            result = chardet.detect(raw_data)
            encoding = result['encoding']
            confidence = result['confidence']

            if encoding  == 'ascii':
                print(f"ASCII detected with {confidence:.2%} confidence")
                print("Upgrading to Latin1 (safer for compatibility)")
                return 'latin1'

            print(f"Detected encoding: {encoding} (confidence: {confidence:.2%})")
            return encoding
        
    except Exception as e:

        print(f"Error detecting encoding {e}. Uisng utf-8")
        return'utf-8'


def find_delimiter(file_path):

    print(f"Detecting delimiter for file: {file_path}")

    with open(file_path,'r',encoding='utf-8',errors='ignore') as fp:
        sample = fp.read(2048)
        sniffer = csv.Sniffer()
        try:
            dialect = sniffer.sniff(sample)
            delimiter = dialect.delimiter
            print("Delimiter detected successfully")
        except:
            delimiter = ","
            print("There is a error in finding the delimiter")
            print("going with default delimiter ','")
        
        print(f"The current delimiter is {delimiter}")

    return delimiter

def read_json(file_path):

    try:
        with open(file_path,'r') as fp:
            data = json.load(fp)
        
        if isinstance(data,list):
            df = pd.DataFrame(data)
        elif isinstance(data,dict):
            df = pd.DataFrame([data])
        else:
            print("Unexpected File format")
            df = -1
    except:

            print("Unexpected File format")
            df = -1

    return df

def read_xml_file(file_path):

    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        data = []
        for child in root:
            row = {}
            for elem in child:
                row[elem.tag] = elem.text
            data.append(row)

        df = pd.DataFrame(data)
        return df
    
    except:
        print("Error while reading the file")
        return -1
    
def clean_dataframe(df):


    df = df.dropna(axis=1, how='all')

    unnamed_cols = [col for col in df.columns if str(col).startswith('Unnamed')]
    for col in unnamed_cols:
        if df[col].isna().all() or (df[col]=='').all():
            df = df.drop(columns=[col])

    total_cols = len(df.columns)
    unnamed_count = df.columns.str.contains("Unnamed").sum()

    if unnamed_count == total_cols and total_cols>0:

        df.columns = df.iloc[0]
        df = df[1:].reset_index(drop=True)
        print("All columns were unnamed - promoted first row to column headers")
    elif unnamed_count > 0:
        print(f"Found {unnamed_count} unnamed column(s) out of {total_cols} - keeping them as legitimate data columns")

    df.columns = df.columns.str.strip()

    return df

    
def open_file(file_path):

    normalized = file_path.lower()
    encoding = detect_encoding(file_path)    
    print(encoding)

    try:
        if normalized.endswith(('.xls','.xlsx')):
            df = pd.read_excel(file_path )
        elif normalized.endswith('.parquet'):
            df = pd.read_parquet(file_path )
        elif normalized.endswith('.json'):
            df = read_json(file_path)
        elif normalized.endswith('.xml'):
            df = read_xml_file(file_path)
        else:
            delimiter = find_delimiter(file_path)
            df = pd.read_csv(file_path,delimiter=delimiter,
                             encoding=encoding,
                             skip_blank_lines=True
                             )
            
        df = clean_dataframe(df)
        print("File read successfully")
        return df
    
    except:
        print(f"Error in opening the file {file_path}")
        return None
    

