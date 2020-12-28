The following steps show how to prepare training dataset to train the mode.

# Libraries to install

```
    pip install ftfy langdetect numpy torch pandas nltk sentencepiece boto3 tqdm regex bs4 newspaper3k htmlmin tldextract 
    git clone https://github.com/mattilyra/LSH
    cd LSH
    python setup.py install
``` 

# Download the dataset

1. Download the deduplicated URLs from [jcpeterson](https://mega.nz/#F!EZZD0YwJ!9_PlEQzdMVLaNdKv_ICNVQ!cc4RgQQZ)
2. Remove blacklisted URLs.
```
python blacklist_urls.py <path to the dowloaded deduplicated URLs> <filename for clean urls. e.g. clean_urls.txt>
```
3. Download the content from the clean urls with [openwebtext's utilities](https://github.com/eukaryote31/openwebtext/blob/master/download.py). 

4. Merge the contents into one loose json file with 1 json per newline of the format `{'text': text, 'url': unique_url}`. It is important for the url to be unique.

# Prepare the data for GPT-2 training:

1. Perform ftfy, english detection and remove documents with less than 128 tokens. This step can be sharded and run on shards.
```
python cleanup_dataset.py <input data file> <output cleaned data filename>
```
2. Using LSH, find possible duplicates and store then in a file for later processing. This step can NOT be sharded and usually takes 12 to 24 hours for OpenWebText dataset.
```
python find_duplicates.py <input cleaned data file> <output possible duplicate urls filename>
```
3. Based on similarity measure defind inside function `is_similar` (default: 0.9), group urls that are similar. Basically, for each group, only one url we should keep and remove the rest.
```
python group_duplicate_urls.py <possible duplicate urls file> <output file containing similar urls>
```
4. Remove similar documents that were detected in the last step.
```
python remove_group_duplicates.py <file containing simialr documents> <cleaned data file> <outputfile containing deduplicate data>
```

5. Shuffle the dataset.
```
shuf <cleaned deduped data file> -o train_data.json
```

