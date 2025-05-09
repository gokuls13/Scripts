#current ngrams function
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
    
for dataset in [pbe_input, pbe_reference]:
                dataset[f"pbe_{pbe_attribute}"] = dataset[f"pbe_{pbe_attribute}"].apply(lambda x: [get_ngrams(token, evaluation_config[pbe_attribute].get('match_ngrams_length',None)) for token in x])
                dataset[f"pbe_{pbe_attribute}"] = dataset[f"pbe_{pbe_attribute}"].apply(lambda x: np.array([item for sublist in x for item in sublist], dtype=str))


###################################################################################################
import pandas as pd
import re
import numpy as np
data = {
    'browser_info': [
        'chrome 124.0|chrome 123.0~google~{1920x929},123.0,123.0,123.0',
        'safari 17.4~apple~{414x721,414x829}',
        'chrome 123.0~google~{2560x1305}',
        'chrome 123.0~google~{842x328}',
        'chrome 124.0|chrome 123.0~google~{1920x1031}',
        'safari~apple~{375x543}',
        'chrome 123.0~google~{1366x641}',
        'chrome 123.0~google~{412x800}',
        ''
    ]
}
df = pd.DataFrame(data)
print (df)
# def get_ngrams(
#     text: str=None, 
#     n: int=None, 
#     min_n: int=2
#     ):
#     """
#     - function prepare n-grams from a text string 
#     """
#     # remove leading and trailing whitespace + convert to uppercase 
#     #text = re.sub(r'\W', r' ', text.upper().strip())
#     #Remove dots and combine characters, then convert to uppercase and strip whitespace
#     text = re.sub(r'\.', '', text.upper().strip())

#     # get n-grams
#     if n is not None:
#         if len(text) < n:
#             #return ['UNIDENTIFIABLE']
#             # Return all characters if the length of text is less than n
#             ngram_list = [text]
#         else:
#             ngrams = zip(*[text[i:] for i in range(n)])
#             ngram_list = [''.join(ngram) for ngram in ngrams]

#     # if n is not specified, get list of words
#     else:
#         ngrams = re.split('\s+', text)
#         ngram_list = [i for i in list(set(ngrams)) if len(i) >= min_n and i!='']

#     ngram_list = ngram_list if ngram_list != [] else ['UNIDENTIFIABLE']

#     return ngram_list 
def get_ngrams(text: str=None, n: int=None, min_n: int=2):
       
    if text!="":
   
        text = re.split(r'[|~{,]+', text.upper().strip())
   
   
        text = ' '.join(text)  # Join the list into a string
        text = re.sub(r'\.', '', text)
        text=re.sub(r'}','',text)
        print(text)
   
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


# df['browser_info'] = df['browser_info'].apply(lambda x: [get_ngrams(token) for token in re.split(r'[\|~,\s]', x)])
# df['browser_info'] = df['browser_info'].apply(lambda x: np.array([item for sublist in x for item in sublist], dtype=str))
df['browser_info'] = df['browser_info'].apply(lambda x: get_ngrams(x,5))
df['browser_info'] = df['browser_info'].apply(lambda x: np.array(x, dtype=str))
 
print(df['browser_info'])
