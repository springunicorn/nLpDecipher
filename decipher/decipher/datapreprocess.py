#%%
import sys
import pandas as pd

df = pd.read_csv(
    'gender.csv', encoding='latin1')
#%%
df = df.loc[df['gender:confidence']==1]
df = df[['gender', 'text']]
df.head(2)
#%%
import re
def cleaning(s):
    s = str(s)
    s = s.lower()
    s = re.sub('\s\W', ' ', s)
    s = re.sub('\W\s', ' ', s)
    s = re.sub('\s+', ' ', s)
    s = re.sub(r"http\S+", "", s)
    s = re.sub(r"@\S+", "", s)
    s = re.sub('\s+', ' ', s)
    s = re.sub('\A\s+', '', s)
    s = re.sub('\s+\Z', '', s)
    s = re.sub(r'[^\x00-\x7f]', r'', s)
    s = s.replace('_', '')
    s = re.sub('\s+', ' ', s)
    s = re.sub('\A\s+', '', s)
    s = re.sub('\s+\Z', '', s)
    return s


df['text'] = [cleaning(s) for s in df['text']]
df.head(2)

#%%
import numpy as np
train, validate, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])
train
#%%
train.to_csv('./sentiment/train.tsv', sep='\t', index=False,
          header=False, encoding='utf-8')


#%%
validate.to_csv('./sentiment/dev.tsv', sep='\t', index=False,
             header=False, encoding='utf-8')


#%%
test.to_csv('./sentiment/test.tsv', sep='\t', index=False,
            header=False, encoding='utf-8')


#%%
