import os, sys
import glob
import pandas as pd

root='/content/drive/Shareddrives/専門演習I/後期前半/SpeechData/'
data={'path':[], 'data_type':[], 'speaker': [] }
for file in glob.glob(os.path.join(root, '*/*wav')):
    print(file)
    speaker = os.path.dirname(file)
    #data['path'].append(os.path.join(root, file))
    #data['data_type'].append('train')
    #data['speaker'].append(speaker)
   
#df = pd.DataFrame.from_dict(data, orient='columns')
#df.to_csv('data.csv', index=False)
