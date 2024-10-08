import os, sys
import glob
import pandas as pd

valid=['11','12','13','14','15']
test=['16','17','18','19','20']

root='/content/drive/Shareddrives/専門演習I/後期前半/SpeechData/'
data={'path':[], 'data_type':[], 'speaker': [] }
for path in glob.glob(os.path.join(root, '*/*wav')):
    print(path)
    speaker = os.path.dirname(path)
    data['path'].append(os.path.join(root, file))
    num=os.path.splitext(os.path.basename(path))[0]).split('_')[1]
    if num in valid:
        data['data_type'].append('valid')
    elif num in test:
        data['data_type'].append('eval')
    else:
        data['data_type'].append('train')
    data['speaker'].append(speaker)
   
df = pd.DataFrame.from_dict(data, orient='columns')
df.to_csv(os.path.join(root, 'data.csv'), index=False)
