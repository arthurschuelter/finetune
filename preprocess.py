import os
import json

path = './downloads'
files = [f for f in os.listdir(path) if f.endswith('.json')]

# print(f'Found {len(files)} JSON files in {path}')

def file_to_messages(f):
    with open(os.path.join(path, f), 'r') as file:
        data = json.load(file)
        segments = data.get('segments', [])
        texts = [segment.get('text', '') for segment in segments]
        return texts
    
def to_dataset(texts):
    roles = ['assistant', 'user']
    formatted = []

    for i, msg in enumerate(texts):
        role = roles[i % 2]
        formatted.append({
            "role": role,
            "content": msg
        })

    return {"messages": formatted}

dataset = []
for i, f in enumerate(files):
    print(f'\nFile {i+1} - Found file: {f}')
    texts = file_to_messages(f)
    datum = to_dataset(texts)
    # print(datum)
    dataset.append(datum)
    # for t in texts:
    #     print(f'    {t}')

dataset = {"train": dataset}
def save_as_json(data):
    with open('dataset.json', 'w') as f:
        json.dump(data, f, indent=4)

save_as_json(dataset)   
