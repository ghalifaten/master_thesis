import pickle 
from datasets import Dataset, DatasetDict, concatenate_datasets
from tqdm import tqdm
import json

def batch_generator(data, batch_size):
    keys = list(data.keys())
    total_size = len(data[keys[0]])
    for i in range(0, total_size, batch_size):
        batch = {key: data[key][i:i+batch_size] for key in keys}
        yield batch

def create_dataset(data, batch_size, N):
    dataset = None
    pbar = tqdm(total=N, desc="Creating dataset")
    for batch in batch_generator(data, batch_size):
        batch_dataset = Dataset.from_dict(batch)
        if dataset is None:
            dataset = batch_dataset
        else:
            dataset = concatenate_datasets([dataset, batch_dataset])
        pbar.update(1)
    return dataset
    
if __name__=="__main__":
    print("Loading triplets")
    with open('data/triplets.pkl', 'rb') as f:
        triplets = pickle.load(f) 

    # dataset = Dataset.from_dict(triplets) 
    batch_size = 10000
    N = len(triplets["anchor"]) // batch_size
    dataset = create_dataset(triplets, batch_size, N)

    # # Make splits: train, test, validation
    # train_test = dataset.train_test_split(test_size=0.3)
    # test_val = train_test["test"].train_test_split(test_size=0.33)
    
    # # Recreate Dataset with the three splits 
    # dataset_dict = DatasetDict({
    #     'train': train_test['train'],
    #     'test': test_val['train'],
    #     'validation': test_val['test']
    # })

    # for split, dataset in dataset_dict.items():
    #     json_file_path = f"data/{split}.json"
    #     dataset.to_json(json_file_path)

    dataset.to_json("data/dataset.json")