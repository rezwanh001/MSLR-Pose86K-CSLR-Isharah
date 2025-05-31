import pandas as pd
import numpy as np
import torch
from utils.metrics import normalize_gloss_sequence

class GaussianNoise(object):
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) or (H, W)
        Returns:
            Tensor: Image with added Gaussian noise
        """
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.from_numpy(np.array(tensor))

        noise = torch.randn_like(tensor) * self.std + self.mean
        return tensor + noise

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"

def remove_duplicates(x):
    if len(x) < 2:
        return x
    fin = ""
    for j in x:
        if fin == "":
            fin = j
        else:
            if j == fin[-1]:
                continue
            else:
                fin = fin + j
    return fin


def decode_predictions(preds, encoder):
    """
    Decodes CTC predictions into gloss sequences.
    - Converts logits to probabilities (softmax).
    - Gets the highest probability gloss at each timestep.
    - Converts numerical predictions back into gloss words.
    - Removes duplicate glosses.
    """
    preds = torch.softmax(preds, 2)  # Convert logits to probabilities
    preds = torch.argmax(preds, 2)  # Get most likely gloss per frame
    preds = preds.detach().cpu().numpy()  # Convert to NumPy

    sign_preds = []
    for j in range(preds.shape[0]):  # Iterate over batch
        temp = []
        for k in preds[j, :]:
            k = k - 1  # Shift index to match vocabulary
            if k == -1:
                temp.append("§")  # Placeholder for blank (CTC)
            else:
                p = encoder.inverse_transform([k])[0]  # Convert number to gloss
                temp.append(p)

        gloss_seq = " ".join(temp).replace("§", "")  # Remove blank characters
        sign_preds.append(remove_duplicates(gloss_seq))  # Remove duplicate glosses

    return sign_preds

def numerize(sents, vocab_map, full_transformer):
    """
    Converts gloss sequences into numerical format.
    """
    outs = []
    for sent in sents:
        if type(sent) != float:
            if full_transformer:
                outs.append([32] + [vocab_map[g] for g in sent.split()] + [0])  # Add BOS and EOS
            else:
                outs.append([vocab_map[g] for g in sent.split()])
    return outs

def invert_to_chars(sents, inv_ctc_map):
    sents = sents.detach().numpy()
    outs = []
    for sent in sents:
        for x in sent:
            if x == 0:
                break
            outs.append(inv_ctc_map[x]) 
    return outs

def get_ctc_vocab(char_list):
    # blank
    ctc_char_list = "_" + char_list
    ctc_map, inv_ctc_map = {}, {}
    for i, char in enumerate(ctc_char_list):
        ctc_map[char] = i
        inv_ctc_map[i] = char
    return ctc_map, inv_ctc_map, ctc_char_list

def get_autoreg_vocab(char_list):
    # blank
    ctc_map, inv_ctc_map = {}, {}
    for i, char in enumerate(char_list):
        ctc_map[char] = i
        inv_ctc_map[i] = char
    return ctc_map, inv_ctc_map, char_list


import pandas as pd

def convert_text_for_ctc(dataset_name, train_csv, dev_csv):
    """
    Reads CSLR annotation CSVs, extracts vocabulary, and encodes annotations for CTC training.

    Args:
        train_csv (str): Path to training annotation file.
        dev_csv (str): Path to development annotation file.

    Returns:
        tuple: (Processed DataFrames, vocab_map, inv_vocab_map, vocab_list)
    """

    # Load all CSVs
    train_data = pd.read_csv(train_csv, delimiter="|")
    dev_data = pd.read_csv(dev_csv, delimiter="|")

    # Concatenate all data
    all_data = pd.concat([train_data, dev_data])

    # Remove rows where filename or annotation is missing
    if "isharah" in dataset_name.lower() or "csl" in dataset_name.lower():
        
        all_data = all_data[all_data['id'].notna()]
        
        #all_data = all_data[all_data['annotation'].notna()]
        all_data = all_data[all_data['gloss'].notna()]

        # Extract all glosses and remove duplicates
        all_glosses = set()
        for annotation in all_data["gloss"]:
            annotation = normalize_gloss_sequence(annotation)
            glosses = annotation.split()  # Split into words
            all_glosses.update(glosses)  # Add unique glosses

        # Create vocabulary mappings
        vocab_list = ["_"] + sorted(all_glosses)  # Ensure "_" is at index 0
        vocab_map = {g: i for i, g in enumerate(vocab_list)}  # "_": 0, "HELLO": 1, "WORLD": 2
        inv_vocab_map = {i: g for i, g in enumerate(vocab_list)}

        print(f"Extracted Vocabulary Size: {len(vocab_map)}")

        # Function to encode annotations into numerical format
        def encode_annotations(df):
            df = df.copy()
         #   print(df)
            # Apply normalization to the annotation string
            df["gloss"] = df["gloss"].apply(normalize_gloss_sequence)
            df["enc"] = df["gloss"].apply(lambda x: [vocab_map[g] for g in x.split()])  # Convert glosses to numbers
            return df[["id", "enc"]]  # Keep only necessary columns
        
    else: #id|folder|signer|annotation
        all_data = all_data[all_data['id'].notna()]
        
        #all_data = all_data[all_data['annotation'].notna()]
        all_data = all_data[all_data['annotation'].notna()]

        # Extract all glosses and remove duplicates
        all_glosses = set()
        for annotation in all_data["annotation"]:
            annotation = normalize_gloss_sequence(annotation)
            glosses = annotation.split()  # Split into words
            all_glosses.update(glosses)  # Add unique glosses

        # Create vocabulary mappings
        vocab_list = ["_"] + sorted(all_glosses)  # Ensure "_" is at index 0
        vocab_map = {g: i for i, g in enumerate(vocab_list)}  # "_": 0, "HELLO": 1, "WORLD": 2
        inv_vocab_map = {i: g for i, g in enumerate(vocab_list)}

        print(f"Extracted Vocabulary Size: {len(vocab_map)}")

        # Function to encode annotations into numerical format
        def encode_annotations(df):
            df = df.copy()
            print(df)
            # Apply normalization to the annotation string
            df["annotation"] = df["annotation"].apply(normalize_gloss_sequence)
            df["enc"] = df["annotation"].apply(lambda x: [vocab_map[g] for g in x.split()])  # Convert glosses to numbers
            return df[["id", "enc"]]  # Keep only necessary columns
        

    train_processed = encode_annotations(train_data)
    print("processsed train")
    dev_processed = encode_annotations(dev_data)
    print("processsed dev")

    return train_processed, dev_processed, vocab_map, inv_vocab_map, vocab_list