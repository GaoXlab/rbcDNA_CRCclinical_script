import sys
import time
from collections import defaultdict
from typing import List, Tuple, Dict

import pandas as pd

_koala_gap = 0

class Info:
    def __init__(self):
        self.seq_id = ""
        self.label = 0
        self.stage = ""
        self.stage_num = 0.0
        self.sex = 0
        self.age = 0

info_dict: Dict[str, Info] = {}

def load_sample_info(input_info: str) -> None:
    print("Loading sample info...")
    with open(input_info, 'r') as f_info:
        n = int(f_info.readline())
        for _ in range(n):
            info = Info()
            parts = f_info.readline().split()
            info.seq_id = parts[0]
            info.label = int(parts[1])
            info.stage = parts[2]
            info.stage_num = float(parts[3])
            info.sex = int(parts[4])
            info.age = int(parts[5])
            info_dict[info.seq_id] = info
    print(f"Total Size: {len(info_dict)}")

def get_seq_id(file_name: str) -> List[str]:
    ret = []
    try:
        if 'gam' in file_name:
            with open(file_name, 'r') as inp:
                line = inp.readline()
                ret = line.split('\t')[3:]
                ret = [i.replace("\n", "") for i in ret]
        else:
            header = pd.read_csv(file_name, sep='\t', header=None, nrows=1, quotechar="'", dtype=str)
            ret = header.iloc[0, 3:].tolist()
            ret = [x.split('.')[0] for x in ret]
    except IOError:
        print(f"FILE_ERROR {file_name}")
    return ret

def load_sample_label(input_tab: str) -> List[int]:
    ids = get_seq_id(input_tab)
    label = []
    for seq_id in ids:
        if seq_id in info_dict:
            label.append(info_dict[seq_id].label)
        else:
            label.append(-1)
    return label

def get_score(v: List[Tuple[float, int]]) -> Tuple[int, float]:
    v.sort()
    if v[0][0] + 0.1 >= v[-1][0]:
        return (0, 0)
    
    tot_pos = sum(1 for _, label in v if label == 1)
    tot_neg = sum(1 for _, label in v if label == 0)
    
    res = (0, 0)
    cur_pos = 0
    cur_neg = 0
    
    for i in range(len(v) - 1):
        if v[i][1] == 0:
            cur_neg += 1
        elif v[i][1] == 1:
            cur_pos += 1
            
        score = max(
            cur_neg + (tot_pos - cur_pos),
            cur_pos + (tot_neg - cur_neg)
        )
        
        gap = v[i+1][0] - v[i][0]
        if gap < _koala_gap or gap < 1e-6:
            continue
            
        if score > res[0] or (score == res[0] and gap > res[1]):
            res = (score, gap)
    
    return res

def load_detail(input_tab: str, output_bed: str, label: List[int]) -> None:
    v = [(0.0, lbl) for lbl in label]
    
    with open(input_tab, 'r') as inp, open(output_bed, 'w') as ou:
        inp.readline()  # Skip header
        while True:
            line = inp.readline()
            if not line:
                break
                
            parts = line.split()
            if len(parts) < 4:
                continue
                
            for i in range(len(label)):
                v[i] = (float(parts[i+3]), label[i])
                
            res = get_score(v)
            chr_id = int(float(parts[0]) + 1e-6)
            
            if chr_id <= 22:
                chr_str = str(chr_id)
            else:
                chr_str = 'X' if chr_id == 23 else 'Y' if chr_id == 24 else 'M'
                
            ou.write(f"{chr_str}\t{int(float(parts[1]) + 1e-6)}\t"
                    f"{int(float(parts[2]) + 1e-6)}\t{res[0]}\t{res[1]:.4f}\n")

def main():
    if len(sys.argv) < 2:
        print("usage: python dim_reduction_single_step.py tab_id")
        return
        
    input_info = "./all.sample.info"
    input_tab = f"./{sys.argv[1]}" # "./train_gam.tab.17" 
    output_bed = f"./{sys.argv[1]}.bed" #"./train_gam.tab.17.bed" 

    print("======= dim selection ======")
    print("Expect runtime: 20s")
    
    start_time = time.time()
    load_sample_info(input_info)
    label = load_sample_label(input_tab)
    load_detail(input_tab, output_bed, label)
    
    print("Load complete.")
    print(f"Time elapsed: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
