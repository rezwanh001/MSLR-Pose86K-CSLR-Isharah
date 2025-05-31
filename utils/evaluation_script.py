
import numpy as np

# ==== WER COSTS ====
WER_COST_DEL = 3
WER_COST_INS = 3
WER_COST_SUB = 4



#==== Edit distance + alignment ====
def edit_distance(r, h):
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=np.uint8)
    for i in range(len(r) + 1): d[i][0] = i * WER_COST_DEL
    for j in range(len(h) + 1): d[0][j] = j * WER_COST_INS
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(
                    d[i - 1][j - 1] + WER_COST_SUB,
                    d[i][j - 1] + WER_COST_INS,
                    d[i - 1][j] + WER_COST_DEL
                )
    return d

def get_alignment(r, h, d):
    x, y = len(r), len(h)
    alignlist = []
    while x > 0 or y > 0:
        if x > 0 and y > 0 and d[x][y] == d[x - 1][y - 1] and r[x - 1] == h[y - 1]:
            alignlist.append("C")
            x, y = x - 1, y - 1
        elif x > 0 and y > 0 and d[x][y] == d[x - 1][y - 1] + WER_COST_SUB:
            alignlist.append("S")
            x, y = x - 1, y - 1
        elif y > 0 and d[x][y] == d[x][y - 1] + WER_COST_INS:
            alignlist.append("I")
            y -= 1
        else:
            alignlist.append("D")
            x -= 1
    return alignlist[::-1]

def wer_single(r, h):
    r = r.strip().split()
    h = h.strip().split()
    if len(r) == 0:
        return {
            "num_del": 0,
            "num_ins": len(h),
            "num_sub": 0,
            "num_err": len(h),
            "num_ref": 1e-8  # to avoid zero division
        }
    d = edit_distance(r, h)
    alignment = get_alignment(r, h, d)
    num_del = alignment.count("D")
    num_ins = alignment.count("I")
    num_sub = alignment.count("S")
    num_err = num_del + num_ins + num_sub
    return {
        "num_del": num_del, "num_ins": num_ins, "num_sub": num_sub,
        "num_err": num_err, "num_ref": len(r)
    }
def wer_list(refs, hyps):
    total_err = total_del = total_ins = total_sub = total_ref = 0
    for r, h in zip(refs, hyps):
        # r_norm = normalize_gloss_sequence(r)
        # h_norm = normalize_gloss_sequence(h)
        res = wer_single(r, h)
        total_err += res["num_err"]
        total_del += res["num_del"]
        total_ins += res["num_ins"]
        total_sub += res["num_sub"]
        total_ref += res["num_ref"]
    return {
        "wer": (total_err / total_ref) * 100,
        "del": (total_del / total_ref) * 100,
        "ins": (total_ins / total_ref) * 100,
        "sub": (total_sub / total_ref) * 100,
    }

# ==== Load predictions and ground truth ====
def read_txt_lines(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = [line.strip() if line.strip() else '[blank]' for line in f]
    return lines

# ==== Main ====
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', type=str, required=True, help='Predictions file')
    parser.add_argument('--gt', type=str, required=True, help='Ground truth file')
    args = parser.parse_args()

    preds = read_txt_lines(args.pred)
    refs = read_txt_lines(args.gt)

    assert len(preds) == len(refs), "Mismatch in number of lines"

    results = wer_list(refs, preds)
    print(f"WER:  {results['wer']:.2f}%")
    print(f"DEL:  {results['del']:.2f}%")
    print(f"INS:  {results['ins']:.2f}%")
    print(f"SUB:  {results['sub']:.2f}%")
    
    # run like  python evaluation_script.py --pred predictions.txt --gt groundtruth.txt