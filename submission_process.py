import os
import csv
import argparse
import zipfile


def parse_dev_file(dev_path):
    with open(dev_path, 'r', encoding='utf-8') as f:
        lines = f.read().strip().split('\n')[1:]  # skip header
        ids = [line.split('|')[0] for line in lines]
    return ids


def parse_predictions_file(pred_path):
    with open(pred_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    preds = []
    for i in range(len(lines)):
        if lines[i].strip().startswith("Pred:"):
            pred = lines[i].strip().replace("Pred:", "").strip()
            if not pred:
                pred = "[blank]"  # or "" if you prefer
            preds.append(pred)
    return preds


def save_csv(ids, preds, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'gloss'])
        for _id, pred in zip(ids, preds):
            pred = str(pred) if pred else "[blank]"  # ensure string
            writer.writerow([_id, pred])


def make_zip(file_path, zip_path):
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(file_path, arcname=os.path.basename(file_path))


def main(dev_file, pred_file, output_dir):
    ids = parse_dev_file(dev_file)
    preds = parse_predictions_file(pred_file)

    assert len(ids) == len(preds), "Mismatch between number of IDs and predictions."

    out_csv = os.path.join(output_dir, 'dev.csv')
    save_csv(ids, preds, out_csv)

    zip_path = os.path.join(output_dir, 'dev.zip')
    make_zip(out_csv, zip_path)

    print(f"[✔] Submission saved to: {zip_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev_file', type=str, default='./annotations_v2/US/dev.txt', help='Path to dev.txt file')
    parser.add_argument('--pred_file', type=str, default='./work_dir/base_US/pred_outputs/predictions_epoch_37.txt', help='Path to prediction file')
    parser.add_argument('--output_dir', type=str, default='./submission/task-2', help='Directory to save dev.csv and dev.zip')
    args = parser.parse_args()

    main(args.dev_file, args.pred_file, args.output_dir)

'''
Task-1 : SI

python submission_process.py --dev_file ./annotations_v2/SI/dev.txt --pred_file ./work_dir/base_SI/pred_outputs/predictions_epoch_40.txt --output_dir ./submission/task-1

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 
Task-2 : US

python submission_process.py --dev_file ./annotations_v2/US/dev.txt --pred_file ./work_dir/base_US/pred_outputs/predictions_epoch_40.txt --output_dir ./submission/task-2

'''