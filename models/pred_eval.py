import argparse
import json

def evaluate_predictions(pred_filename, gold_filename):
  
  '''
    returns first word accuracy, exact match accuracy
  '''
  with open(pred_filename, "r") as pred_f, open(gold_filename) as gold_f:
    pred_lines = pred_f.readlines()
    gold_lines = gold_f.readlines()
    
    total = 0.0
    full_correct = 0.0
    first_correct = 0.0
    for i in range(len(pred_lines)):
      pred_line = pred_lines[i].strip()
      if gold_filename.endswith(".json"):
        gold_json = json.loads(gold_lines[i])
        gold_line = gold_json["translation"]["tgt"]
      else:  
        gold_line = gold_lines[i].strip().split("\t")[1]
      # remove space before period/question mark
      gold_line = gold_line.replace(" ?", "?").replace(" .", ".").replace(" ,", ",") 
      
      total +=1
      
      if pred_line == gold_line:
        full_correct += 1
        first_correct += 1
      else:
        pred_words = pred_line.split()
        gold_words = gold_line.split()
        if len(pred_words) > 0 and pred_words[0] == gold_words[0]:
          first_correct += 1
      
  
  return  (first_correct / total), (full_correct / total)
      
  


def main():
  
  argparser = argparse.ArgumentParser()
  argparser.add_argument("--pred_file", type=str, required=True)
  argparser.add_argument("--gold_file", type=str, required=True)
  
  args = argparser.parse_args()
  
  
  first_acc, full_acc = evaluate_predictions(args.pred_file, args.gold_file)
  
  print(f"Exact match accuracy: {full_acc}")
  print(f"First word accuracy: {first_acc}")
  
  

if __name__ == '__main__':
  main()
