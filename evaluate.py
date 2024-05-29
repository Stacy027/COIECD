from collections import Counter
import re
import string
import json

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def evaluate_em(gold_answers, predictions):
    em_score = 0
    for ground_truth, prediction in zip(gold_answers, predictions):
        em_score += exact_match_score(prediction, ground_truth)
    em_score /= len(gold_answers)
    return round(em_score, 4)

def evaluate_f1(gold_answers, predictions):
    f1_total = 0
    for ground_truth, prediction in zip(gold_answers, predictions):
        f1_total += f1_score(prediction, ground_truth)
    mean_f1 = f1_total / len(gold_answers)
    return round(mean_f1, 4)

if __name__ == '__main__':
    data_path = './data_path'
    with open(data_path, 'r') as f:
        lines = json.load(f)

    labels = []
    cad_answers = []
    cond_answers = []
    uncond_answers = []
    
    for line in lines:
        id = line['id']
        label = line['True Answer'].lower()
        
        coiecd_answer = line['coiecd_answer'].lower()
        cond_answer = line['cond_answer'].lower()
        uncond_answer = line['uncond_answer'].lower()
        
        labels.append(label)
        cad_answers.append(coiecd_answer)
        cond_answers.append(cond_answer)
        uncond_answers.append(uncond_answer)

    print(len(labels))
    print('|------------Total eval dataset-----------------|')        
    print('EM_CAD:', evaluate_em(labels, cad_answers))
    print('EM_context:', evaluate_em(labels, cond_answers))
    print('EM_w/o_context:', evaluate_em(labels, uncond_answers))

    print('F1_CAD:', evaluate_f1(labels, cad_answers))
    print('F1_context:', evaluate_f1(labels, cond_answers))
    print('F1_w/o_context:', evaluate_f1(labels, uncond_answers))

    
    alpha = 0.5
    labels = []
    cad_answers = []
    cond_answers = []
    uncond_answers = []

    
    for line in lines:
        id = line['id']
        label = line['True Answer'].lower()
        if label == "true":
            label = 'yes'
        elif label == "false":
            label = 'no'
        coiecd_answer = line['coiecd_answer'].lower()
        cond_answer = line['cond_answer'].lower()
        uncond_answer = line['uncond_answer'].lower()
    
        if not (f1_score(label, uncond_answer) >= alpha):
            labels.append(label)
            cad_answers.append(coiecd_answer)
            cond_answers.append(cond_answer)
            uncond_answers.append(uncond_answer)
            
    
    print('|---------Conf. knowledge in model--------|')
            
    print('EM_CAD:', evaluate_em(labels, cad_answers))
    print('EM_context:', evaluate_em(labels, cond_answers))
    print('EM_w/o_context:', evaluate_em(labels, uncond_answers))

    print('F1_CAD:', evaluate_f1(labels, cad_answers))
    print('F1_context:', evaluate_f1(labels, cond_answers))
    print('F1_w/o_context:', evaluate_f1(labels, uncond_answers))

    labels = []
    cad_answers = []
    cond_answers = []
    uncond_answers = []
    
    for line in lines:
        id = line['id']
        label = line['True Answer'].lower()
        if label == "true":
            label = 'yes'
        elif label == "false":
            label = 'no'
        coiecd_answer = line['coiecd_answer'].lower()
        cond_answer = line['cond_answer'].lower()
        uncond_answer = line['uncond_answer'].lower()
        

        if f1_score(label, uncond_answer) >= alpha:
            labels.append(label)
            cad_answers.append(coiecd_answer)
            cond_answers.append(cond_answer)
            uncond_answers.append(uncond_answer)
            
    
    print('|---------Non-Conf. knowledge in model------------|')
            
    print('EM_CAD:', evaluate_em(labels, cad_answers))
    print('EM_context:', evaluate_em(labels, cond_answers))
    print('EM_w/o_context:', evaluate_em(labels, uncond_answers))

    print('F1_CAD:', evaluate_f1(labels, cad_answers))
    print('F1_context:', evaluate_f1(labels, cond_answers))
    print('F1_w/o_context:', evaluate_f1(labels, uncond_answers))
