import json

#in_f_name = 'nbest_predictions_pred.json'
in_f_name = '../data/dev-evaluate-in1.json'
with open(in_f_name, 'r', encoding="UTF-8") as fp:
    after = json.load(fp)

print(type(json.dumps(after, ensure_ascii=False)))  # <class 'str'>
print(json.dumps(after, ensure_ascii=False))  # ensure_ascii 옵션을 잊지 말자

#out_f_name='nbest_predictions_pred1.json'
out_f_name = '../data/dev-evaluate-in2.json'
with open(out_f_name, 'w', encoding="UTF-8") as fp:
    json.dump(after, fp, ensure_ascii=False, indent=4)