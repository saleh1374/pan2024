import os
import argparse
import json
import glob
import re
from transformers import pipeline #AutoModelForSequenceClassification, AutoTokenizer

def pars_args():
    parser= argparse.ArgumentParser(description= "PAN 2024 Style Change Detection Task.")
    parser.add_argument("--input", type= str, help= "Folder containing input files for task(.txt)", required= True)
    parser.add_argument("--output", type= str, help= "Folder containing output/solution files(.json)", required= True)
    args = parser.parse_args()

    return args

def read_problem_files(problems_folder):
    problems = {}
    
    solution_files = glob.glob(f'{problems_folder}/problem-*.txt') \
        + glob.glob(f'{problems_folder}/**/problem-*.txt') \
        + glob.glob(f'{problems_folder}/**/**/problem-*.txt')
    
    
    for file in solution_files:
        file_num= re.findall(r'\d+', str(file))[0]
        with open(file, 'r') as f:
            data= []
            problem= f.readlines()
            for i in range(len(problem)-1):
                data.append(problem[i] + ' ' + problem[i + 1])
    #             # print('append : '+ str(lines[i]) + ' ' + str(lines[i + 1]))
            problems[f'problem-{file_num}']= data
    return problems
            


def easy_test(problems, output_path):
    classifier= pipeline('text-classification', model='MohammadKarami/simple-roberta', tokenizer="MohammadKarami/simple-roberta", max_length= 512, truncation= True)
    os.makedirs(output_path, exist_ok= True)
    for problem in problems:
        print(f'{problem} file is working...')
        preds= []
        for text in problems[problem]:
            output= classifier(text)
            if output[0]['label']== 'isnt':
                preds.append(0)
            else:
                preds.append(1)
        file_num= re.findall(r'\d+', problem)[0]
        with open(output_path+"/solution-"+file_num+".json", 'w') as out:
            prediction = {'changes': preds}
            out.write(json.dumps(prediction)) 

def most_frequent(List):
    return max(set(List), key = List.count)

def medium_test(problems, output_path):
    roberta_classifier= pipeline('text-classification', model='MohammadKarami/medium-roberta', tokenizer="MohammadKarami/medium-roberta", max_length= 512, truncation= True)
    electra_classifier= pipeline('text-classification', model='MohammadKarami/medium-electra', tokenizer="MohammadKarami/medium-electra", max_length= 512, truncation= True)
    bert_classifier= pipeline('text-classification', model='MohammadKarami/medium-bert', tokenizer="MohammadKarami/medium-bert", max_length= 512, truncation= True)
    whole_roberta_classifier= pipeline('text-classification', model='MohammadKarami/whole-roBERTa', tokenizer="MohammadKarami/whole-roBERTa", max_length= 512, truncation= True)
    whole_electra_classifier= pipeline('text-classification', model='MohammadKarami/whole-electra', tokenizer="MohammadKarami/whole-electra", max_length= 512, truncation= True)

    os.makedirs(output_path, exist_ok= True)

    for problem in problems:
        print(f'{problem} file is working...')
        preds= []
        sample_pred= []

        for text in problems[problem]:
            roberta_output= roberta_classifier(text)
            if roberta_output[0]['label']== 'isnt':
                sample_pred.append(0)
            else:
                sample_pred.append(1)

            electra_output= electra_classifier(text)
            if electra_output[0]['label']== 'isnt':
                sample_pred.append(0)
            else:
                sample_pred.append(1)

            bert_output= bert_classifier(text)
            if bert_output[0]['label']== 'isnt':
                sample_pred.append(0)
            else:
                sample_pred.append(1)

            whole_roberta_output= whole_roberta_classifier(text)
            if whole_roberta_output[0]['label']== 'isnt':
                sample_pred.append(0)
            else:
                sample_pred.append(1)

            whole_electra_output= whole_electra_classifier(text)
            if whole_electra_output[0]['label']== 'isnt':
                sample_pred.append(0)
            else:
                sample_pred.append(1)
            preds.append(most_frequent(sample_pred))
        file_num= re.findall(r'\d+', problem)[0]
        with open(output_path+"/solution-"+file_num+".json", 'w') as out:
            prediction = {'changes': preds}
            out.write(json.dumps(prediction))

def hard_test(problems, output_path):
    roberta_classifier= pipeline('text-classification', model='MohammadKarami/hard-roberta', tokenizer="MohammadKarami/hard-roberta", max_length= 512, truncation= True )
    electra_classifier= pipeline('text-classification', model='MohammadKarami/hard-electra', tokenizer="MohammadKarami/hard-electra", max_length= 512, truncation= True)
    whole_roberta_classifier= pipeline('text-classification', model='MohammadKarami/whole-roBERTa', tokenizer="MohammadKarami/whole-roBERTa", max_length= 512, truncation= True)

    os.makedirs(output_path, exist_ok= True)

    for problem in problems:
        print(f'{problem} file is working...')
        preds= []
        sample_pred= []

        for text in problems[problem]:
            roberta_output= roberta_classifier(text)
            if roberta_output[0]['label']== 'isnt':
                sample_pred.append(0)
            else:
                sample_pred.append(1)

            electra_output= electra_classifier(text)
            if electra_output[0]['label']== 'isnt':
                sample_pred.append(0)
            else:
                sample_pred.append(1)

            whole_roberta_output= whole_roberta_classifier(text)
            if whole_roberta_output[0]['label']== 'isnt':
                sample_pred.append(0)
            else:
                sample_pred.append(1)

            preds.append(most_frequent(sample_pred))

        file_num= re.findall(r'\d+', problem)[0]
        with open(output_path+"/solution-"+file_num+".json", 'w') as out:
            prediction = {'changes': preds}
            out.write(json.dumps(prediction))


def main():
    args= pars_args()

    for subtask in ['hard']:#, 'easy', 'medium']:
        if subtask =='easy':
            problems= read_problem_files(args.input+f"/{subtask}")
            easy_test(problems, args.output)
        elif subtask=='medium':
            problems= read_problem_files(args.input+f"/{subtask}")
            medium_test(problems, args.output)
        else:
            problems= read_problem_files(args.input+f"/{subtask}")
            hard_test(problems, args.output)


    



if __name__=='__main__':
    main()
