"""
https://okvqa.allenai.org/
"""

import os
import time

from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
import json
import re

from datasets import general_postprocessing, all_answers_from_dict

import nltk
nltk.download("punkt")
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

def stem_sentence(sentence):
    stemmer = PorterStemmer()
    tokenized_words = word_tokenize(sentence)
    stemmed_words = [stemmer.stem(word) for word in tokenized_words]
    return ' '.join(stemmed_words)

def most_common_from_dict(dct):
    lst = [x["answer"] for x in dct]
    return max(set(lst), key=lst.count)


def most_common_from_dict_raw(dct):
    lst = [x["raw_answer"] for x in dct]
    return max(set(lst), key=lst.count)


class OKVQADataset(Dataset):
    IMAGE_PATH = {
        "train": ("train2014", "OpenEnded_mscoco_train2014_questions.json", "mscoco_train2014_annotations.json"),
        "test": ("val2014", "OpenEnded_mscoco_val2014_questions.json", "mscoco_val2014_annotations.json")}

    def __init__(self, split, data_path="",
                 image_transforms=None, question_transforms=None, tokenize=None,
                 # answer_selection=most_common_from_dict,
                 answer_selection=all_answers_from_dict,
                 verbose=False, testing=False, max_samples=None):
        """
        split train, val, test
        balanced True, False
        image_transforms
        question_transforms
        """
        self.stemmer = PorterStemmer()
        
        start_time = time.time()
        self.split = split
        self.testing = testing
        self.answer_selection = answer_selection
        assert split in ["train", "test"]
        self.data_path = data_path
        self.image_transforms = image_transforms
        self.question_transforms = question_transforms
        self.tokenize = tokenize
        self.input_type = 'image'

        if verbose:
            path = self.data_path
            print(f"Start loading OKVQA Dataset from {path}", flush=True)

        # Questions
        path = os.path.expanduser(os.path.join(data_path, self.IMAGE_PATH[split][1]))
        with open(path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data["questions"])
        df["image_path"] = df["image_id"].apply(
            lambda x: f"{self.IMAGE_PATH[split][0]}/COCO_{self.IMAGE_PATH[split][0]}_{x:012d}.jpg")

        # Annotations
        if not testing:
            path = os.path.expanduser(os.path.join(data_path, self.IMAGE_PATH[split][2]))
            with open(path, 'r') as f:
                data = json.load(f)
            df_annotations = pd.DataFrame(data["annotations"])
            df = pd.merge(df, df_annotations, left_on='question_id', right_on='question_id', how='left')
            # Check if image_id are still correct, remove newly created columns with x and y ending and just use the name image_id
            assert df["image_id_x"].tolist() == df[
                "image_id_y"].tolist(), "image_id in df and df_annotations does not match."
            df["image_id"] = df["image_id_x"]
            del df["image_id_x"]
            del df["image_id_y"]
        df["mc_answer"] = df.answers.apply(most_common_from_dict_raw)
        self.df = df

        if max_samples is not None:
            self.df = self.df.sample(n=max_samples)

        self.n_samples = self.df.shape[0]
        if verbose:
            print(
                f"Loading OKVQA Dataset done in {time.time() - start_time:.1f} seconds. Loaded {self.n_samples} samples.")

        # For evaluation
        self.contractions = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
                             "couldnt": "couldn't", "couldn'tve": "couldn't've", "couldnt've": "couldn't've",
                             "didnt": "didn't","doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't",
                             "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't",
                             "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've", "hes": "he's", "howd": "how'd",
                             "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", "Im": "I'm",
                             "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've",
                             "itll": "it'll", "let's": "let's", "maam": "ma'am", "mightnt": "mightn't",
                             "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
                             "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've",
                             "oclock": "o'clock", "oughtnt": "oughtn't", "ow's'at": "'ow's'at", "'ows'at": "'ow's'at",
                             "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've",
                             "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't",
                             "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", "somebody'd": "somebodyd",
                             "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've",
                             "somebodyll": "somebody'll", "somebodys": "somebody's", "someoned": "someone'd",
                             "someoned've": "someone'd've", "someone'dve": "someone'd've", "someonell": "someone'll",
                             "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've",
                             "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's",
                             "thered": "there'd", "thered've": "there'd've", "there'dve": "there'd've",
                             "therere": "there're", "theres": "there's", "theyd": "they'd", "theyd've": "they'd've",
                             "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've",
                             "twas": "'twas", "wasnt": "wasn't", "wed've": "we'd've", "we'dve": "we'd've",
                             "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're",
                             "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd",
                             "wheres": "where's", "whereve": "where've", "whod": "who'd", "whod've": "who'd've",
                             "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've",
                             "whyll": "why'll", "whyre": "why're", "whys": "why's", "wont": "won't",
                             "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
                             "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll",
                             "yall'd've": "y'all'd've", "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've",
                             "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", "youll": "you'll",
                             "youre": "you're", "youve": "you've"}
        self.manualMap = {'none': '0',
                          'zero': '0',
                          'one': '1',
                          'two': '2',
                          'three': '3',
                          'four': '4',
                          'five': '5',
                          'six': '6',
                          'seven': '7',
                          'eight': '8',
                          'nine': '9',
                          'ten': '10'
                          }
        self.articles = ['a',
                         'an',
                         'the'
                         ]

        self.periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
        self.commaStrip = re.compile("(\d)(\,)(\d)")
        self.punct = [';', r"/", '[', ']', '"', '{', '}',
                      '(', ')', '=', '+', '\\', '_', '-',
                      '>', '<', '@', '`', ',', '?', '!']

    def get_img_path(self, index):
        image_id = self.df.iloc[index]["image_path"]
        image_path = os.path.expanduser(os.path.join(self.data_path, image_id))
        return image_path

    def get_index_from_sample_id(self, sample_id):
        return self.df[self.df["sample_id"] == sample_id].index[0]

    def __getitem__(self, index):
        # image input
        image_id = self.df.iloc[index]["image_id"]
        image_path = self.df.iloc[index]["image_path"]
        # question input
        question_id = self.df.iloc[index]["question_id"]
        question = self.df.iloc[index]["question"]
        # # answer and question type
        # answer_type = self.df.iloc[index]["answer_type"]
        # question_type = self.df.iloc[index]["question_type"]
        # # split
        # split = self.split
        # specify target if available (i.e. answer)
        selected_answers = None
        if not self.testing:
            answer_list = self.df.iloc[index]["answers"]  # Return whole list
            selected_answers = self.answer_selection(
                self.df.iloc[index]["answers"])  # Apply answer_selection() function to list of dict

        # Load and transform image
        image_path = os.path.expanduser(os.path.join(self.data_path, image_path))
        with open(image_path, "rb") as f:
            pil_img = Image.open(f).convert("RGB")
        if self.image_transforms:
            img = self.image_transforms(pil_img)
        else:
            img = pil_img

        # Load, transform and tokenize question
        if self.question_transforms:
            question = self.question_transforms(question)
        if self.tokenize:
            question = self.tokenize(question)

        # Return
        if self.testing:
            return {"sample_id": question_id, "img": img, "question": question, 'pil_img': pil_img, 'index': index,
                    'possible_answers': [], 'info_to_prompt': question, 'question_type': -1}

        else:
            return {"sample_id": question_id, 'answer': selected_answers, "img": img, "question": question,
                    'pil_img': pil_img, 'index': index, 'possible_answers': [], 'info_to_prompt': question,
                    "question_type": -1}

    def post_process(self, prediction, stem=True):
        """
        Code from https://github.com/GT-Vision-Lab/VQA/blob/master/PythonEvaluationTools/vqaEvaluation/vqaEval.py,
        as indicated here https://okvqa.allenai.org/leaderboard.html
        :return:
        """
        prediction = general_postprocessing(prediction)

        prediction = prediction.replace('\n', ' ')
        prediction = prediction.replace('\t', ' ')
        prediction = prediction.strip()
        prediction = self.processPunctuation(prediction)
        prediction = self.processDigitArticle(prediction)
        
        if stem:
            
            prediction = stem_sentence(prediction)

        return prediction

    def accuracy(self, prediction, ground_truth, *args):
        """
        Args:
            prediction (list): List of predicted answers.
            ground_truth (list): List of ground truth answers. Every element in the list is a list of strings with 10
            possible answers
        Returns:
            score (float): Score of the prediction.
        """
        assert len(prediction) == len(ground_truth)
        score = 0
        for p, g in zip(prediction, ground_truth):
            # There are 10 answers per question (10 annotators), most of them are the same
            item_score = self.get_item_score(p, g)
            score += item_score

        return score / len(prediction)

    def get_item_score(self, p, g):
        g = [self.post_process(g_, stem=False) for g_ in g]
        p = self.post_process(p)

        # The formulation is explained here: https://visualqa.org/evaluation.html
        accs = []
        for i, g_ in enumerate(g):
            other_answers = [item for j, item in enumerate(g) if i != j] # if it's 'other indices' instead of 'other answers'
            # for g_ in g:
            #     other_answers = [item for item in g if item != g_]
            matching_answers = [item for item in other_answers if item == p]
            acc = min(1., float(len(matching_answers)) / 3)
            accs.append(acc)
        item_score = sum(accs) / len(accs)
        return item_score

    def processPunctuation(self, inText):
        outText = inText
        for p in self.punct:
            if (p + ' ' in inText or ' ' + p in inText) or (re.search(self.commaStrip, inText) != None):
                outText = outText.replace(p, '')
            else:
                outText = outText.replace(p, ' ')
        outText = self.periodStrip.sub("", outText, re.UNICODE)
        return outText

    def processDigitArticle(self, inText):
        outText = []
        tempText = inText.lower().split()
        for word in tempText:
            word = self.manualMap.setdefault(word, word)
            if word not in self.articles:
                outText.append(word)
            else:
                pass
        for wordId, word in enumerate(outText):
            if word in self.contractions:
                outText[wordId] = self.contractions[word]
        outText = ' '.join(outText)
        return outText

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples
