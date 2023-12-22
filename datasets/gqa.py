import os
import time
import re

from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
import numpy as np

from datasets import general_postprocessing


class GQADataset(Dataset):
    BALANCED_TYPE = {
        True: "balanced",
        False: "all"
    }

    def __init__(self, split, balanced=True, data_path="",
                 image_transforms=None, question_transforms=None, tokenize=None,
                 verbose=False, testing=False, max_samples=None, first_n=None, return_pil=True):
        """
        Args:
            split (str): Data split. One of ["challenge", "submission", "test", "testdev", "train", "val"]
            balanced (bool): You balanced version or full version.
            image_transforms:
            question_transforms:
            tokenize (fct):
            verbose (bool): Print some infos. Default=True
            testing (bool): Set to true for data splits without targets. Default=False.
            first_n (int): Only use the first n samples. Default=None. Only valid if loading from hdf.
        """
        start_time = time.time()
        self.split = split
        self.testing = testing
        assert split in ["challenge", "submission", "test", "testdev", "train", "val"]
        self.balanced = balanced
        self.balanced_type = self.BALANCED_TYPE[balanced]
        self.data_path = data_path
        self.image_transforms = image_transforms
        self.question_transforms = question_transforms
        self.tokenize = tokenize
        self.input_type = 'image'
        self.return_pil = return_pil

        if not balanced and split == "train":
            raise NotImplementedError
        else:
            # check path to cached df exists
            if self.split == 'train' and self.balanced_type == 'balanced' and os.path.exists(
                    os.path.join(data_path, f"questions/{self.split}_{self.balanced_type}_questions.h5")):
                if verbose:
                    print(f"Loading GQA Dataset from {data_path}", flush=True)
                self.df = pd.read_hdf(
                    os.path.join(data_path, f"questions/{self.split}_{self.balanced_type}_questions.h5"), "table", stop=first_n)
            else:
                self.file_name = f"questions/{self.split}_{self.balanced_type}_questions.json"
                path = os.path.expanduser(os.path.join(data_path, self.file_name))
                if verbose:
                    print(f"Loading GQA Dataset from {path}", flush=True)
                self.df = pd.read_json(path, orient="index")

        if max_samples is not None:
            self.df = self.df.sample(n=max_samples)

        self.n_samples = self.df.shape[0]
        if verbose:
            print(
                f"Loading GQA Dataset done in {time.time() - start_time:.1f} seconds. Loaded {self.n_samples} samples.")
            
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
        
        self.max_words = 50
    
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

    def get_img_path(self, index):
        if "imageId" in self.df.columns:
            image_id = self.df.iloc[index]["imageId"]
        else:
            image_id = self.df.iloc[index]["image_id"]
        return os.path.expanduser(os.path.join(self.data_path, "../images", f"{image_id}.jpg"))

    def get_index_from_sample_id(self, sample_id):
        return np.where(self.df.index == sample_id)[0][0].item()

    def __getitem__(self, index):
        # image input
        sample_id = self.df.iloc[index].name
        if "imageId" in self.df.columns:
            image_id = self.df.iloc[index]["imageId"]
        else:
            image_id = self.df.iloc[index]["image_id"]
        question = self.df.iloc[index]["question"]

        question_type = -1
        answer = -1
        if not self.testing:
            answer = self.df.iloc[index]["answer"]
            question_type = self.df.iloc[index]["groups"]["global"]
            if question_type is None:
                question_type = -1  # can't have None for DataLoader

        # Load and transform image
        image_path = os.path.expanduser(os.path.join(self.data_path, "images", f"{image_id}.jpg"))
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
            if (sample_id is None) or (img is None) or (question is None):
                raise Exception(f"Error in GQA Dataset: sample_id={sample_id}, img={img}, question={question}")
            out_dict = {"sample_id": sample_id, "img": img, "question": question, 'index': index}
            if self.return_pil:
                out_dict["pil_img"] = pil_img
            return out_dict
        else:
            out_dict = {"sample_id": sample_id, "answer": answer, "img": img, "question": question, 'pil_img': pil_img,
                        "question_type": question_type, 'index': index, 'possible_answers': [],
                        'info_to_prompt': question}
            return out_dict

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
        return prediction

    def accuracy(self, prediction, ground_truth, *args):
        """
        Args:
            prediction (list): List of predicted answers.
            ground_truth (list): List of ground truth answers.
        Returns:
            score (float): Score of the prediction.
        """
        if len(prediction) == 0:  # if no prediction, return 0
            return 0
        assert len(prediction) == len(ground_truth)
        score = 0
        for p, g in zip(prediction, ground_truth):
            if self.post_process(p) == g:
                score += 1
        return score / len(prediction)

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples
