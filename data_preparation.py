import string
import re
from pickle import dump, load
from unicodedata import normalize
from numpy import array
import pandas as pd

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, mode='rt', encoding='utf-8')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

def pairs(inputs, outputs):
  pairs = [[inputs[i],outputs[i]]for i in range(len(inputs))]
  return pairs
  

def clean_pairs(lines):

  re_print = re.compile('[^%s]' %re.escape(string.printable)) #0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
  table = str.maketrans("","", string.punctuation) #!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
  clean_list = []
  for pair in lines: #(english, german) pairs in each row
    clean_pairs = []
    for phrase in pair: # english and then german phrases separately
      #normalize latin characters
      phrase = normalize("NFD", phrase).encode("ascii","ignore")
      phrase = phrase.decode('UTF-8')
      #tokenize on white space
      phrase = phrase.split()
      #lowecase the words
      phrase = [word.lower() for word in phrase]
      #remove punctuation
      phrase = [word.translate(table) for word in phrase]
      #remove non-printable
      phrase = [re_print.sub("", word) for word in phrase]
      #remove non neumeric alpahbet
      phrase = [word for word in phrase if word.isalpha()]
      #store as string
      clean_pairs.append(' '.join(phrase))
    clean_list.append(clean_pairs)
  return array(clean_list)

# save a list of clean sentences to file
def save_clean_data(sentences, filename):
	dump(sentences, open(filename, 'wb'))
	print('Saved: %s' % filename)

def load_clean_data(filename):
  return load(open(filename,'rb'))



# load dataset

my_file = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/deu.txt",sep ="\t")
inputs = my_file.iloc[:, 0]
outputs = my_file.iloc[:, 1]
# split into english-german pairs
pairs = pairs(inputs,outputs)

clean = clean_pairs(pairs)
save_clean_data(clean_pairs(pairs),'english-german.pkl')



