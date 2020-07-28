from nltk import sent_tokenize, word_tokenize, pos_tag, ne_chunk
from nltk.corpus import brown
import nltk
import numpy
from io import StringIO
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from nltk.tree import Tree
import os
import sys
from collections import Counter
import math

#reload(sys)
#sys.setdefaultencoding("utf-8")
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
delimiters = [".",",",";",":","?","/","!","'s","'ll","'d","'nt","(",")","{","}",
              "[","]"]
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
def f7(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]  
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
# pdf to text conversion function
def convert(fname, pages=None):
    if not pages:
        pagenums = set()
    else:
        pagenums = set(pages)

    output = StringIO()
    manager = PDFResourceManager()
    converter = TextConverter(manager, output, laparams=LAParams())
    interpreter = PDFPageInterpreter(manager, converter)

    infile = open(fname, 'rb')
    for page in PDFPage.get_pages(infile, pagenums):
        interpreter.process_page(page)
    infile.close()
    converter.close()
    text = output.getvalue()
    output.close
    return text 
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
# Preparing stopwords list
stopwords = []

temp_1 = 0
temp_2 = 0
temp_3 = 0

with open('stopwords_collection.txt', 'r+') as f1:
    temp_1 = f1.readlines()
f1.close()

temp_2 = ''.join(temp_1)
temp_3 = temp_2.replace('\n', ", ").strip()

stopwords = temp_3.split(", ")
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
class SparseVector_class:

    def __init__(self, wv1, wv2):
        self.wv1 = wv1
        self.wv2 = wv2
        
    def union_intersection(self, wv1, wv2):
        len1 = len(wv1)
        len2 = len(wv2)

        union_size = 0
        intersect_size = 0
        i = 0
        j = 0

        while(i<len1 and j<len2):
            idx1 = wv1[i]
            idx2 = wv2[j]
            
            if (idx1 < idx2):
                union_size = union_size + 1
                i = i + 1
            
            elif (idx2 < idx1):
                union_size = union_size + 1
                j = j + 1
                
            elif (idx1 == idx2):
                union_size = union_size + 1
                intersect_size = intersect_size + 1
                i = i + 1
                j = j + 1

        if (i<len1 and j>=len2):
            while(i < len1):
                union_size = union_size + 1
                i= i + 1
                	
        elif (i>=len1 and j<len2):
            while (j < len2):
                union_size = union_size + 1
                j = j + 1

        return (union_size, intersect_size)
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
class Unigram:
    def __init__(self,str):
        self.str = str
    def compute_unigrams(self,str):
        ugs = [word for word in str if (word not in stopwords and word not in delimiters)]
        #print ("Unigram List: ", ugs)
        return ugs
#------------------------------------------------------------------------------#        

#------------------------------------------------------------------------------#
class Bigram:
    def __init__(self,str):
        self.str = str
    def compute_bigrams(self,str):
        final_list=[]
        bigramtext = list(nltk.bigrams(str))
        for item in bigramtext:
            if item[0] not in delimiters and item[len(item)-1] not in delimiters:
                if not item[0].isdigit() and not item[1].isdigit():
                    if item[0] not in stopwords and item[len(item)-1] not in stopwords:    
                        if len(item[0])>1  and len(item[len(item)-1])>1:
                            final_list.append(item)    
        #print ("Bigram List: ", final_list)
        return final_list
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#        
class Trigram:
    def __init__(self,str):
        self.str = str
    def compute_trigrams(self,str):
        final_list=[]
        trigramtext = list(nltk.trigrams(str))
        for item in trigramtext:
            if item[0] not in delimiters and item[1] not in delimiters and item[len(item)-1] not in delimiters:
                if not item[0].isdigit() and not item[1].isdigit() and not item[len(item)-1].isdigit():
                    if item[0] not in stopwords and item[len(item)-1] not in stopwords:    
                        if len(item[0])>1  and len(item[len(item)-1])>1:
                            final_list.append(item)  
        #print ("Trigram List: ", final_list)                      
        return final_list
#------------------------------------------------------------------------------#
        
#------------------------------------------------------------------------------#
#  Object Creation of Unigram, Bigram and Trigram class
unigram_obj = Unigram("")
bigram_obj = Bigram("")
trigram_obj = Trigram("")
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
# Preparing domain_specific_phrases list

input_domain_file = "lifescience_domain.txt"
out_file_1 = "domain_prepared_1.txt"
if(os.path.isfile(out_file_1)):
    os.remove(out_file_1)

# removal of numbers
out_file_1_fd = open(out_file_1, "w")
for line in open(input_domain_file, "r"):
    line_1 = ''.join([i for i in line if not i.isdigit()])
    out_file_1_fd.write(line_1)
out_file_1_fd.close()


# duplicate line removal
out_file_2 = "domain_prepared_2.txt"
if(os.path.isfile(out_file_2)):
    os.remove(out_file_2)

out_file_2_fd = open(out_file_2, "w")

lines_seen = set() # holds lines already seen
for line in open(out_file_1, "r"):
    if line not in lines_seen: # not a duplicate
        out_file_2_fd.write(line)
        lines_seen.add(line)
out_file_2_fd.close()
 
domain_specific_phrases = []

temp_1 = 0
temp_2 = 0
temp_3 = 0

with open(out_file_2, 'r') as f1:
    temp_1 = f1.readlines()
f1.close()

temp_2 = ''.join(temp_1)
temp_3 = temp_2.replace('\n', ", ").strip()
#print temp_3

domain_specific_phrases = temp_3.split(", ")
#print domain_specific_phrases     
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
# Taking the input file what is to be summarised    
input_file_tobe_summarised = sys.argv[1]
text_out = convert(input_file_tobe_summarised)

extracted = "pdf_output.txt"
if(os.path.isfile(extracted)):
    os.remove(extracted)

with open (extracted, "w", encoding="utf-8") as fo_pdf:
    fo_pdf.write(text_out)

input_file = "pdf_output.txt" 
fo = open(input_file, "r", encoding="utf-8")

input_file_data = fo.read()
fo.close()

#print (len(input_file_data))

input_text = input_file_data.lower()


#  Sentence representation
input_text_sents = sent_tokenize(input_text)
#print "input_text_sents = %s" %input_text_sents

original_sentences = sent_tokenize(input_file_data)
print ("Length - original_sentences = %s" %len(original_sentences))

#  Word representation
word_tokens = [word_tokenize(sent) for sent in input_text_sents]

#  Accessing the computed List of Unigram, Bigram and Trigram
sentenceugs = []
sentencebgs = []
sentencetgs = []


for token in word_tokens:
    a = unigram_obj.compute_unigrams(token)
    sentenceugs.append(a)
    b = bigram_obj.compute_bigrams(token)
    sentencebgs.append(b)
    c = trigram_obj.compute_trigrams(token)
    sentencetgs.append(c)


# Preparing ngram_list
ngram_list = []

for i in range(0, len(input_text_sents)):
    cnt = 0    
    for unigram in sentenceugs[i]:
        if str(unigram) in domain_specific_phrases:
            cnt = cnt + 1
    
    for bigram in sentencebgs[i]:    
        new_bigram=bigram[0]+" "+bigram[1]
        if str(new_bigram) in domain_specific_phrases:
            cnt = cnt + 1

    for trigram in sentencetgs[i]:    
        new_trigram=trigram[0]+" "+trigram[1]+" "+trigram[2]
        if str(new_trigram) in domain_specific_phrases:
            cnt = cnt + 1

    ngram_list.append(cnt)
#print (ngram_list)
 
len_list = len(domain_specific_phrases)

ngram_score_list = []
# Preparing ngram_score_list 
for i in ngram_list:
    ngram_score = round((float(i) / float(len_list)), 3)
    ngram_score_list.append(ngram_score)
#print ("Length - ngram_score_list = %s" %len(ngram_score_list))

sorted_ngram_score_list = sorted(range(len(ngram_score_list)), 
    key=lambda k: ngram_score_list[k], reverse=True)
#print ("Length - sorted_ngram_score_list = %s" %len(sorted_ngram_score_list)) 

sent_list = []    
for i in sorted_ngram_score_list:
    sent_list.append(original_sentences[i])
print ("Length - sent_list = %s" %len(sent_list))

# Preparing the required_input_sentences list
required_input_sentences = []
for sentence in sent_list:
    if sentence not in required_input_sentences:
        required_input_sentences.append(sentence)
print ("Length of required_input_sentences = %s" %len(required_input_sentences)) 

#  Word representation from required_input_sentences
required_input_sentences_tokens =  [word_tokenize(sent.lower()) for sent in required_input_sentences]

input_sentences_ugs = []
input_sentences_bgs = []
input_sentences_tgs = []

for token in required_input_sentences_tokens:
    unigram_temp = unigram_obj.compute_unigrams(token)
    bigram_temp = bigram_obj.compute_bigrams(token)
    trigram_temp = trigram_obj.compute_trigrams(token)

    input_sentences_ugs.append(unigram_temp)
    input_sentences_bgs.append(bigram_temp)
    input_sentences_tgs.append(trigram_temp)
    
#print ("input_sentences_ugs: \n", input_sentences_ugs)
#------------------------------------------------------------------------------# 

#------------------------------------------------------------------------------#
def build_dictionary():

    # for unigram
    ui = 0
    du = {}

    for sentence in input_sentences_ugs:
        for unigram in sentence:
            if unigram not in du:
                du.update({unigram : ui})
                ui += 1

    for lst_u in input_sentences_ugs:
        l = len(lst_u)
        for i in range(0, l):
            lst_u[i] = du[lst_u[i]]


    # for bigram
    bi = 0
    db = {}

    for sentence in input_sentences_bgs:
        for bigram in sentence:
            if bigram not in db:
                db.update({bigram : bi})
                bi += 1

    for lst_b in input_sentences_bgs:
        l = len(lst_b)
        for i in range(0, l):
            lst_b[i] = db[lst_b[i]]
            
            
    # for trigram
    ti = 0
    dt = {}

    for sentence in input_sentences_tgs:
        for trigram in sentence:
            if trigram not in dt:
                dt.update({trigram : ti})
                ti += 1

    for lst_t in input_sentences_tgs:
        l = len(lst_t)
        for i in range(0, l):
            lst_t[i] = dt[lst_t[i]]        

    #print ("input_sentences_ugs", input_sentences_ugs)
    #print ("input_sentences_bgs", input_sentences_bgs)
    #print ("input_sentences_tgs", input_sentences_tgs)
    
    return input_sentences_ugs, input_sentences_bgs, input_sentences_tgs
#------------------------------------------------------------------------------#

#-------------------Unigram, Bigram, Trigram proximity computation-------------#
def compute_ubt_proximity():

    # SparseVector_class object creation
    sparseVector = SparseVector_class ([], [])

    #--------------- Unigram proximity --------------------#
    ugs_len_list = len(input_sentences_ugs)
    i_rng_u = range(ugs_len_list)

    ugs_intersect_list = []
    ugs_proximity_list = []

    for i in i_rng_u:    
        for j in i_rng_u:
            u_wv1 = input_sentences_ugs[i]
            u_wv2 = input_sentences_ugs[j]
            set_ugs = sparseVector.union_intersection(u_wv1, u_wv2)
            ugs_intersect_list.append(set_ugs)
    #print ugs_intersect_list

    for i in ugs_intersect_list:
        try:
            unigram_proximity = round((float(i[1]) / float(i[0])), 3)
        except ZeroDivisionError:                                      
            #print "divide by zero"
            unigram_proximity = 0    
        ugs_proximity_list.append(unigram_proximity)
            
    ugs_proximity_list = [0.33 * x for x in ugs_proximity_list]
    #print ("ugs_proximity_list: \n", ugs_proximity_list)
    #print ("Length - ugs_proximity_list = %s" %len(ugs_proximity_list))
    #------------------------------------------------------#
    
    #-------------- Bigram proximity ----------------------#
    bgs_len_list = len(input_sentences_bgs)
    i_rng_b = range(bgs_len_list)
    
    bgs_intersect_list = []
    bgs_proximity_list = []

    for i in i_rng_b:    
        for j in i_rng_b:
            b_wv1 = input_sentences_bgs[i]
            b_wv2 = input_sentences_bgs[j]
            set_bgs = sparseVector.union_intersection(b_wv1, b_wv2)
            bgs_intersect_list.append(set_bgs)
    #print bgs_intersect_list

    for i in bgs_intersect_list:
        try:
            bigram_proximity = round((float(i[1]) / float(i[0])), 3)
        except ZeroDivisionError:
            #print "divide by zero"
            bigram_proximity = 0    
        bgs_proximity_list.append(bigram_proximity)
            
    bgs_proximity_list = [0.66 * x for x in bgs_proximity_list]
    #print ("bgs_proximity_list: \n", bgs_proximity_list)
    #print ("Length - bgs_proximity_list = %s" %len(bgs_proximity_list))
    #------------------------------------------------------#
    
    #----------------- Trigram proximity ------------------#
    tgs_len_list = len(input_sentences_tgs)
    i_rng_t = range(tgs_len_list)

    tgs_intersect_list = []
    tgs_proximity_list = []

    for i in i_rng_t:    
        for j in i_rng_t:
            t_wv1 = input_sentences_tgs[i]
            t_wv2 = input_sentences_tgs[j]
            set_tgs = sparseVector.union_intersection(t_wv1, t_wv2)
            tgs_intersect_list.append(set_tgs)
    

    for i in tgs_intersect_list:
        try:
            trigram_proximity = round((float(i[1]) / float(i[0])), 3)
        except ZeroDivisionError:
            #print "divide by zero"
            trigram_proximity = 0                
        tgs_proximity_list.append(trigram_proximity)
            
    tgs_proximity_list = [0.99 * x for x in tgs_proximity_list]
    #print ("tgs_proximity_list: \n", tgs_proximity_list)
    #print ("Length - tgs_proximity_list = %s" %len(tgs_proximity_list))

    ubt_proximity_list = []
    ubt_proximity_list = tgs_proximity_list + bgs_proximity_list + ugs_proximity_list 
    #print ("Combined Proximity: \n", ubt_proximity_list)
    #print ("Length - ubt_proximity_list = %s" %len(ubt_proximity_list))
    
    return ubt_proximity_list 
#------------------------------------------------------------------------------#

#-----------Final proximity calculation----------------------------------------#
def compute_final_proximity():

    proximity_result = compute_ubt_proximity()
    
    i = 0
    new_list = []
    while i < len(proximity_result):
        new_list.append(proximity_result[i:i+len(required_input_sentences)])
        i += len(required_input_sentences)
    
    #print ("new_list length = %s" %len(new_list))
    proximity_matrix = numpy.array(new_list)
    #print ("proximity_matrix: \n", proximity_matrix)

    i,j = numpy.indices(proximity_matrix.shape)
    proximity_matrix[i==j] = 0

    len_prox_mat = len(proximity_matrix)
    #print ("Proximity Matrix Length = %s" %len_prox_mat)
    
    rng_prox_mat = range(len_prox_mat)
    avg_mat = [sum(proximity_matrix[i]) / len_prox_mat for i in rng_prox_mat]


    #---Sorting the matrix element in descending order and getting the index
    sorted_avg_mat = sorted(range(len(avg_mat)), key=lambda x:avg_mat[x], 
        reverse=True)

    #print ("sorted_avg_mat: \n", sorted_avg_mat) 
    return sorted_avg_mat 
#------------------------------------------------------------------------------#  

#------------------------------------------------------------------------------#    
def rank_sentence():    
    required_index = compute_final_proximity()
    #print ("required_index = %s" %required_index)

    final_sentences = []
    sents_dpl = []

    index_consider = len(required_input_sentences)
    #print ("index_consider = %s" %index_consider)  
    
    i = 0
    j = 0
    temp_list = []
    for i in required_index:
        j = i % index_consider
        temp_list.append(required_input_sentences[j])
    #print ("temp_list = %s" %temp_list)
    
    temp_list_2 = []
    # removal of duplicate from_temp_list
    temp_list_2 = f7(temp_list)
    #print (temp_list_2)


    pruning_length = int(math.ceil(len(temp_list_2) * 0.15)) # 15% summary
    #print ("Pruning Length = %s" %pruning_length)

    final_sent_list = temp_list[:pruning_length]
    #print ("final_sent_list: \n", final_sent_list)
    #print ("final_sent_list Length = %s" %len(final_sent_list))
    

    i = 0
    sent_index_list = []
    for i in final_sent_list:
        sent_index_list.append(original_sentences.index(i))

    sorted_list1 = sorted(sent_index_list)
    #print "sorted_list1 = %s" %sorted_list1

    i = 0
    for i in sorted_list1:
        sents_dpl.append(original_sentences[i])

    for sentence in sents_dpl:
        if sentence not in final_sentences:
            final_sentences.append(sentence)
    #print ("Summarized Result: \n", final_sentences)
    print ("Length - final_sentences = %s" %len(final_sentences))

    with open (summary_result, "w", encoding="utf-8") as fs:
        for sentence in final_sentences:
            fs.write(sentence)
#------------------------------------------------------------------------------#            

#------------------------------------------------------------------------------#    
summary_result = "summary_output.txt"
if(os.path.isfile(summary_result)):
    os.remove(summary_result)

build_dictionary()
rank_sentence()

# Removal of the intermediate files
if(os.path.isfile(out_file_1)):
    os.remove(out_file_1)

if(os.path.isfile(out_file_2)):
    os.remove(out_file_2)

if(os.path.isfile(input_file)):
    os.remove(input_file)

#------------------------------------------------------------------------------#
