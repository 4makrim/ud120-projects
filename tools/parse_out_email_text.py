#!/usr/bin/python

from nltk.stem.snowball import SnowballStemmer
import string

def parseOutText(f):
    """ given an opened email file f, parse out all text below the
        metadata block at the top
        (in Part 2, you will also add stemming capabilities)
        and return a string that contains all the words
        in the email (space-separated) 
        
        example use case:
        f = open("email_file_name.txt", "r")
        text = parseOutText(f)
        
        """


    f.seek(0)  ### go back to beginning of file (annoying)
    all_text = f.read()
    #print(all_text)
    ### split off metadata
    #content = all_text.decode('utf-8').split('X-FileName:')
    content = all_text.split('X-FileName:')
    #print(string.punctuation)
    
    words = ""
    if len(content) > 1:
        ### remove punctuation, changed code for Python 3 here.
        text_string = content[1].translate(str.maketrans("", "", string.punctuation))

        ### project part 2: comment out the line below
        temp_words = text_string.split()
        stemmer = SnowballStemmer('english', ignore_stopwords=False)
        #print(stemmer.stem('running'))
        word_list = [stemmer.stem(w) for w in temp_words]
        #print(words)
        ### split the text string into individual words, stem each word,
        ### and append the stemmed word to words (make sure there's a single
        ### space between each stemmed word)
        words = ' '.join(word_list)

    return words

    

def main():
    ff = open("../text_learning/test_email.txt", "rb")
    text = parseOutText(ff)
    print(text)



if __name__ == '__main__':
    main()

