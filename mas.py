from osbrain import run_agent
from osbrain import run_nameserver
import re
import nltk
import operator

def preprocess(agent,text):
    clean_text = str(text).lower()
    clean_text = re.sub (r"\d", " ", clean_text)
    clean_text = re.sub (r"\W", " ", clean_text)
    clean_text = re.sub (r"\s+", " ", clean_text)
    return clean_text
def sentencesTokenize(agent,text):
        sentences = nltk.sent_tokenize (str(text))
        return sentences
def stopwordRemover(agent,clean_text):
    stop_words = nltk.corpus.stopwords.words('english')
    word_count_dict = {}

    for word in nltk.word_tokenize(clean_text):
        if word not in stop_words:
            if word not in word_count_dict.keys():
                word_count_dict[word] = 1
            else:
                word_count_dict[word] += 1
    return word_count_dict
def normalizing(agent,word_count_dict):
    # Find the total number of terms (not necessarily unique) = sum of values in the word_count_dict
    total_terms = sum(word_count_dict[term] for term in word_count_dict)
    
    for key in word_count_dict.keys():
        word_count_dict[key] = word_count_dict[key]/total_terms
    return word_count_dict

if __name__ == '__main__':
    
    ns = run_nameserver()

    #create agents
    main = run_agent('main')
    preprocessor = run_agent('preprocess')
    tokenizer=run_agent('tokenizer')
    swremover=run_agent('swremover')
    normalizer=run_agent('normalizer')


    pre_addr = preprocessor.bind('REP', handler=preprocess)
    main.connect(pre_addr, alias='preprocess')
    
    text = """The coronavirus disease pandemic has claimed 400,000 lives globally since it surfaced in the Chinese city of Wuhan late last year and went on to ravage countries across the world that are now attempting to revive economies battered by the lockdowns imposed to curb the spread of the Sars-Cov-2 virus that causes the disease.

It took nearly four months for the death toll from the respiratory illness to reach the grim milestone of 100,000 deaths. As the virus spread from China – where it first originated in December 2019 – to find a strong foothold in Europe, the number of fatalities doubled to 200,000 in another 15 days.

The subsequent 100,000 deaths were added in 20 and 23 days, respectively, offering a glimmer of hope that many of the hot spots such as Spain, Italy, UK and France may have seen the worst.

Although European countries have begun to reopen businesses and industries, Latin America, particularly Brazil, has emerged as the latest epicentre of the viral disease, according to the World Health Organization.

In the United States– the hardest-hit country with 1.97 million cases and 111,658 deaths -- the numbers have continued to mount even as the rate of the infection appears to be slowing down.

On Friday, US President Donald Trump said the economy was bouncing back and that the country was “largely through” this “horrible pandemic”.

“I think we’re doing really well,” he added.

Countries such as Mexico, Russia and India too are clocking thousands of daily new cases and hundreds more fatalities, driving part of the third wave of the pandemic after the first in China and the second in Europe and the US.

Till Saturday, 400,012 fatalities had been recorded from 6,916,826cases world over. It means the case fatality rate – defined as the proportion of deaths to total infections -- from Covid-19 stood at 5.8% on Saturday, according to data by worldometers.info. """
    #send data to prerocessing agent
    main.send('preprocess', text)
    clean=main.recv('preprocess')

    

    #send data to tokenizor agent
    tok_addr=tokenizer.bind('REP', handler=sentencesTokenize)
    main.connect(tok_addr, alias='tokenizor')
    main.send('tokenizor', text)
    sentences=main.recv('tokenizor')

    # Remove stop words and create a dictionary of word-count
    swr_addr=tokenizer.bind('REP', handler=stopwordRemover)
    main.connect(swr_addr, alias='swremover')
    main.send('swremover', clean)
    word_count_dict=main.recv('swremover')

    # Normalize the word-frequency dictionary (weighted word count matrix/dictionary)
    nml_addr=tokenizer.bind('REP', handler=normalizing)
    main.connect(nml_addr, alias='normalizer')
    main.send('normalizer', word_count_dict)
    word_count_dict_normalized=main.recv('normalizer')
    
    # Create sentece scores
    sentence_score_dict = {}
    for sentence in sentences:
        for word in nltk.word_tokenize(sentence.lower()):
            if word in word_count_dict.keys():
                # Now here's a problem, very long sentences will always have high scores, so here we can ignore very long sentences
                if len(sentence.split(' ')) < 20: # ignore sentences having words more than 20
                    if sentence not in sentence_score_dict.keys():
                        sentence_score_dict[sentence] = word_count_dict[word]
                    else:
                        sentence_score_dict[sentence] += word_count_dict[word]
    

    
    # Print the summary
    print(max(sentence_score_dict, key=sentence_score_dict.get))
    ns.shutdown()