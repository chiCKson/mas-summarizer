from osbrain import run_agent
from osbrain import run_nameserver
import re
import nltk
import heapq

def preprocess(agent,text):
    clean_text = str(text).lower()
    clean_text = re.sub (r"\d", " ", clean_text)
    clean_text = re.sub (r"\W", " ", clean_text)
    clean_text = re.sub (r"\s+", " ", clean_text)
    return clean_text
def textTokenize(agent,text):
        sentences = nltk.sent_tokenize (str(text))
        return sentences


if __name__ == '__main__':
    
    ns = run_nameserver()
    main = run_agent('main')
    preprocessor = run_agent('preprocess')
    tokenizer=run_agent('tokenizer')
    pre_addr = preprocessor.bind('REP', handler=preprocess)
    main.connect(pre_addr, alias='preprocess')
    
    text = """Here's a term you're going to hear much more often: plug-in vehicle, and the acronym PEV. 
    It's what you and many other people will drive to work in, ten years and more from now. At that time, 
    before you drive off in the morning you will first unplug your car - your plug-in vehicle. Its big onboard
    batteries will have been fully charged overnight, with enough power for you to drive 50-100 kilometres through city traffic. 
    When you arrive at work you'll plug in your car once again, this time into a socket that allows power to flow from your car's batteries
    to the electricity grid. One of the things you did when you bought your car was to sign a contract with your favourite electricity 
    supplier, allowing them to draw a limited amount of power from your car's batteries should they need to, perhaps because of a blackout, 
    or very high wholesale spot power prices. The price you get for the power the distributor buys form your car would not only be most attractive to you, but it would also be a good deal for them too, their alternative being very expensive power form peaking stations. If driving home or for some other reason your batteries looked like running flat, a relatively small,
    but quiet and efficient engine running on petrol, diesel or compressed natural gas, even bio-fuel, 
    would automatically cut in, driving a generator that supplied the batteries so you could complete your journey.
    Concerns over 'peak oil', increasing greenhouse gas emissions, and the likelihood that by the middle of this century
    there could be five times as many motor vehicles registered worldwide as there are now, mean that the world's almost total 
    dependence on petroleum-based fuels for transport is, in every sense of the word, unsustainable. """
    #send data to prerocessing agent
    main.send('preprocess', text)
    clean=main.recv('preprocess')

    tok_addr=tokenizer.bind('REP', handler=textTokenize)
    main.connect(tok_addr, alias='tokenizor')
    #send data to tokenizor agent
    main.send('tokenizor', clean)
    print(main.recv('tokenizor'))
    ns.shutdown()