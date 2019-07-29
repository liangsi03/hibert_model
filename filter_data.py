from nltk.tokenize import RegexpTokenizer
import pandas as pd
from nltk.tokenize import word_tokenize
import re
import json


tokenizer = RegexpTokenizer(r'\w+')
reader = pd.read_json("RC_2015-01.jsonl", lines=True, chunksize=1, orient="value")
URLS = [".com",".org",".net",".us",".co",".int",".mil",".edu",".gov",".biz",".info",".jobs",".mobi",".name",".ly",".tel",".kitchen",".email",".tech",".estate",".xyz",".codes",".bargains",".bid",".expert",".ca",".cn",".fr",".ch",".au",".in",".de",".jp",".nl",".uk",".mx",".no",".ru",".br",".se",".es",".jpg",".png",".gif"]
EMOTICONS = r"""
    (?:
      [<>]?
      [:;=8]                     # eyes
      [\-o\*\']?                 # optional nose
      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
      |
      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
      [\-o\*\']?                 # optional nose
      [:;=8]                     # eyes
      [<>]?
      |
      <3                         # heart
    )"""

EMOTICON_RE = re.compile(EMOTICONS, re.VERBOSE | re.I | re.UNICODE)

def getClean(text):
    t = text.lower().replace("\n","<n>").replace("/u/","USER ").replace("/r/","SUBREDDIT ")
    text = []
    for w in t.split(" "):
        if not EMOTICON_RE.search(w):
            check = False
            for url in URLS:
                if url in w:
                    check = True
                    break
            if not check:
                text.append(w)
    t = word_tokenize(" ".join(text))
    ignore = [1 for w in t]
    stack = []
    for i in range(0,len(t)):
        if t[i] == "(" or t[i] == "[" or t[i] == "{":
            stack.append((t[i],i))
        elif t[i] == ")" or t[i] == "]" or t[i] == "}":
            for j in range(len(stack)-1,-1,-1):
                if (t[i] == ")" and stack[j][0] == "(") or (t[i] == "]" and stack[j][0] == "[") or (t[i] == "}" and stack[j][0] == "{"):
                    for k in range(stack[j][1],i+1):
                        ignore[k] = 0
                    stack.pop(j)       
    good = []
    for i in range(0,len(t)):
        if ignore[i] != 0:
            if t[i] == ":" and (i+1) < len(t) and t[i+1] == "<n>":
                ignore[i+1] = 0
            if (t[i] == "USER" or t[i] == "SUBREDDIT") and (i+1) < len(t):
                good.append(t[i] + " ")
                ignore[i+1] = 0
            elif (i+1) < len(t) and t[i+1] != ".":
                good.append(t[i] + " ")
            else:
                good.append(t[i])
                
    return "".join(good).replace("< n >","\n")



def filterData():
    texts = []
    index = 0
    for chunk in reader:
        index += 1
        body = chunk['body'].values[0]
        try:
            body = getClean(body)
            tokens = tokenizer.tokenize(body)
            count = len(tokens)
            if count > 200 and count < 300:
                texts.append(body)
            if index != 0 and index % 10000 == 0:
                print("Saving file at index " + str(index))
                with open("rs/" + str(int(index/10000)) + ".json" , 'w') as f:
                    json.dump(texts, f)
                    texts = []
        except:
            print(index)
    with open("rs/final.json", 'w') as f:
        json.dump(texts, f)

filterData()