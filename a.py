import nltk
from nltk import corpus
new = "Big cat ate the little mouse who ate whole cheese.The cheese was made to make cake." \
      "so,decided to make new cheese."
word = nltk.word_tokenize(new)
print(word)
new_tag = nltk.pos_tag(word)
print(new_tag)
grammar = 'NP:{<DT>?<JJ>*<NN>}'
ChunkParser = nltk.RegexpChunkParser(grammar)
chunked = ChunkParser.parse(new_tag)
print(chunked)
chunked.draw()
