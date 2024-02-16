# simple and weak language model that takes on character and predicts the next character in a first name. simple & a good place to start.

from pprint import pprint

words = open('names.txt', 'r').read().splitlines()


# this is how we get the counts of all the words of the individial bigrams
b = {}
for w in words:
	chs = ['<S>'] + list(w) + ['<E>']
	for ch1, ch2 in zip(chs, chs[1:]):
		# count how often the combinations occur
		bigram = (ch1, ch2)
		b[bigram] = b.get(bigram, 0) + 1
		print(ch1, ch2)

# sorting data by the count of the elements
#  kv[1] for smallest
# -kv[1] for largest
sorted(b.items(), key = lambda kv: -kv[1]) 
pprint(b)
