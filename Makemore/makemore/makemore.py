# simple and weak language model that takes on character and predicts the next character in a first name. simple & a good place to start.

from pprint import pprint

words = open('names.txt', 'r').read().splitlines()
# words[:10]
for w in words[:1]:
	for ch1, ch2 in zip(w, w[1:]):
		print(ch1, ch2)

	