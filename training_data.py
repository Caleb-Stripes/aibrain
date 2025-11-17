from pprint import pp
import numpy as np

DATA = (
    ("cookie", "cream "), 
    ("peanut", "jelly "),
    ("salt  ", "pepper"),
    ("mac   ", "cheese"),
    ("bread ", "butter"),
    ("fish  ", "chips "),
    ("ham   ", "eggs  "),
    # ("spaghetti", "meatballs"),
    # ("pancakes", "syrup"),
    ("bacon ", "eggs  "),
    ("milk  ", "cereal"),
    # ("bacon ", "tomato" ),
    ("fries ", "french")
    )

# squish all the words together
def squish_to_set(pairs):
    squish = ''.join(word for pair in pairs for word in pair)
    return sorted(set(squish))

# squish = squish_to_set(DATA)
# squish = 'abcdefghijklmnopqrstuvwxyz '
import string
squish = string.ascii_letters + " "


bits = {}
def dictify(word):
    thedict = {}
    for i, letter in enumerate(word):
        thedict[letter] = (i+1)/len(word)
    return thedict

stib = {}


def map_input_to_values(pairs, thedict):
    result = []
    for pair in pairs:
        mapped_pair = []
        for word in pair:
            mapped_word = tuple(thedict[x] for x in word)
            mapped_pair.append(mapped_word)
        result.append(mapped_pair)
    return result

result = map_input_to_values(DATA, dictify(squish))

def get_word(word):
    global bits
    return tuple(bits[x] for x in word)

def setup():
    global bits
    global DATA 
    global stib
    max_len = max(len(word) for pair in DATA for word in pair)
    DATA = tuple(
        (w1.ljust(max_len), w2.ljust(max_len)) for w1, w2 in DATA
    )
    bits = dictify(squish)
    pp(bits)
    stib = {v: k for k, v in bits.items()}


def convert_data_to_floats(data):
    res = ()
    for pair in data:
        input_word, output_word = pair
        input_floats = tuple(bits[char] for char in input_word)
        output_floats = tuple(bits[char] for char in output_word)
        item = (input_floats, output_floats)
        res += (item,)
    return res

