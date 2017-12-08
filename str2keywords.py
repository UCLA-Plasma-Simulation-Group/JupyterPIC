import re
import numpy as np

# split id and keywords by whitespece or , or ;
id_key_pattern = re.compile(r'[\s,;]+')
# split keywords by commas except for those inside () and []
keyword_pattern = re.compile(r',(?!(?:[^([]*[(\[][^)\]]*[)\]])*[^()\[\]]*[)\]])')


# Here we assume the brackets are all balanced.
# Otherwise the keyword dictionary would cause runtime failure sooner or later
class str2keywords:
    def __init__(self, string):
        self.id, string = id_key_pattern.split(string + ' ', maxsplit=1)
        # remove whitespaces
        self.id = "".join(self.id.split())
        string = "".join(string.split())
        string = keyword_pattern.split(string)
        if string[-1] == '':
            string.pop(-1)
        # store keywords as dictionary
        self.keywords = dict(tuple(item.split('=')) for item in string)
        for k, v in self.keywords.iteritems():
            self.keywords[k] = eval(v)

    def __eq__(self, other):
        return self.id == other

# some tests
if __name__ == '__main__':
    # first section of the string is the operation id, the rest is keywords
    kw = str2keywords('fft norm="ortho" , axes=(0, 1 ),  s=[3,3]')
    a = np.mgrid[:3, :3][0]
    if kw == 'fft':
        # use ** to unpack the dictionary
        a = np.fft.fft2(a, **kw.keywords)
    print(a)
