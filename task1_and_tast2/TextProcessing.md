# 文本预处理

文本是一类序列数据，一篇文章可以看错是字符或单词的序列。

文本数据预处理步骤：

1.读入文本

2.分词

3.建立字典，将每个词映射到一个唯一索引

4.将文本从词的序列转换成索引序列，方便输入模型

## code

以Time Machine为例，对文本进行预处理。

手动建立一个文本词典

```python
import re
import collections

#read txt
def read_a_white_king():
    with open('a_white_king_in_east_africa.txt','r',encoding='utf-8') as f:
        #lines = [re.sub('[^a-z]+',' ',line.strip().lower()) for line in f]
        lines = [re.sub('[^a-z]+', ' ', line.strip().lower()) for line in f]
    return lines

lines = read_a_white_king()
print(len(lines))

#分词，把文章进行分句
def tokenize(sentences,token='word'):
    """
    :param sentences: 文本内容
    :param token: 分词模式
    :return: list，每个子列别为一个句子
    """
    # word、char两种模式，word以句子为单位，char以字符为单位
    if token == 'word':
        return [sentence.split(' ') for sentence in sentences]
    elif token == 'char':
        return [list(sentence) for sentence in sentences]
    else:
        print('ERROR:unkown token type' + token)

token = tokenize(lines)


#统计词频
def count_corpus(sentences):
    """
    :param sentences: 文本，list格式
    :return: 返回一个字典，记录每个词的出现次数
    """
    tokens = [tk for st in sentences for tk in st]
    print(tokens[0:4])
    return collections.Counter(tokens)  #collections.Counter()，统计出现字符出现的次数，以字典形式返回

# print(count_corpus(token))
#建立字典
"""
 1、去重；2、进行映射；3、构建字典
"""
class Vocab(object):
    def __init__(self,tokens,min_freq=0,use_special_tokens=False):
        counter = count_corpus(tokens)      #<key,value>:<词，词频>
        # print('counter.items()',counter.items())
        # 字典.items() 以列表返回可遍历的（键、值）元组数组。
        self.token_freqs = list(counter.items())        #转换成list
        self.index_to_token = []    #
        if use_special_tokens:
            # pad为填充的方式，sentence大小不一，对短句进行补充，
            # 为句子添加开始（bos）和结尾标志（eos），
            # 语料库中没有出现时，把其当作Unkown进行处理，使用
            self.pad,self.bos,self.eos,self.unk = (0,1,2,3)
            self.index_to_token += ['<pad>','<bos>','<eos>','<unk>']
        else:
            self.unk = 0
            self.index_to_token += ['<unk>']

        # 去重，并去除freq小于min_freq的词
        self.index_to_token += [token for token,freq in self.token_freqs
                                if freq >= min_freq and token not in self.index_to_token]

        # 重新建立字典
        self.token_to_idx = dict()
        for idx,token in enumerate(self.index_to_token):
            self.token_to_idx[token] = idx

    # 计算字典长度
    def __len__(self):
        return len(self.index_to_token)

    # 给定词查询索引
    def __getitem__(self, tokens):
        if not isinstance(tokens,(list,tuple)): #不是列表或者元组，则直接到tokens中寻找，若没有则返回unk
            return self.token_to_idx.get(tokens,self.unk)
        #如果是list或者tuple，则依次查询
        return [self.__getitem__(token) for token in tokens]

    # 给定索引返回对应的词
    def to_tokens(self,indices):
        if not isinstance(indices,(list,tuple)):
            return self.index_to_token[indices]
        return [self.index_to_token[index] for index in indices]

res = Vocab(token)
print(list(res.token_to_idx.items())[0:10])
```

使用说动分词有几个缺点：

1.标点符号通常提供语义信息，但我们的方法会直接将其丢弃

2.类似“shouldn't、doesn't”这样的词会被错误的处理

3.类似"Mr、Dr"这样的词被错误的处理

故，使用现有工具进行分词：（spacy和nltk）



