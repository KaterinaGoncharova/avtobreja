
# coding: utf-8

import nltk
from nltk.collocations import *
from nltk.metrics.spearman import *

words = open('sud.csv').read().split('\n')
finder = BigramCollocationFinder.from_words(words)
words_together = []
for line in words:
    n = line.replace(' ','')
    words_together.append(n.split(';')[1:3])
finder = BigramCollocationFinder.from_documents(words_together)
finder.apply_freq_filter(3)
point_list = finder.nbest(bigram_measures.pmi, 10)# метрика Pointwise Mutual Information.
log_like_list = finder.nbest(bigram_measures.likelihood_ratio, 10) #метрика Log-Likelihood.
golden_standart = [("UDOVLETVORIT'", 'ISK'),
 ("UDOVLETVORIT'", 'HODATAJSTVO'),
 ("PRINYAT'", 'RESHENIE'),
 ('VYNESTI', 'PRIGOVOR'),
 ("PRIZNAT'", 'VINOVNAYA'),
 ("NALOZHIT'", 'AREST'),
 ("SANKCIONIROVAT'", 'AREST'),
 ("PRIZNAT'", 'PRAVOTA'),
 ('PROJTI', 'PRENIE'),
 ("SANKCIONIROVAT'", 'AREST')]
spearman_correlation(ranks_from_sequence(golden_standart),ranks_from_sequence(log_like_list)) #0.5892857142857143
spearman_correlation(ranks_from_sequence(golden_standart),ranks_from_sequence(point_list)) #0.0
spearman_correlation(ranks_from_sequence(log_like_list),ranks_from_sequence(point_list)) #-31.0

#Я импользовала метрики : Log-Likelihood и Pointwise Mutual Information. Золотой стандарт я выбрала руками. 
#Он написан латинецей, так как у меня проблемы с питонов, и кириллицу он ни в какую н захотел читать.
#С помощью теста корреляции Спирманая я нашлакорреляции между моим золотым стандартом и результатми метрик.
#Корреляция между золотым стандартом и Pointwise Mutual Information оказалочсь нулевой.
#Совпадения с золотоым стандартом есть, и достаточно.
#Корреляция составила -31 говорит о том, что что-то не так, так как коэффицент корреляции
#Спирмена принимает значения из отрезка [-1; 1]
