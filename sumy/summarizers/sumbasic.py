from __future__ import absolute_import, division

from collections import defaultdict
from warnings import warn

from ._summarizer import AbstractSummarizer


class SumBasicSummarizer(AbstractSummarizer):
    _stop_words = frozenset()

    @property
    def stop_words(self):
        return self._stop_words

    @stop_words.setter
    def stop_words(self, words):
        self._stop_words = frozenset(map(self.normalize_word, words))

    def __call__(self, document, sentences_count):
        distribution = self._get_distribution(document)
        sentences = list(document.sentences)

        if len(distribution) < len(sentences):
            message = (
                "Number of words (%d) is lower than number of sentences (%d). "
                "SumBasic algorithm may not work properly."
            )
            warn(message % (len(distribution), len(sentences)))

        ranks = defaultdict(int)
        step = 0

        while sentences:
            word = sorted(distribution, key=distribution.get, reverse=True)[0]
            ith_sentence = self._get_best_sentence(word, sentences, distribution)
            if not ith_sentence:
                # this word is not present in any of remaining sentences
                # we can safely remove it
                del distribution[word]
                continue
            ranks[ith_sentence] = 1 / (step + 1)
            sentences.remove(ith_sentence)
            for word in ith_sentence.words:
                distribution[self.stem_word(word)] **= 2
            step += 1
        return self._get_best_sentences(document.sentences, sentences_count, ranks)

    def _get_distribution(self, document):
        counts = defaultdict(int)
        for word in document.words:
            if word not in self.stop_words:
                counts[self.stem_word(word)] += 1

        for word in counts:
            counts[word] /= len(counts)

        return counts

    def _get_best_sentence(self, main_word, sentences, distribution):
        averages = {}
        for sentence in sentences:
            weight = 0
            is_candidate = False
            for word in sentence.words:
                stemmed = self.stem_word(word)
                weight += distribution[stemmed]
                if stemmed == main_word:
                    is_candidate = True
            if is_candidate:
                averages[sentence] = weight / len(sentence.words)
        if averages:
            return sorted(averages, key=averages.get, reverse=True)[0]
        return None
