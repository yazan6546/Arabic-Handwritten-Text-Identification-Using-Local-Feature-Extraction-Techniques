import os

def extract_words(directory):
    words = set()
    for root, _, files in os.walk(directory):
        for file in files:
            parts = file.split('_')
            if len(parts) > 2:
                words.add(parts[1])
    return words

def convert_to_arabic(word_set):
    list_words = ['غزال', 'شطيرة', 'فسيكفيكهم', 'قشطة', 'صخر', 'اذن', 'مستهدفين', 'محراس', 'غليظ', 'ابجدية']
    dict_words = {key:word for key, word in zip(word_set, list_words)}

    return dict_words

directory = 'data/isolated_words_per_user/user001'
words_set = extract_words(directory)
print(words_set)
print(convert_to_arabic(words_set))



