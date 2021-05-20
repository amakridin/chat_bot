# from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
# from nltk import edit_distance
# from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.svm import LinearSVC
import pymorphy2
import re
# from nltk.corpus import stopwords
# import nltk
# nltk.download('stopwords')
import random
uniq_intents, intents, examples, stubs = {}, [], [], []
vectorizer = CountVectorizer(ngram_range=(2,4), analyzer='char')
# vectorizer = CountVectorizer()
# vectorizer = TfidfVectorizer()
# regression = LinearSVC()
# regression = LogisticRegression()
regression = SGDClassifier(loss='log')
BOT_CONFIG = ''
regressions = {}
# example: regressions = {"root": regression_root, "plot|host": regression_plot_host, "plot|host|type": regression_plot_host_type ...}
user_position = {}
# example: user_position = c
user_data = {}
# example: user_data = {"user_id1": ["url-iticrm-app1", "cpu"], "user_id2": []}
def load_config():
    global regression, BOT_CONFIG
    with open('bot_config', encoding='utf-8') as f:
        BOT_CONFIG = eval(f.read())
    nn, nn_beg, nn_end = 0, 0, 0
    root_intents = []
    root_examples = []
    stop_examples = []
    stop_intents = []
    for v in BOT_CONFIG['intents']['stop']['examples']:
        stop_intents.append('stop')
        stop_examples.append(v)
    for k, v in BOT_CONFIG['intents'].items():
        if k not in('stub') and k.find('|') < 0:
            for exm in v['examples']:
                exm = clear_text(exm, clear_text=True, to_normal_form=True)
                root_examples.append(exm)
                root_intents.append(k)
        elif k not in('stub') and k.find('|') > 0:
            child_intents = []
            child_examples = []
            for exm in v['examples']:
                exm = clear_text(exm, clear_text=True, to_normal_form=True)
                if exm:
                    child_examples.append(exm)
                    child_intents.append(k)
            print(k, child_examples)
            vectors = vectorizer.fit_transform(child_examples+stop_examples)
            regressions[k] = regression.fit(vectors, child_intents+stop_intents)
        else:
            stubs = v
    print(k, root_examples)
    vectors = vectorizer.fit_transform(root_examples+stop_examples)
    regressions['root'] = regression.fit(vectors, root_intents+stop_intents)


def clear_text(text, clear_text=False, to_normal_form=False):
    ret = text.lower()
    if clear_text:
        ret = ''
        ret = ''.join([char for char in text if char in('абвгдеёжзийклмнопрстуфхцчшщъыьэюя -?')])
    if to_normal_form:
        my_stopwords = ['а', 'более', 'будто', 'бы', 'в', 'вдруг', 'ведь', 'во', 'вот', 'впрочем', 'даже', 'до',
                        'другой', 'если', 'есть', 'еще', 'ж', 'же', 'за', 'и', 'из', 'или', 'им', 'иногда', 'их', 'к',
                        'ли', 'на', 'над', 'ни', 'но', 'ну', 'о', 'об', 'опять', 'от', 'перед', 'по', 'под', 'после',
                        'потом', 'потому', 'почти', 'при', 'про', 'раз', 'разве', 'с', 'сам', 'сейчас', 'со', 'совсем',
                        'так', 'такой', 'там', 'тем', 'теперь', 'то', 'тогда', 'того', 'тоже', 'только', 'том', 'тот',
                        'тут', 'у', 'хоть', 'чего', 'чем', 'через', 'чтоб', 'чтобы', 'чуть', 'я']
        # ret = re.sub(r'[^\w\s]', ' ', ret)
        ret = ret.split(" ")
        ret = [ss for ss in ret if ss != '']
        morph = pymorphy2.MorphAnalyzer()
        f = []
        for word in ret:
            m = morph.parse(word)
            if len(m) != 0:
                wrd = m[0]
                if wrd.normal_form not in (my_stopwords):
                    f.append(wrd.normal_form)
        ret = " ".join(f)
    return ret


if __name__ == "__main__":
    load_config()
    while True:
        replica = input("input: ")
        replica = clear_text(text=replica, clear_text=True, to_normal_form=True)
        vector = vectorizer.transform([replica])
        predict = regressions['root'].predict(vector)
        proba = regressions['root'].predict_proba(vector)
        print(proba)
        ret = False
        for v in proba[0]:
            if v > 0.3:
                ret = True

        # replica = ' '.join(sorted(replica.split(' ')))
        # for exm in BOT_CONFIG['intents'][predict[0]]['examples']:
        #     distance = nltk.edit_distance(s1=replica, s2=exm, transpositions=True)
        #     print(replica, exm, distance, len(exm))
        #     if exm and distance/len(exm) < 0.3:
        #         ret = True
        if ret:
            print(predict)
        else:
            print('stub', predict)
