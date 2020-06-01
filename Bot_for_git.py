
# -*- coding: utf-8 -*-
import random
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from nltk.metrics.distance import edit_distance
from Data_set import BOT_CONFIG

# обработка диалогов из художественной литературы
with open('dialogues.txt', 'r', encoding='utf-8') as f:
    data = f.read()

dialogues = []

for dialogue in data.split('\n\n'):

    replicas = []
    for replica in dialogue.split('\n')[:2]:
        replica = replica[2:].lower()
        replicas.append(replica)

    if len(replicas) == 2 and 3 < len(replicas[0]) < 25 and 3 < len(replicas[1]) < 25:
        dialogues.append(replicas)

GENERATIVE_DIALOGUES = dialogues[:50000]

#Unpacking data_set
X_text = []  # предложения
y = []  # и соответствующие им интенты (группа)

for intent, value in BOT_CONFIG['intents'].items():
    for example in value['examples']:
        X_text.append(example)
        y.append(intent)
#Подбор векторайзера,классификатора и обучение модели.
VECTORIZER = TfidfVectorizer(analyzer='char', norm='l2', ngram_range=(1, 3), sublinear_tf=True)
X = VECTORIZER.fit_transform([x.lower() for x in X_text])

clf = LogisticRegression(C=1000, class_weight='balanced')
clf.fit(X, y)

# TEST
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
# clf.fit(X_train,y_train)
# N=10
# scores=[]
# for i in range(N):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
#     clf = LogisticRegression()
#     clf.fit(X_train, y_train)
#     score=clf.score(X_test,y_test)
#     scores.append(score)
#     print(sum(scores)/len(scores))
statistic = {
    'requests': 0,
    'bydataset': 0,
    'bygenerative': 0,
    'failer': 0
}

CLASSIFIER_THRESHOLD = 0.3
GENERATIVE_THRESHOLD = 0.7


def get_intent(text):
    probas = clf.predict_proba(VECTORIZER.transform([text]))
    max_proba = max(probas[0])
    if max_proba >= CLASSIFIER_THRESHOLD:
        index = list(probas[0]).index(max_proba)
        return clf.classes_[index]


def get_answer_by_generative_model(text):
    text = text.lower()
    for question, answer in GENERATIVE_DIALOGUES:
        if abs(len(text) - len(question)) / len(question) < 1 - GENERATIVE_THRESHOLD:
            dist = edit_distance(text, question)
            l = len(question)
            similarity = 1 - dist / l
            if similarity > GENERATIVE_THRESHOLD:
                return answer


def get_response_by_intent(intent):
    responses = BOT_CONFIG['intents'][intent]['responses']
    return random.choice(responses)


def get_failure_phrase():
    phrases = BOT_CONFIG['failure_phrases']
    return random.choice(phrases)


def generate_answer(text):
    statistic['requests'] += 1

    # NLU
    intent = get_intent(text)

    # Make answer

    # by script
    if intent:
        statistic['bydataset'] += 1
        response = get_response_by_intent(intent)
        return response

    # use generative model
    if not intent:
        answer_by_generative_model = get_answer_by_generative_model(text)
        if answer_by_generative_model:
            statistic['bygenerative'] += 1
            return answer_by_generative_model

    # use stub
    failure_phrase = get_failure_phrase().capitalize()
    statistic['failer'] += 1
    return failure_phrase



from telegram.ext import Updater, CommandHandler, MessageHandler, Filters


def start(update, context):
    """Send a message when the command /start is issued."""
    update.message.reply_text('Whassup:)')


def help(update, context):
    """Send a message when the command /help is issued."""
    update.message.reply_text('Help!')


def text(update, context):
    """Echo the user message."""
    answer = generate_answer(update.message.text)
    print(update.message.text, '->', answer)
    print(statistic)
    print()

    update.message.reply_text(answer)


def error(update, context):
    """Log Errors caused by Updates."""
    update.message.reply_text('Я  понимаю только текст :(')


def main():
    """Start the bot."""
    updater = Updater("TOKEN",
                      request_kwargs={'proxy_url': 'socks5://5.133.194.171:12951/'},
                      use_context=True)

    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help))
    dp.add_handler(MessageHandler(Filters.text, text))
    dp.add_error_handler(error)
    updater.start_polling()
    updater.idle()


main()
