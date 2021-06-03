# Используем обработку текста из nltk
import nltk
nltk.download("stopwords")
# Для создания dataframe
import pandas as pd

# Используем модель эмбедингов на базе CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Считаем Эвклидову метрику векторов из scipy
import scipy as sp

# Получем список файлов из директории с текстами
import os

# Наши глобальные параметры
DB_PATH = "db" # Путь к папке с базой статей
FILE_FORMAT = ".txt" # Формат фалов с статьями
RES_SIZE = 3 # Кол-во рекомендуемых статей на выходе


# Создаем обьект нашей модели, в форме класса, наследуя его от CountVectorizer
# Используем CountVectorizer из бибилиотеки sklearn.feature_extraction.text
# Он анализирует входящий корпус теекстов и создает по нимсловарь встречающихся в нем слов
# сопоставляя ему матрицу частоты использования каждого слова в конкретном тексте
# Формат класса StemmedCountVectorizer определен в документации к бибиотеке
class OurModel(CountVectorizer):
    def __init__(self, **kv):
        super(OurModel, self).__init__(**kv)
        # Для токенов(нормализованных слов) перед векторизацией используем Стемминг - те оставлеям только корень слова
        # Другой вариант это Лемматизация - использование инфинитива слова (более ресурсоемкая)
        self._stemmer = nltk.stem.snowball.RussianStemmer(
            'russian'
        )

    def build_analyzer(self):
        analyzer = super(OurModel, self).build_analyzer()
        return lambda doc: (self._stemmer.stem(w) for w in analyzer(doc))


# Имя вектора частоты терминов в каждом тексте наиболее простой способ анализа схожести текстов
# это эвклидова метрика расстояния между векторами
def euclid_metric(vec1, vec2):
    delta = vec1/sp.linalg.norm(vec1.toarray()) - vec2/sp.linalg.norm(vec2.toarray())
    return sp.linalg.norm(delta.toarray())


def main():
    # Получем список файлов с текстами
    file_list = []
    # Первый список с полным путем, второй просто имя файла
    file_index = []
    for root, dirs, files in os.walk(DB_PATH):
        for file in files:
            if (file.endswith(FILE_FORMAT)):
                file_index.append(file)
                file_list.append(os.path.join(root, file))

    # Создаем список содержимого файлов
    texts = []
    headers = []
    for file_name in file_list:
        fd = open(file_name, 'r', encoding="cp1251")
        link = fd.readline().strip()
        header = fd.readline().strip()
        headers.append((link, header))
        text = fd.read()
        texts.append(header + "\n" + text)

    # Печатаем содержимое базы данных
    print("Содрежание базы данных -")
    for i in range(len(headers)):
        n = i + 1
        print(f"({n})", headers[i][1], "-", headers[i][0])

    # инициализируем класс нашей модели задавая шаблон (в форме внутреннего реуглярного выражения) для выделения слов
    vectorizer = OurModel(
        min_df=1,
        # token_pattern ожидает свой формат регулрного выражения где в списке указаны все возможные символы слова
        # а в {} размер слова от и до
        token_pattern=r'[ЁАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяё]{4,}'
    )

    # Получаем наш тензор, где кадому слову сопоставлена его частота в конкретном тексте
    x = vectorizer.fit_transform(texts)

    # Размерность нашей матрицы тензора
    texts_num, words_num = x.shape

    # Создаем DataFrame корпуса текстов
    # Мешок слов - будет индексом для строк исходящего DataFrame
    words_index = vectorizer.get_feature_names()
    # Столбы - это имена файлов с статьями из file_index
    # Это матрица частоты слов
    nums = x.toarray().transpose()
    df = pd.DataFrame(nums, words_index, file_index)
    # Экспортируем в csv
    df.to_csv('dataframe.csv', encoding="cp1251")
    print("\nDataFrame корпуса текстов выгружен в dataframe.csv (кодировка cp1251)")

    # Запрашиваем понравившиеся статьи
    target = input("\nВведите № понравившейся статьи - ")
    target = int(target) - 1

    # Проходим по матрице и считаем метрику между целевым вектором и остальными
    res = []
    for i in range(0, texts_num):
        res.append([i, euclid_metric(x[target], x[i])])
    # Сортируем результат
    res.sort(key = lambda x: x[1])

    # Вывод результата
    print("\nРанжированный список рекомендуемых статей на основании понравившейся:")
    for i in range(1, min(RES_SIZE + 1, len(res))):
        n = res[i][0] + 1
        euc = res[i][1]
        print(i, "-", f"({n})", headers[res[i][0]][1], "-", headers[res[i][0]][0])


if __name__ == "__main__":
    main()

