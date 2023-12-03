#імпортуємо клас TfidfVectorizer з бібліотеки scikit-learn, що використовується для обробки текстових даних і створення TF-IDF векторів
from sklearn.feature_extraction.text import TfidfVectorizer

#присвоєюмо змінні для двох текстових файлів
text_path_1 = 'Lazarus.txt'
text_path_2 = 'psychology.txt'

#відкриваємо і читаємо їх
with open(text_path_1, 'r', encoding='utf-8') as file1:
    text1 = file1.read()

with open(text_path_2, 'r', encoding='utf-8') as file2:
    text2 = file2.read()

#Створення векторизатора TF-IDF з власними налаштуваннями
tfidf_vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words=None, encoding='utf-8')

#Перетворення тексту в матриці TF-IDF
tfidf_matrix1 = tfidf_vectorizer.fit_transform([text1])
tfidf_matrix2 = tfidf_vectorizer.transform([text2])

#Отримання назв елементів з векторизатора
terms = tfidf_vectorizer.get_feature_names_out()

#Обчислення суми оцінок TF-IDF для кожного елементу в матрицях
tfidf_sum1 = tfidf_matrix1.sum(axis=0)
tfidf_sum2 = tfidf_matrix2.sum(axis=0)

#Створення списку кортежів, що містять термін та його оцінку TF-IDF для кожного тексту
tfidf_scores1 = [(term, tfidf_sum1[0, index]) for term, index in enumerate(range(len(terms)))]
tfidf_scores2 = [(term, tfidf_sum2[0, index]) for term, index in enumerate(range(len(terms)))]

#Сортування списків кортежів на основі оцінок TF-IDF за спаданням
tfidf_scores1.sort(key=lambda x: x[1], reverse=True)
tfidf_scores2.sort(key=lambda x: x[1], reverse=True)

#виводимо результат (обрізаємо 10)
print("Лазарус:")
for term, score in tfidf_scores1[:10]:
    print(f"{terms[term]}: {score}")

print("\nПсихологія:")
for term, score in tfidf_scores2[:10]:
    print(f"{terms[term]}: {score}")

