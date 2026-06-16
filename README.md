# Групповой проект 5.
## Задача 1. Работа с табличными данными
### Датасет
https://www.kaggle.com/datasets/willianoliveiragibin/10000-data-about-movies-1915-2023?select=data.csv
### Колонки
- movie_name: The title of the movie.
- year: The year of the movie's release.
- rating: IMDb user rating.
- metascore: Metascore rating.
- gross_income: Gross income of the movie.
- votes: Number of votes on IMDb.
- runtime: Duration of the movie in minutes.
- genre: Genre(s) of the movie.
- certificate: Certification or rating of the movie.
- description: A brief summary or plot of the movie.
- directors: Director(s) of the movie.
- stars: Main cast or actors of the movie.
### Бизнес-задача
Хотим прогнозировать зрительский рейтинг по его характеристикам. Это позволит нам:
- принимать решения о том, **какие фильмы покупать** в каталог;
- определять, **какой контент продвигать** на главной странице.
### Выбранные модели
1. Мелкая сеть с ReLu
```
model1 = nn.Sequential(
    nn.Linear(X_train_t.shape[1], 64),
    nn.ReLU(),
    nn.Linear(64, 1))
```

2. Глубокая сеть с ReLu
```
model2 = nn.Sequential(
    nn.Linear(X_train_t.shape[1], 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)
```

3. Глубокая сеть с ReLu и Dropout
```
model3 = nn.Sequential(
    nn.Linear(X_train_t.shape[1], 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 128),
    nn.ReLU(), 
    nn.Dropout(0.3),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)
```

4. Глубокая сеть с GELU + AdamW
```
model4 = nn.Sequential(
    nn.Linear(X_train_t.shape[1], 256),
    nn.GELU(),
    nn.Dropout(0.3),
    nn.Linear(256, 128),
    nn.GELU(),
    nn.Dropout(0.3),
    nn.Linear(128, 64),
    nn.GELU(),
    nn.Linear(64, 1)
)
```

### Получившиеся результаты
<img width="962" height="175" alt="Screenshot 2026-06-16 at 10 56 33" src="https://github.com/user-attachments/assets/f4df1f96-0d00-4ea3-bfe3-41227d522023" />

### Вывод
Видим, что наилучшее качество на тесте выбилось у мелкой сети, а также у глубокой с GeLu и AdamW
Средняя ошибка около 0.46, что по предсказаниям по 10балльной шкале с шагом 0.1 довольно неплохо, учитывая, что эти оценки довольно субективны в рамках одного отзыва. Дальнейшее улучшение качества предсказания возможно с помощью добавления новых признаков, которые будут иметь корреляцию сильнее, а также более точный подбор архитектуры модели для решения задачи

