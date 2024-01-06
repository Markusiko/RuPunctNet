# RuPunctNet

Данный проект посвящен автоматическим способам коррекции пунктуации в тексте. Главной целью является создание сервиса (например, telegram-бота), который будет обрабатывать предложения пользователя и возвращать аналогичный текст, в котором верно (с точки зрения правил русского языка) расставлены знаки препинания (`.`, `,`, `?`, `!`).

Автоматическая корректировка будет совершаться моделью машинного обучения, выбранной по результатам исследования, проводимого также в рамках данного проекта. Сама задача восстановления правильных знаков препинания обычно решается как задачи классификации сущностей (NER).


## Примерный план работы
1. В рамках разведочного анализа данных будут сделаны: просмотр примеров и фильтрация данных (проверка на их корректность), статистический анализ различных метрик используемых текстов (например, длин слов и предложений в них), просмотр баланса классов, анализ контекстов слов, при которых обычно ставятся знаки препинания. На основе EDA могут быть приняты решения о некоторых изменениях в данных перед самим этапом моделирования.

2. Далее будет построен некий бейзлайн. В качестве него, скорее всего, выступит некий rule-based алгоритм или модель на основе частот слов (например, бустинг на основе TF-IDF признаках). 

3. На следующем этапе будут построены модели глубинного обучения (реккурентные сети и трансформеры) для решения поставленной задачи.

4. В конце концов будет произведена разработка сервиса, в котором используется наилучшая модель, выбранная в рамках пунктов 2 и 3.

Еще пункты, которые хочется сделать: 
* Хочется посмотреть возможность решения задачи не как NER, а как задач seq2seq или заполнение масок.
* Дополнительно в рамках исследования хочется сравнить качество построенных моделей с zero-shot подходом LLM-моделей (ChatGPT, GigaChat и другие).
* Возможно сравнение качества работы модели с человеческими способностями аналогично тому, как это было сделано в научных работах по данной тематике.


## Команда

* Никифоров Николай (@nikiforov_uze_bezit)
* Столяров Марк (@markusikk)


## Описание папок
* Все материалы первого этапа  (сбор данных) содержатся в папке `data/`
* Файлы с разведочным анализом данных содержатся в папке `EDA/`. Там же содержится скрипт с созданием разметки
