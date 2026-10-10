import os
import re
import logging
from telegram import (
    Update, ReplyKeyboardMarkup, KeyboardButton
)
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    filters, ContextTypes, ConversationHandler
)
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================================
# Настройка логирования
# ==========================================================
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

TOKEN = os.getenv("BOT_TOKEN")
if not TOKEN:
    raise ValueError("Не задан BOT_TOKEN в переменных окружения")

# ==========================================================
# Положение вуза
# ==========================================================
POLOZHENIE = """
3.1. Документ считается действительным при наличии подписи и печати вуза.
3.2. Справка об обучении должна содержать: ФИО студента, название вуза, направление подготовки, форму обучения, период обучения, номер и дату выдачи.
3.3. Заявление на академический отпуск должно содержать: ФИО, группу и курс, причину отпуска, дату начала и окончания отпуска, дату заявления, подпись студента.
3.4. Справка о практике должна содержать: ФИО, организацию, сроки практики, руководителя практики, оценку, номер и дату выдачи, подпись и печать.
3.5. Для перевода из другого вуза обязательна справка об обучении с указанием периода обучения.
3.6. Отсутствие подписи или печати делает документ недействительным.
3.7. Окончательное решение о приёме документов принимает сотрудник деканата.
"""

rules = []
for line in POLOZHENIE.strip().split("\n"):
    m = re.match(r"(\d+\.\d+)\.\s*(.+)", line.strip())
    if m:
        rules.append({"number": m.group(1), "text": m.group(2)})

vectorizer = TfidfVectorizer()
matrix = vectorizer.fit_transform([r["text"] for r in rules])


def search_rules(query: str, top_k: int = 3):
    q = vectorizer.transform([query])
    sims = cosine_similarity(q, matrix)[0]
    idx = sims.argsort()[::-1][:top_k]
    return [
        {
            "number": rules[i]["number"],
            "text": rules[i]["text"],
            "score": float(sims[i]),
        }
        for i in idx
    ]


# ==========================================================
# Типы документов
# ==========================================================
DOCUMENT_TYPES = {
    "Справка об обучении": [
        "ФИО", "Вуз", "Направление", "Форма обучения",
        "Период обучения", "Номер и дата выдачи", "Печать"
    ],
    "Заявление на отпуск": [
        "ФИО", "Группа и курс", "Причина отпуска",
        "Дата начала отпуска", "Дата окончания отпуска",
        "Дата заявления", "Подпись"
    ],
    "Справка о практике": [
        "ФИО", "Организация", "Сроки практики",
        "Руководитель практики", "Оценка",
        "Номер и дата выдачи", "Печать"
    ],
}


# ==========================================================
# База интентов
# ==========================================================
INTENTS = {
    "перевод_возможен": {
        "keywords": ["перевод", "перевестись", "перейти", "другой вуз", "другое учебное", "сменить вуз"],
        "questions": [
            "хватит ли этой справки для перевода",
            "можно ли перевестись с этим документом",
            "достаточно ли документа для перевода",
            "подходит ли справка для перевода",
            "можно ли перейти в другой вуз",
            "примут ли меня с этой справкой",
            "достаточно ли для перевода на второй курс",
            "хватит ли для перевода",
            "можно ли перевестись",
            "получится ли перевестись",
            "подойдёт ли справка для перевода",
            "перевод возможен",
            "есть ли возможность перевода",
            "достаточно данных для перевода",
            "можно ли сменить вуз",
            "переведут ли меня",
            "примут ли документ для перевода",
            "как перевестись",
            "что нужно для перевода",
        ],
        "answer_template": (
            "Для перевода из другого вуза обязательна справка об обучении "
            "с указанием периода обучения (п. 3.5 Положения). "
            "Проверьте, что в документе указан период обучения."
        ),
    },

    "проверить_документ": {
        "keywords": ["в порядке", "всё хорошо", "проверить", "правильно", "соответствует", "годен", "действителен"],
        "questions": [
            "всё ли в порядке с документом",
            "документ в порядке",
            "документ правильный",
            "справка в порядке",
            "всё ли хорошо",
            "нет ли ошибок",
            "соответствует ли документ",
            "правильно ли оформлен документ",
            "можно ли принять документ",
            "примут ли документ",
            "годен ли документ",
            "действителен ли документ",
            "всё ли правильно",
            "проверь документ",
            "проверьте документ",
            "что скажешь про документ",
            "всё ли на месте",
            "все ли поля заполнены",
            "всё ли заполнено",
            "есть ли проблемы с документом",
            "найди ошибки",
            "есть ли ошибки в документе",
        ],
        "answer_template": None,
    },

    "чего_не_хватает": {
        "keywords": ["не хватает", "чего нет", "отсутствует", "недостаёт", "пропущено", "не заполнено"],
        "questions": [
            "чего не хватает",
            "чего не хватает в документе",
            "чего не хватает в справке",
            "что отсутствует",
            "чего нет в документе",
            "каких полей нет",
            "какие поля отсутствуют",
            "чего недостаёт",
            "что нужно добавить",
            "что дописать",
            "какие данные отсутствуют",
            "какие обязательные поля",
            "что должно быть в документе",
            "недостающие поля",
            "не заполнено",
            "пустые поля",
            "что не так с документом",
        ],
        "answer_template": None,
    },

    "печать": {
        "keywords": ["печать", "штамп", "оттиск"],
        "questions": [
            "есть ли печать",
            "есть ли печать на документе",
            "есть ли печать вуза",
            "есть ли печать организации",
            "поставлена ли печать",
            "нужна ли печать",
            "обязательна ли печать",
            "есть ли оттиск печати",
            "печать есть",
            "печать отсутствует",
            "где печать",
            "почему нет печати",
            "можно ли без печати",
            "примут ли без печати",
            "что делать если нет печати",
            "печать вуза",
            "есть ли штамп",
        ],
        "answer_template": None,
    },

    "подпись": {
        "keywords": ["подпись", "подписал", "подписано"],
        "questions": [
            "есть ли подпись",
            "есть ли подпись студента",
            "есть ли подпись руководителя",
            "поставлена ли подпись",
            "нужна ли подпись",
            "обязательна ли подпись",
            "есть ли подпись на документе",
            "подпись есть",
            "подпись отсутствует",
            "где подпись",
            "почему нет подписи",
            "можно ли без подписи",
            "примут ли без подписи",
            "что делать если нет подписи",
            "подпись студента",
        ],
        "answer_template": None,
    },

    "даты": {
        "keywords": ["дата", "срок", "период", "когда"],
        "questions": [
            "есть ли дата выдачи",
            "есть ли дата",
            "указана ли дата",
            "есть ли период обучения",
            "указан ли период",
            "есть ли сроки практики",
            "указаны ли сроки",
            "есть ли даты начала и окончания",
            "какие даты указаны",
            "есть ли номер и дата выдачи",
            "есть ли дата заявления",
            "все ли даты указаны",
            "когда выдана справка",
            "даты практики",
            "даты отпуска",
            "дата выдачи",
        ],
        "answer_template": None,
    },

    "фио": {
        "keywords": ["фио", "имя", "фамилия", "студент"],
        "questions": [
            "есть ли фио",
            "указано ли фио",
            "есть ли фамилия",
            "есть ли имя",
            "есть ли фио студента",
            "чьё имя в документе",
            "как зовут студента",
            "фио указано",
            "фио не указано",
        ],
        "answer_template": None,
    },

    "вуз_организация": {
        "keywords": ["вуз", "университет", "институт", "организация", "компания"],
        "questions": [
            "есть ли название вуза",
            "указан ли вуз",
            "есть ли университет",
            "есть ли название организации",
            "указана ли организация",
            "какой вуз в документе",
            "какая организация",
            "название вуза указано",
            "вуз не указан",
        ],
        "answer_template": None,
    },

    "направление_форма": {
        "keywords": ["направление", "специальность", "форма обучения", "очная", "заочная"],
        "questions": [
            "есть ли направление",
            "указано ли направление",
            "есть ли специальность",
            "указана ли специальность",
            "есть ли форма обучения",
            "указана ли форма обучения",
            "очная или заочная",
            "какая форма обучения",
            "какое направление",
        ],
        "answer_template": None,
    },

    "причина_отпуска": {
        "keywords": ["причина отпуска", "почему отпуск", "основание отпуска"],
        "questions": [
            "есть ли причина отпуска",
            "указана ли причина отпуска",
            "есть ли основание для отпуска",
            "почему академический отпуск",
            "какая причина отпуска",
        ],
        "answer_template": None,
    },

    "оценка": {
        "keywords": ["оценка", "балл", "отлично", "хорошо"],
        "questions": [
            "есть ли оценка",
            "указана ли оценка",
            "есть ли балл",
            "какая оценка",
            "поставлена ли оценка",
        ],
        "answer_template": None,
    },

    "руководитель": {
        "keywords": ["руководитель", "руководитель практики", "наставник"],
        "questions": [
            "есть ли руководитель практики",
            "указан ли руководитель",
            "есть ли руководитель",
            "кто руководитель практики",
        ],
        "answer_template": None,
    },

    "кто_решает": {
        "keywords": ["кто решает", "кто принимает", "кто решает вопрос"],
        "questions": [
            "кто принимает решение",
            "кто решает вопрос",
            "кто принимает документ",
            "кто рассматривает документы",
            "от кого зависит решение",
            "как принимается решение",
            "кто решает окончательно",
            "кто ответственный",
        ],
        "answer_template": (
            "Окончательное решение о приёме документов принимает "
            "сотрудник деканата (п. 3.7 Положения)."
        ),
    },

    "что_делать": {
        "keywords": ["что делать", "как быть", "куда обращаться"],
        "questions": [
            "что делать",
            "что делать если документ не подходит",
            "что делать если не хватает полей",
            "куда обращаться",
            "куда обратиться",
            "к кому обратиться",
            "как исправить документ",
            "что делать дальше",
        ],
        "answer_template": (
            "Обратитесь в деканат — сотрудник поможет исправить документ "
            "и подскажет, что нужно дополнить."
        ),
    },
}


def detect_intent(question: str):
    """Определяет интент вопроса."""
    q = question.lower().strip()
    if not q:
        return None

    best_intent = None
    best_score = 0.0

    for intent_name, intent_data in INTENTS.items():
        keyword_hits = sum(1 for kw in intent_data["keywords"] if kw in q)

        question_hits = 0
        q_words = set(w for w in q.split() if len(w) > 3)
        for ready_q in intent_data["questions"]:
            rq_words = set(w for w in ready_q.split() if len(w) > 3)
            if not rq_words:
                continue
            overlap = len(q_words & rq_words) / len(rq_words)
            if overlap >= 0.5:
                question_hits += 1

        total_score = keyword_hits * 2 + question_hits

        if total_score > best_score:
            best_score = total_score
            best_intent = intent_name

    return best_intent if best_score >= 2 else None


def detect_doc_type(text: str):
    """Автоопределение типа документа."""
    t = text.lower()

    if ("академическ" in t and "отпуск" in t) or "причина отпуска" in t:
        return "Заявление на отпуск"

    if "практик" in t and ("руководитель" in t or "сроки" in t or "оценка" in t):
        return "Справка о практике"

    if "справка" in t and (
        "обучени" in t or "период обучения" in t or "форма обучения" in t
    ):
        return "Справка об обучении"

    return None


# ==========================================================
# OCR
# ==========================================================
def recognize_text(image_path: str) -> str:
    """OCR с предобработкой."""
    try:
        img = Image.open(image_path)
        img = img.convert("L")
        w, h = img.size
        if max(w, h) < 2000:
            img = img.resize((w * 2, h * 2), Image.LANCZOS)
        img = ImageEnhance.Contrast(img).enhance(1.8)
        img = img.filter(ImageFilter.MedianFilter(size=3))
        img = ImageEnhance.Sharpness(img).enhance(1.5)
        img = img.point(lambda x: 0 if x < 140 else 255, "1")
        config = "--psm 6 --oem 3"
        return pytesseract.image_to_string(img, lang="rus+eng", config=config)
    except Exception as e:
        logger.error(f"OCR error: {e}")
        return ""


# ==========================================================
# УМНОЕ ИЗВЛЕЧЕНИЕ ФИО
# ==========================================================
DATE_PATTERN = r"\b\d{2}[./-]\d{2}[./-]\d{2,4}\b"

# Стоп-слова — то, что НЕ является ФИО
FIO_STOPWORDS = {
    "российская", "федерация", "федеральный", "министерство", "науки",
    "высшего", "образования", "университет", "институт", "академия",
    "колледж", "школа", "лицей", "гимназия", "факультет", "кафедра",
    "деканат", "ректорат", "студент", "студентка", "обучающийся",
    "группа", "курс", "справка", "заявление", "практика", "практики",
    "руководитель", "подпись", "печать", "дата", "номер", "место",
    "город", "москва", "санкт", "петербург", "россия", "рф", "г",
    "полное", "наименование", "фамилия", "имя", "отчество",
    "декану", "декана", "проректору", "ректору", "заведующему",
    "выдана", "выдан", "выдано", "справка", "справки",
}

# Маркеры — слова, после которых обычно идёт ФИО
FIO_MARKERS = [
    "фио", "фамилия имя отчество", "студент", "студентка",
    "обучающийся", "обучающаяся", "на имя", "кому выдан",
    "ф.и.о", "ф. и. о", "выдан", "выдана", "я,",
]

# Известные русские фамилии (топ-100)
COMMON_SURNAMES = {
    "иванов", "петров", "сидоров", "смирнов", "кузнецов", "попов",
    "васильев", "соколов", "михайлов", "новиков", "фёдоров", "морозов",
    "волков", "алексеев", "лебедев", "семёнов", "егоров", "павлов",
    "козлов", "степанов", "николаев", "орлов", "андреев", "макаров",
    "никитин", "захаров", "зайцев", "соловьёв", "борисов", "яковлев",
    "григорьев", "романов", "воробьёв", "сергеев", "кузьмин", "фролов",
    "александров", "дмитриев", "королёв", "гусев", "киселёв", "ильин",
    "максимов", "поляков", "сорокин", "виноградов", "ковалёв", "белов",
    "медведев", "антонов", "тарасов", "жуков", "баранов", "филиппов",
    "комарров", "давыдов", "беляев", "герасимов", "богданов", "осипов",
    "сидоров", "матвеев", "титов", "марков", "миронов", "крючков",
    "куликов", "карпов", "афанасьев", "власов", "масленников",
    "исаев", "тюрин", "горбунов", "шубин", "евдокимов", "ершов",
    "фомин", "данилов", "сафонов", "евсеев", "никифоров", "грачёв",
    "шумилов", "щукин", "трофимов", "малов", "ермаков", "кириллов",
    "селиверстов", "панов", "савельев", "голубев", "веселов",
    "зуев", "смирнова", "иванова", "петрова", "кузнецова",
}

# Паттерны ФИО
FIO_PATTERNS = [
    # Иванов Иван Иванович
    r"\b([А-ЯЁ][а-яё]{2,}(?:ов|ев|ин|ын|ский|цкий|ой|ая|ко|енко|ук|юк|ян|янц|дзе|швили|оглы))\s+([А-ЯЁ][а-яё]{2,})\s+([А-ЯЁ][а-яё]{2,}(?:вич|вна|ична|инична))\b",
    # Иванова Мария Петровна (женская форма)
    r"\b([А-ЯЁ][а-яё]{2,}(?:ова|ева|ина|ына|ская|цкая|ая|ко|енко|ук|юк|ян|янц|дзе|швили))\s+([А-ЯЁ][а-яё]{2,})\s+([А-ЯЁ][а-яё]{2,}(?:вич|вна|ична|инична))\b",
    # Иванов Иван
    r"\b([А-ЯЁ][а-яё]{2,}(?:ов|ев|ин|ын|ский|цкий|ой|ая|ко|енко|ук|юк|ян|янц|дзе|швили|оглы))\s+([А-ЯЁ][а-яё]{2,})\b",
    # Иванов И. И.
    r"\b([А-ЯЁ][а-яё]{2,}(?:ов|ев|ин|ын|ский|цкий|ой|ая|ко))\s+([А-ЯЁ]\.\s?[А-ЯЁ]\.)\b",
    # И. И. Иванов
    r"\b([А-ЯЁ]\.\s?[А-ЯЁ]\.)\s+([А-ЯЁ][а-яё]{2,}(?:ов|ев|ин|ын|ский|цкий))\b",
    # Общая форма из 3 слов
    r"\b([А-ЯЁ][а-яё]{2,})\s+([А-ЯЁ][а-яё]{2,})\s+([А-ЯЁ][а-яё]{2,})\b",
    # Общая форма из 2 слов
    r"\b([А-ЯЁ][а-яё]{2,})\s+([А-ЯЁ][а-яё]{2,})\b",
]


def _is_valid_fio(parts: list) -> bool:
    """Проверяет, что найденное сочетание похоже на ФИО."""
    if not parts:
        return False

    # Отсеиваем по стоп-словам
    for word in parts:
        if word.lower().rstrip(".,") in FIO_STOPWORDS:
            return False
        if len(word) > 20 or len(word) < 2:
            return False

    # Хотя бы одно слово должно быть похоже на фамилию
    for word in parts:
        w = word.lower().rstrip(".,")
        # Проверка по словарю популярных фамилий
        if w in COMMON_SURNAMES:
            return True
        # Проверка по типичным суффиксам фамилий
        if w.endswith(("ов", "ев", "ин", "ын", "ский", "цкий", "ко", "енко",
                        "ук", "юк", "ян", "янц", "дзе", "швили", "оглы",
                        "ова", "ева", "ина", "ына", "ская", "цкая")):
            return True

    # Если 3 слова, похожих на ФИО (все с большой буквы) — считаем валидным
    if len(parts) >= 3:
        return True

    return False


def extract_fio(text: str) -> str:
    """
    Умное извлечение ФИО:
    1. Ищем маркеры (студент, ФИО, выдан) и берём ФИО после них.
    2. Если не получилось — пробуем общие паттерны с фильтром.
    """
    # 1. Поиск по маркерам
    low = text.lower()
    for marker in FIO_MARKERS:
        idx = low.find(marker)
        if idx == -1:
            continue

        # Ищем ФИО в окне 250 символов после маркера
        window = text[idx: idx + 250]

        for pattern in FIO_PATTERNS:
            for m in re.finditer(pattern, window):
                groups = [g.strip() for g in m.groups() if g]
                if _is_valid_fio(groups):
                    return " ".join(groups)

    # 2. Поиск по всему тексту
    for pattern in FIO_PATTERNS:
        for m in re.finditer(pattern, text):
            groups = [g.strip() for g in m.groups() if g]
            if _is_valid_fio(groups):
                return " ".join(groups)

    return "не найдено"


def find_field(text: str, field: str) -> str:
    if field == "ФИО":
        return extract_fio(text)

    if "Дата" in field or "Период" in field or "Сроки" in field:
        dates = re.findall(DATE_PATTERN, text)
        return ", ".join(dates) if dates else "не найдено"

    if field == "Номер и дата выдачи":
        num = re.search(r"(?:№|номер)\s*([\w\-/]+)", text, re.IGNORECASE)
        date = re.search(DATE_PATTERN, text)
        if num and date:
            return f"№{num.group(1)} от {date.group(0)}"
        if num:
            return f"№{num.group(1)}"
        if date:
            return f"от {date.group(0)}"
        return "не найдено"

    if field == "Вуз":
        m = re.search(
            r"(?:университет|институт|академия|колледж)[^\n,.]*",
            text, re.IGNORECASE
        )
        return m.group(0).strip() if m else "не найдено"

    if field == "Организация":
        m = re.search(
            r"(?:ООО|АО|ПАО|ЗАО|ИП)\s+[«\"]?[А-ЯЁа-яёA-Za-z0-9\s\-]+[»\"]?",
            text
        )
        return m.group(0).strip() if m else "не найдено"

    if field == "Оценка":
        m = re.search(
            r"\b(отлично|хорошо|удовлетворительно|неудовлетворительно|[2-5])\b",
            text, re.IGNORECASE
        )
        return m.group(0) if m else "не найдено"

    if field == "Печать":
        low = text.lower()
        markers = ["м.п.", "место печати", "печать вуза", "печать организации",
                   "печать:", "печать —", "печать -", "круглая печать"]
        if any(m in low for m in markers):
            return "есть"
        if "печать" in low or "штамп" in low:
            return "есть"
        return "нет"

    if field == "Подпись":
        low = text.lower()
        markers = ["подпись:", "подпись —", "подпись -", "подписал", "подписано"]
        if any(m in low for m in markers):
            return "есть"
        return "не найдено"

    keywords = {
        "Направление": ["направление"],
        "Форма обучения": ["очная", "заочная", "очно-заочная"],
        "Причина отпуска": ["причина", "по состоянию здоровья", "семейные обстоятельства"],
        "Группа и курс": ["группа", "курс"],
        "Руководитель практики": ["руководитель"],
    }
    if field in keywords:
        for kw in keywords[field]:
            idx = text.lower().find(kw)
            if idx != -1:
                return text[idx:idx + 80].strip()
    return "не найдено"


def extract_fields(text: str, doc_type: str) -> dict:
    return {f: find_field(text, f) for f in DOCUMENT_TYPES[doc_type]}


# ==========================================================
# Логика ответа
# ==========================================================
def build_answer(fields: dict, question: str, best_rule: dict, intent: str) -> tuple:
    # Интенты с готовым ответом
    if intent and INTENTS[intent].get("answer_template"):
        template = INTENTS[intent]["answer_template"]
        source = f"п. {best_rule['number']} Положения" if best_rule else "Положение"
        return template, source

    missing = [f for f, v in fields.items() if v in ("не найдено", "нет")]

    # Печать
    if intent == "печать":
        if "Печать" not in fields:
            return (
                "ℹ️ Для этого типа документа печать не требуется по Положению.",
                "нет источника",
            )
        if fields.get("Печать") == "есть":
            return (
                "🟢 Печать в документе есть. Это соответствует п. 3.1 Положения.",
                "п. 3.1 Положения",
            )
        return (
            "🟡 Печать в документе отсутствует. "
            "Без печати документ недействителен (п. 3.6).",
            "п. 3.6 Положения",
        )

    # Подпись
    if intent == "подпись":
        if "Подпись" not in fields:
            return (
                "ℹ️ Для этого типа документа подпись не проверяется по Положению.",
                "нет источника",
            )
        if fields.get("Подпись") == "есть":
            return (
                "🟢 Подпись в документе есть. Это соответствует п. 3.1 Положения.",
                "п. 3.1 Положения",
            )
        return (
            "🟡 Подпись в документе отсутствует. "
            "Без подписи документ недействителен (п. 3.6).",
            "п. 3.6 Положения",
        )

    # ФИО
    if intent == "фио":
        fio = fields.get("ФИО", "не найдено")
        if fio != "не найдено":
            return f"🟢 ФИО указано: {fio}.", "поле документа"
        return "🟡 ФИО в документе не найдено.", "нет источника"

    # Даты
    if intent == "даты":
        dates = [
            v for k, v in fields.items()
            if ("Дата" in k or "Период" in k or "Сроки" in k) and v != "не найдено"
        ]
        if dates:
            return f"🟢 Даты в документе указаны: {'; '.join(dates)}.", "поле документа"
        return "🟡 Даты в документе не найдены.", "нет источника"

    # Вуз / организация
    if intent == "вуз_организация":
        vuz = fields.get("Вуз") or fields.get("Организация")
        if vuz and vuz != "не найдено":
            return f"🟢 Указано: {vuz}.", "поле документа"
        return "🟡 Название вуза или организации не найдено.", "нет источника"

    # Направление / форма
    if intent == "направление_форма":
        napr = fields.get("Направление")
        forma = fields.get("Форма обучения")
        if napr and napr != "не найдено":
            return f"🟢 Направление указано: {napr}.", "поле документа"
        if forma and forma != "не найдено":
            return f"🟢 Форма обучения указана: {forma}.", "поле документа"
        return "🟡 Ни направление, ни форма обучения не найдены.", "нет источника"

    # Причина отпуска
    if intent == "причина_отпуска":
        if "Причина отпуска" not in fields:
            return "ℹ️ Это поле не проверяется для данного типа документа.", "нет источника"
        prichina = fields.get("Причина отпуска")
        if prichina and prichina != "не найдено":
            return f"🟢 Причина отпуска указана: {prichina}.", "поле документа"
        return "🟡 Причина отпуска не найдена.", "нет источника"

    # Оценка
    if intent == "оценка":
        if "Оценка" not in fields:
            return "ℹ️ Это поле не проверяется для данного типа документа.", "нет источника"
        ocenka = fields.get("Оценка")
        if ocenka and ocenka != "не найдено":
            return f"🟢 Оценка указана: {ocenka}.", "поле документа"
        return "🟡 Оценка не найдена.", "нет источника"

    # Руководитель
    if intent == "руководитель":
        if "Руководитель практики" not in fields:
            return "ℹ️ Это поле не проверяется для данного типа документа.", "нет источника"
        ruk = fields.get("Руководитель практики")
        if ruk and ruk != "не найдено":
            return f"🟢 Руководитель практики указан: {ruk}.", "поле документа"
        return "🟡 Руководитель практики не найден.", "нет источника"

    # Вне темы
    if not best_rule or best_rule["score"] < 0.05:
        return (
            "🔴 В документе и в Положении нет информации по этому вопросу. "
            "Обратитесь в деканат.",
            "нет источника",
        )

    # Полнота
    if missing:
        return (
            f"🟡 Не хватает: {', '.join(missing)}. "
            f"Эти поля обязательны по п. {best_rule['number']} Положения.",
            f"п. {best_rule['number']} Положения",
        )

    return (
        f"🟢 Документ содержит все данные, которые требует "
        f"п. {best_rule['number']} Положения. "
        f"Окончательное решение примет сотрудник деканата.",
        f"п. {best_rule['number']} Положения",
    )


# ==========================================================
# Состояния
# ==========================================================
WAITING_PHOTO, WAITING_TYPE, WAITING_QUESTION = range(3)

BUTTON_SKIP = "✅ Проверить без вопроса"
BUTTON_AGAIN = "🔄 Проверить другой документ"
BUTTON_HELP = "❓ Помощь"
BUTTON_RESET = "🔄 Начать заново"


# ==========================================================
# Команды
# ==========================================================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()
    await update.message.reply_text(
        "🎓 Добро пожаловать в онлайн-деканат!\n\n"
        "Я — цифровой секретарь. Умею читать студенческие документы "
        "с фото и отвечать на вопросы по правилам вуза — со ссылкой "
        "на конкретный пункт Положения.\n\n"
        "Что я делаю:\n"
        "📸 Читаю фото справок и заявлений\n"
        "🔍 Сверяю с Положением\n"
        "💬 Говорю, чего не хватает\n"
        "🔗 Ссылаюсь на пункт правил\n\n"
        "Пришлите фото документа — начнём."
    )
    return WAITING_PHOTO


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "❓ *Как пользоваться ботом*\n\n"
        "1. Отправьте фото документа (JPG или PNG)\n"
        "2. Бот сам определит тип (или попросит выбрать)\n"
        "3. Можете написать вопрос *или нажать* "
        "«✅ Проверить без вопроса»\n"
        "4. Получите ответ со ссылкой на Положение\n\n"
        "*Что можно спросить (примеры):*\n\n"
        "📄 *Про перевод:*\n"
        "• Хватит ли этой справки для перевода?\n"
        "• Можно ли перевестись в другой вуз?\n"
        "• Что нужно для перевода?\n\n"
        "🔍 *Про полноту документа:*\n"
        "• Всё ли в порядке?\n"
        "• Чего не хватает в документе?\n"
        "• Какие поля отсутствуют?\n"
        "• Есть ли ошибки?\n\n"
        "📌 *Про конкретные поля:*\n"
        "• Есть ли печать?\n"
        "• Есть ли подпись?\n"
        "• Указано ли ФИО?\n"
        "• Есть ли даты?\n"
        "• Указан ли вуз?\n"
        "• Есть ли направление?\n"
        "• Какая форма обучения?\n\n"
        "🎓 *Про практику:*\n"
        "• Указан ли руководитель?\n"
        "• Есть ли оценка?\n"
        "• Указаны ли сроки практики?\n\n"
        "📝 *Про отпуск:*\n"
        "• Указана ли причина отпуска?\n"
        "• Есть ли даты начала и окончания?\n\n"
        "👤 *Общие вопросы:*\n"
        "• Кто принимает решение?\n"
        "• Что делать, если не хватает полей?",
        parse_mode="Markdown",
    )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()
    await update.message.reply_text(
        "🔄 Сброшено. Пришлите новое фото документа."
    )
    return WAITING_PHOTO


# ==========================================================
# Фото
# ==========================================================
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        photo_file = await update.message.photo[-1].get_file()
    except Exception as e:
        logger.error(f"Photo download error: {e}")
        await update.message.reply_text(
            "Не удалось загрузить фото. Попробуйте ещё раз."
        )
        return WAITING_PHOTO

    path = f"/tmp/{update.message.chat_id}.jpg"
    await photo_file.download_to_drive(path)
    context.user_data["image_path"] = path

    await update.message.reply_text("🔍 Определяю тип документа...")
    text = recognize_text(path)

    if not text.strip():
        await update.message.reply_text(
            "⚠️ Не удалось распознать текст на фото.\n"
            "Попробуйте более чёткое фото при хорошем освещении."
        )
        return WAITING_PHOTO

    auto_type = detect_doc_type(text)

    kb_skip = ReplyKeyboardMarkup(
        [[KeyboardButton(BUTTON_SKIP)]],
        resize_keyboard=True,
        one_time_keyboard=True,
    )

    if auto_type:
        context.user_data["doc_type"] = auto_type
        await update.message.reply_text(
            f"📄 Похоже, это *{auto_type}*\n\n"
            "Напишите ваш вопрос — или нажмите кнопку ниже, "
            "чтобы проверить документ сразу.\n\n"
            "Примеры вопросов:\n"
            "• Всё ли в порядке?\n"
            "• Чего не хватает?\n"
            "• Есть ли печать?\n"
            "• Есть ли подпись?\n"
            "• Хватит ли для перевода?",
            parse_mode="Markdown",
            reply_markup=kb_skip,
        )
        return WAITING_QUESTION
    else:
        keyboard = [[name] for name in DOCUMENT_TYPES.keys()]
        await update.message.reply_text(
            "📄 Фото получено. Не смог определить тип автоматически.\n"
            "Выберите тип документа:",
            reply_markup=ReplyKeyboardMarkup(
                keyboard, resize_keyboard=True, one_time_keyboard=True
            ),
        )
        return WAITING_TYPE


async def handle_type(update: Update, context: ContextTypes.DEFAULT_TYPE):
    doc_type = update.message.text
    if doc_type not in DOCUMENT_TYPES:
        await update.message.reply_text("Пожалуйста, выберите тип из кнопок.")
        return WAITING_TYPE

    context.user_data["doc_type"] = doc_type
    kb_skip = ReplyKeyboardMarkup(
        [[KeyboardButton(BUTTON_SKIP)]],
        resize_keyboard=True,
        one_time_keyboard=True,
    )
    await update.message.reply_text(
        f"Тип: *{doc_type}*\n\n"
        "Напишите ваш вопрос — или нажмите кнопку ниже, "
        "чтобы проверить документ сразу.\n\n"
        "Примеры вопросов:\n"
        "• Всё ли в порядке?\n"
        "• Чего не хватает?\n"
        "• Есть ли печать?",
        parse_mode="Markdown",
        reply_markup=kb_skip,
    )
    return WAITING_QUESTION


# ==========================================================
# Вопрос → ответ
# ==========================================================
async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text_input = update.message.text

    if text_input == BUTTON_SKIP:
        question = "Всё ли в порядке с этим документом? Чего не хватает?"
    else:
        question = text_input

    image_path = context.user_data.get("image_path")
    doc_type = context.user_data.get("doc_type")

    if not image_path or not doc_type:
        await update.message.reply_text(
            "Что-то пошло не так. Начните заново: /start"
        )
        return WAITING_PHOTO

    await update.message.reply_text("⏳ Читаю документ, ищу ответ...")

    text = recognize_text(image_path)
    if not text.strip():
        await update.message.reply_text(
            "Не удалось распознать текст. Попробуйте более чёткое фото."
        )
        return WAITING_PHOTO

    fields = extract_fields(text, doc_type)
    intent = detect_intent(question)

    query = question + " " + " ".join(str(v) for v in fields.values())
    candidates = search_rules(query, top_k=3)
    best = candidates[0] if candidates else None

    answer_text, source = build_answer(fields, question, best, intent)

    found = [
        f"✅ {k}: {v}"
        for k, v in fields.items()
        if v not in ("не найдено", "нет")
    ]
    missing = [
        f"❌ {k}: не найдено"
        for k, v in fields.items()
        if v in ("не найдено", "нет")
    ]

    fields_str = ""
    if found:
        fields_str += "*Найдено:*\n" + "\n".join(found)
    if missing:
        fields_str += "\n\n*Не найдено:*\n" + "\n".join(missing)

    response = (
        f"📋 *Карточка полей:*\n{fields_str}\n\n"
        f"💬 *Ответ:*\n{answer_text}\n\n"
        f"🔗 *Источник:* {source}"
    )

    try:
        await update.message.reply_text(response, parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Send error: {e}")
        await update.message.reply_text(response)

    kb = ReplyKeyboardMarkup(
        [[KeyboardButton(BUTTON_AGAIN)]],
        resize_keyboard=True,
    )
    await update.message.reply_text(
        "Можете прислать следующий документ или нажать кнопку ниже.",
        reply_markup=kb,
    )
    return WAITING_PHOTO


async def handle_text_without_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()

    if text == BUTTON_HELP:
        await help_command(update, context)
        return WAITING_PHOTO

    if text == BUTTON_RESET:
        await reset(update, context)
        return WAITING_PHOTO

    await update.message.reply_text(
        "📄 Чтобы я мог проверить документ, сначала пришлите его фото.\n\n"
        "Отправьте фото справки, заявления или справки о практике."
    )
    return WAITING_PHOTO


# ==========================================================
# Запуск
# ==========================================================
def main():
    app = Application.builder().token(TOKEN).build()

    conv = ConversationHandler(
        entry_points=[
            CommandHandler("start", start),
            MessageHandler(filters.PHOTO, handle_photo),
            MessageHandler(filters.Regex(f"^{BUTTON_AGAIN}$"), reset),
        ],
        states={
            WAITING_PHOTO: [
                MessageHandler(filters.PHOTO, handle_photo),
                MessageHandler(filters.Regex(f"^{BUTTON_AGAIN}$"), reset),
                MessageHandler(
                    filters.TEXT & ~filters.COMMAND, handle_text_without_photo
                ),
            ],
            WAITING_TYPE: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_type),
            ],
            WAITING_QUESTION: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_question),
            ],
        },
        fallbacks=[
            CommandHandler("reset", reset),
            CommandHandler("help", help_command),
            CommandHandler("start", start),
        ],
    )

    app.add_handler(conv)
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("reset", reset))

    logger.info("Бот запущен!")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
