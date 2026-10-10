import os
import re
import logging
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    filters, ContextTypes, ConversationHandler
)
import pytesseract
from PIL import Image
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================================
# Настройка
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
# Положение вуза (правила)
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


def search_rules(query, top_k=3):
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
# Типы документов и их поля
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
# OCR через Tesseract
# ==========================================================
def recognize_text(image_path: str) -> str:
    """Распознаёт текст с картинки через Tesseract."""
    try:
        img = Image.open(image_path)
        # Предобработка: серый + увеличение контраста
        img = img.convert("L")
        text = pytesseract.image_to_string(img, lang="rus+eng")
        return text
    except Exception as e:
        logger.error(f"OCR error: {e}")
        return ""


# ==========================================================
# Извлечение полей
# ==========================================================
DATE_PATTERN = r"\b\d{2}[./-]\d{2}[./-]\d{2,4}\b"
FIO_PATTERN = r"[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)?"


def find_field(text: str, field: str) -> str:
    if field == "ФИО":
        m = re.search(FIO_PATTERN, text)
        return m.group(0) if m else "не найдено"

    if "Дата" in field or "Период" in field or "Сроки" in field:
        dates = re.findall(DATE_PATTERN, text)
        return ", ".join(dates) if dates else "не найдено"

    if field == "Номер и дата выдачи":
        num = re.search(r"(?:№|номер)\s*([\w\-/]+)", text, re.IGNORECASE)
        date = re.search(DATE_PATTERN, text)
        if num and date:
            return f"№{num.group(1)} от {date.group(0)}"
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

    if field in ("Печать", "Подпись"):
        low = text.lower()
        return "есть" if ("печать" in low or "подпись" in low) else "нет"

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
def build_answer(fields: dict, question: str, best_rule: dict) -> tuple:
    if not best_rule or best_rule["score"] < 0.05:
        return (
            "🔴 В документе и в Положении нет информации по этому вопросу. "
            "Обратитесь в деканат.",
            "нет источника"
        )

    missing = [f for f, v in fields.items() if v in ("не найдено", "нет")]

    if missing:
        return (
            f"🟡 Не хватает: {', '.join(missing)}. "
            f"Эти поля обязательны по п. {best_rule['number']} Положения.",
            f"п. {best_rule['number']} Положения"
        )

    return (
        f"🟢 Документ содержит все данные, которые требует "
        f"п. {best_rule['number']} Положения. "
        f"Окончательное решение примет сотрудник деканата.",
        f"п. {best_rule['number']} Положения"
    )


# ==========================================================
# Состояния диалога
# ==========================================================
WAITING_PHOTO, WAITING_TYPE, WAITING_QUESTION = range(3)


# ==========================================================
# Команды
# ==========================================================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()
    await update.message.reply_text(
        "👋 Здравствуйте! Я цифровой секретарь деканата.\n\n"
        "Пришлите фото студенческого документа (JPG или PNG)."
    )
    return WAITING_PHOTO


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Как пользоваться:\n"
        "1. Отправьте фото документа\n"
        "2. Выберите тип\n"
        "3. Напишите вопрос\n"
        "4. Получите ответ со ссылкой на Положение\n\n"
        "Сбросить: /reset"
    )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()
    await update.message.reply_text("Сброшено. Пришлите новое фото документа.")
    return WAITING_PHOTO


# ==========================================================
# Приём фото
# ==========================================================
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        photo_file = await update.message.photo[-1].get_file()
    except Exception as e:
        logger.error(f"Photo download error: {e}")
        await update.message.reply_text("Не удалось загрузить фото. Попробуйте ещё раз.")
        return WAITING_PHOTO

    path = f"/tmp/{update.message.chat_id}.jpg"
    await photo_file.download_to_drive(path)
    context.user_data["image_path"] = path

    # Клавиатура с типами документов
    keyboard = [[name] for name in DOCUMENT_TYPES.keys()]
    await update.message.reply_text(
        "📄 Фото получено. Выберите тип документа:",
        reply_markup=ReplyKeyboardMarkup(
            keyboard, resize_keyboard=True, one_time_keyboard=True
        )
    )
    return WAITING_TYPE


# ==========================================================
# Выбор типа
# ==========================================================
async def handle_type(update: Update, context: ContextTypes.DEFAULT_TYPE):
    doc_type = update.message.text
    if doc_type not in DOCUMENT_TYPES:
        await update.message.reply_text(
            "Пожалуйста, выберите тип из кнопок."
        )
        return WAITING_TYPE

    context.user_data["doc_type"] = doc_type
    await update.message.reply_text(
        f"Тип: *{doc_type}*\n\nТеперь напишите ваш вопрос:",
        parse_mode="Markdown"
    )
    return WAITING_QUESTION


# ==========================================================
# Приём вопроса и ответ
# ==========================================================
async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    question = update.message.text
    image_path = context.user_data.get("image_path")
    doc_type = context.user_data.get("doc_type")

    if not image_path or not doc_type:
        await update.message.reply_text(
            "Что-то пошло не так. Начните заново: /start"
        )
        return WAITING_PHOTO

    await update.message.reply_text("⏳ Читаю документ, ищу ответ...")

    # OCR
    text = recognize_text(image_path)

    if not text.strip():
        await update.message.reply_text(
            "Не удалось распознать текст. Попробуйте более чёткое фото."
        )
        return WAITING_PHOTO

    # Извлечение полей
    fields = extract_fields(text, doc_type)

    # Поиск по Положению
    query = question + " " + " ".join(str(v) for v in fields.values())
    candidates = search_rules(query, top_k=1)
    best = candidates[0] if candidates else None

    # Ответ
    answer_text, source = build_answer(fields, question, best)

    fields_str = "\n".join([f"• {k}: {v}" for k, v in fields.items()])

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

    await update.message.reply_text(
        "Пришлите следующее фото или /start для нового диалога."
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
        ],
        states={
            WAITING_PHOTO: [
                MessageHandler(filters.PHOTO, handle_photo),
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
        ],
    )

    app.add_handler(conv)
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("reset", reset))

    logger.info("Бот запущен!")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
