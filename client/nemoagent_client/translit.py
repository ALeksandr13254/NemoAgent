"""Latin words inside Russian speech -> Cyrillic, the way a Russian speaker would say them.

TeraTTSv2 reads an <en> span with the Russian voice badly ("futuristic" comes out as noise), and the
model does not always follow the "transliterate brands" rule. So before synthesis every Latin word
in a predominantly Russian sentence is rewritten: a dictionary of common tech/brand words first,
all-caps abbreviations are spelled with English letter names, everything else goes through
approximate English reading rules. Whole-English sentences are left alone (English voice).
"""
from __future__ import annotations

import re

WORDS: dict[str, str] = {
    # brands / products
    "windows": "виндоус", "microsoft": "майкрософт", "google": "гугл", "chrome": "хром", "youtube": "ютуб",
    "github": "гитхаб", "gitlab": "гитлаб", "git": "гит", "python": "пайтон", "java": "джава",
    "javascript": "джаваскрипт", "typescript": "тайпскрипт", "node": "ноуд", "react": "реакт", "vue": "вью",
    "linux": "линукс", "ubuntu": "убунту", "android": "андроид", "iphone": "айфон", "ipad": "айпэд", "apple": "эпл",
    "nvidia": "энвидиа", "intel": "интел", "amd": "а-эм-дэ", "discord": "дискорд", "telegram": "телеграм",
    "whatsapp": "вотсап", "steam": "стим", "unity": "юнити", "unreal": "анриал", "excel": "эксель", "word": "ворд",
    "powershell": "пауэршелл", "bash": "баш", "docker": "докер", "visual": "визуал", "studio": "студио",
    "code": "код", "vscode": "ви-эс-код", "firefox": "файрфокс", "edge": "эдж", "opera": "опера", "yandex": "яндекс",
    "vk": "вэ-ка", "twitch": "твич", "netflix": "нетфликс", "spotify": "спотифай", "zoom": "зум", "skype": "скайп",
    "notion": "ноушен", "obsidian": "обсидиан", "figma": "фигма", "photoshop": "фотошоп", "blender": "блендер",
    "minecraft": "майнкрафт", "roblox": "роблокс", "fortnite": "фортнайт", "dota": "дота", "counter": "каунтер",
    "strike": "страйк", "cyberpunk": "киберпанк", "witcher": "витчер", "elden": "элден", "ring": "ринг",
    "genshin": "геншин", "impact": "импакт", "playstation": "плейстейшн", "xbox": "иксбокс", "nintendo": "нинтендо",
    "switch": "свитч", "wifi": "вайфай", "bluetooth": "блютус", "usb": "ю-эс-би", "hdmi": "эйч-ди-эм-ай",
    "gpu": "джи-пи-ю", "cpu": "си-пи-ю", "ram": "рам", "ssd": "эс-эс-ди", "hdd": "эйч-ди-ди", "api": "эй-пи-ай",
    "url": "ю-ар-эл", "json": "джейсон", "html": "эйч-ти-эм-эл", "css": "си-эс-эс", "sql": "эс-кью-эл",
    "pdf": "пэ-дэ-эф", "png": "пэ-эн-гэ", "jpg": "джейпег", "jpeg": "джейпег", "mp3": "эм-пэ-три", "mp4": "эм-пэ-четыре",
    "exe": "экзе", "dll": "дэ-эл-эл", "zip": "зип", "rar": "рар", "txt": "тэ-экс-тэ", "docx": "докс", "xlsx": "эксель",
    "ai": "эй-ай", "gpt": "джи-пи-ти", "chatgpt": "чат-джи-пи-ти", "openai": "оупен-эй-ай", "claude": "клод",
    "nemotron": "немотрон", "deepseek": "дипсик", "llama": "лама", "gemini": "джемини", "whisper": "виспер",
    "cuda": "куда", "onnx": "оникс", "torch": "торч", "pytorch": "пайторч", "tensorflow": "тензорфлоу",
    # common words that slip into Russian answers
    "futuristic": "футуристик", "design": "дизайн", "gameplay": "геймплей", "game": "гейм", "anime": "аниме",
    "machine": "машин", "cyber": "сайбер", "table": "тейбл", "power": "пауэр", "manager": "менеджер",
    "station": "стейшн", "computer": "компьютер", "player": "плеер", "controller": "контроллер",
    "keyboard": "кибоард", "mouse": "маус", "monitor": "монитор", "camera": "камера", "speaker": "спикер",
    "micro": "майкро", "phone": "фон", "smart": "смарт", "watch": "вотч", "music": "мьюзик", "movie": "муви",
    "series": "сириез", "season": "сизон", "episode": "эпизод", "trailer": "трейлер", "review": "ревью",
    "online": "онлайн", "offline": "офлайн", "update": "апдейт", "upgrade": "апгрейд", "file": "файл", "folder": "фолдер",
    "desktop": "десктоп", "download": "даунлоуд", "upload": "аплоуд", "browser": "браузер", "server": "сервер",
    "client": "клиент", "internet": "интернет", "email": "имейл", "mail": "мейл", "login": "логин", "user": "юзер",
    "admin": "админ", "password": "пассворд", "ok": "окей", "okay": "окей", "yes": "йес", "no": "ноу", "hello": "хеллоу",
    "hi": "хай", "thanks": "сэнкс", "sorry": "сорри", "please": "плиз", "wow": "вау", "cool": "кул", "nice": "найс",
    "super": "супер", "level": "левел", "boss": "босс", "quest": "квест", "skill": "скилл", "item": "айтем",
    "mode": "мод", "default": "дефолт", "cloud": "клауд", "drive": "драйв", "home": "хоум", "office": "офис",
    "team": "тим", "engine": "энджин", "script": "скрипт", "plugin": "плагин", "mod": "мод", "patch": "патч",
    "release": "релиз", "version": "вёршн", "bug": "баг", "fix": "фикс", "feature": "фича", "task": "таск",
    "deadline": "дедлайн", "meeting": "митинг", "call": "колл", "chat": "чат", "voice": "войс", "sound": "саунд",
    "video": "видео", "photo": "фото", "screenshot": "скриншот", "screen": "скрин", "text": "текст", "print": "принт",
    "start": "старт", "stop": "стоп", "run": "ран", "open": "оупен", "close": "клоуз", "save": "сейв", "load": "лоуд",
    "new": "нью", "old": "олд", "big": "биг", "small": "смолл", "fast": "фаст", "slow": "слоу", "light": "лайт",
    "dark": "дарк", "pro": "про", "max": "макс", "mini": "мини", "plus": "плюс", "ultra": "ультра", "air": "эйр",
    "flash": "флэш", "the": "зе", "of": "оф", "and": "энд", "in": "ин", "on": "он", "for": "фор", "with": "виз",
    "app": "апп", "apps": "аппс", "tool": "тул", "tools": "тулз", "bot": "бот", "web": "веб", "site": "сайт",
    "link": "линк", "page": "пейдж", "search": "сёрч", "share": "шер", "like": "лайк", "post": "пост", "feed": "фид",
    "story": "стори", "stream": "стрим", "live": "лайв", "tv": "ти-ви", "pc": "пи-си", "id": "ай-ди", "vip": "вип",
}

LETTERS = {"a": "эй", "b": "би", "c": "си", "d": "ди", "e": "и", "f": "эф", "g": "джи", "h": "эйч", "i": "ай",
           "j": "джей", "k": "кей", "l": "эл", "m": "эм", "n": "эн", "o": "оу", "p": "пи", "q": "кью", "r": "ар",
           "s": "эс", "t": "ти", "u": "ю", "v": "ви", "w": "дабл-ю", "x": "экс", "y": "уай", "z": "зед"}

# (pattern, replacement) applied left to right on the lowercase word; longer patterns first
_RULES: list[tuple[str, str]] = [
    ("tion", "шн"), ("sion", "жн"), ("ough", "о"), ("augh", "о"), ("igh", "ай"), ("tch", "ч"), ("sch", "ск"),
    ("ch", "ч"), ("sh", "ш"), ("th", "т"), ("ph", "ф"), ("wh", "в"), ("ck", "к"), ("qu", "кв"), ("gh", "г"),
    ("ee", "и"), ("ea", "и"), ("oo", "у"), ("ou", "ау"), ("ow", "оу"), ("ay", "эй"), ("ai", "эй"), ("ei", "ей"),
    ("ey", "ей"), ("oy", "ой"), ("oi", "ой"), ("oa", "оу"), ("au", "о"), ("aw", "о"), ("ew", "ью"), ("ue", "ью"),
    ("ie", "и"), ("ur", "ёр"), ("ir", "ёр"), ("er", "ер"), ("ar", "ар"), ("or", "ор"),
    ("ng", "нг"), ("ll", "лл"), ("ss", "сс"), ("tt", "тт"), ("pp", "пп"), ("nn", "нн"), ("mm", "мм"), ("rr", "рр"),
    ("dd", "дд"), ("bb", "бб"), ("gg", "гг"), ("ff", "фф"), ("zz", "зз"),
]
_SINGLE = {"a": "а", "b": "б", "c": "к", "d": "д", "e": "е", "f": "ф", "g": "г", "h": "х", "i": "и", "j": "дж",
           "k": "к", "l": "л", "m": "м", "n": "н", "o": "о", "p": "п", "q": "к", "r": "р", "s": "с", "t": "т",
           "u": "у", "v": "в", "w": "в", "x": "кс", "y": "и", "z": "з"}
_VOWELS = set("aeiouy")


def _rules(word: str) -> str:
    w = word.lower()
    # magic e: "game" -> "гейм", "site" -> "сайт", "cute" -> "кьют" (vowel must stand alone: not "house")
    if (len(w) >= 4 and w.endswith("e") and w[-2] not in _VOWELS and w[-3] in "aiu"
            and (len(w) == 4 or w[-4] not in _VOWELS)):
        core = {"a": "эй", "i": "ай", "u": "ю"}[w[-3]]
        w = w[:-3] + "\x00" + w[-2]          # \x00 = placeholder for the magic vowel
        magic = core
    else:
        magic = ""
    if w.endswith("e") and len(w) > 3 and w[-2] not in _VOWELS and not magic:
        w = w[:-1]                            # silent final e: "house" -> "хаус"
    if w.endswith("y") and len(w) > 1 and not any(c in _VOWELS for c in w[:-1]):
        w = w[:-1] + "\x01"                   # one-syllable "sky", "fly" -> "ай"
    out: list[str] = []
    i = 0
    while i < len(w):
        ch = w[i]
        if ch == "\x00":
            out.append(magic)
            i += 1
            continue
        if ch == "\x01":
            out.append("ай")
            i += 1
            continue
        matched = False
        for pat, rep in _RULES:
            if w.startswith(pat, i):
                out.append(rep)
                i += len(pat)
                matched = True
                break
        if matched:
            continue
        nxt = w[i + 1] if i + 1 < len(w) else ""
        if ch == "c":
            out.append("с" if nxt in "eiy" else "к")
        elif ch == "g":
            out.append("дж" if nxt in "eiy" and i > 0 else "г")
        elif ch == "y":
            out.append("й" if nxt in _VOWELS else "и")
        elif ch == "e" and i == 0:
            out.append("э")
        elif ch == "u" and i == 0:
            out.append("а")
        elif ch == "w" and nxt in _VOWELS:
            out.append("у")
        else:
            out.append(_SINGLE.get(ch, ch))
        i += 1
    return "".join(out)


def transliterate_word(word: str) -> str:
    low = word.lower()
    if low in WORDS:
        rep = WORDS[low]
    elif low.endswith("s") and low[:-1] in WORDS:            # plural: downloads, apps
        rep = WORDS[low[:-1]] + "с"
    elif word.isupper() and len(word) <= 5:
        rep = "-".join(LETTERS.get(c, c) for c in low)      # GTA -> джи-ти-эй
    elif len(word) == 1:
        rep = LETTERS.get(low, word)
    else:
        rep = _rules(word)
    if word[:1].isupper() and rep:
        rep = rep[0].upper() + rep[1:]
    return rep


_LATIN_WORD = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")
_CYR = re.compile(r"[Ѐ-ӿ]")
_LAT = re.compile(r"[A-Za-z]")
_CYR_WORD = re.compile(r"[Ѐ-ӿ]{2,}")


def is_russian_context(text: str) -> bool:
    """A Russian sentence with English names in it is still Russian; only a sentence with (almost)
    no Cyrillic is English. Used both for transliteration and for the voice choice."""
    ru = len(_CYR.findall(text))
    en = len(_LAT.findall(text))
    if ru == 0:
        return False
    if en == 0:
        return True
    return len(_CYR_WORD.findall(text)) >= 2 or ru >= 0.3 * (ru + en)


def transliterate_latin(text: str, force: bool = False) -> str:
    """Rewrite Latin words in Cyrillic when the text is a Russian sentence (or force=True — the
    'read everything in Russian' setting); untouched otherwise."""
    if not _LAT.search(text) or not (force or is_russian_context(text)):
        return text
    return _LATIN_WORD.sub(lambda m: transliterate_word(m.group(0)), text)
