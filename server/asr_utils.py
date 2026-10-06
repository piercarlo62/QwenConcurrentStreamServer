# ASR Concurrent Stream Server v1.0.6
SAMPLE_RATE = 16000
MAX_ASR_INPUT_SECONDS = 1200
MAX_FORCE_ALIGN_INPUT_SECONDS = 180
MIN_ASR_INPUT_SECONDS = 0.5
SUPPORTED_LANGUAGES = [
    "Chinese", "English", "Cantonese", "Arabic", "German", "French",
    "Spanish", "Portuguese", "Indonesian", "Italian", "Korean",
    "Russian", "Thai", "Vietnamese", "Japanese", "Turkish", "Hindi",
    "Malay", "Dutch", "Swedish", "Danish", "Finnish", "Polish",
    "Czech", "Filipino", "Persian", "Greek", "Romanian", "Hungarian",
    "Macedonian",
]
_ASR_TEXT_TAG = "<asr_text>"
_LANG_PREFIX = "language "


def normalize_language_name(language: str) -> str:
    if language is None:
        raise ValueError("language is None")
    s = str(language).strip()
    if not s:
        raise ValueError("language is empty")
    return s[:1].upper() + s[1:].lower()


def validate_language(language: str) -> None:
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language: {language}. Supported: {SUPPORTED_LANGUAGES}")


def detect_and_fix_repetitions(text, threshold=20):
    def fix_char_repeats(s, thresh):
        res = []
        i = 0
        n = len(s)
        while i < n:
            count = 1
            while i + count < n and s[i + count] == s[i]:
                count += 1
            if count > thresh:
                res.append(s[i])
                i += count
            else:
                res.append(s[i:i+count])
                i += count
        return ''.join(res)

    def fix_pattern_repeats(s, thresh, max_len=20):
        n = len(s)
        min_repeat_chars = thresh * 2
        if n < min_repeat_chars:
            return s
        i = 0
        result = []
        while i <= n - min_repeat_chars:
            found = False
            for k in range(1, max_len + 1):
                if i + k * thresh > n:
                    break
                pattern = s[i:i+k]
                valid = True
                for rep in range(1, thresh):
                    start_idx = i + rep * k
                    if s[start_idx:start_idx+k] != pattern:
                        valid = False
                        break
                if valid:
                    total_rep = thresh
                    end_index = i + thresh * k
                    while end_index + k <= n and s[end_index:end_index+k] == pattern:
                        total_rep += 1
                        end_index += k
                    result.append(pattern)
                    result.append(fix_pattern_repeats(s[end_index:], thresh, max_len))
                    i = n
                    found = True
                    break
            if found:
                break
            else:
                result.append(s[i])
                i += 1
        if not found:
            result.append(s[i:])
        return ''.join(result)

    text_raw = text
    text = fix_char_repeats(text_raw, threshold)
    text = fix_pattern_repeats(text, threshold)
    return text


def parse_asr_output(raw: str, user_language: str = None):
    if raw is None:
        return "", ""
    s = str(raw).strip()
    if not s:
        return "", ""

    s = detect_and_fix_repetitions(s)

    if user_language:
        return user_language, s

    meta_part = s
    text_part = ""
    has_tag = _ASR_TEXT_TAG in s
    if has_tag:
        meta_part, text_part = s.split(_ASR_TEXT_TAG, 1)
    else:
        return "", s.strip()

    meta_lower = meta_part.lower()

    if "language none" in meta_lower:
        t = text_part.strip()
        if not t:
            return "", ""
        return "", t

    lang = ""
    for line in meta_part.splitlines():
        line = line.strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith(_LANG_PREFIX):
            val = line[len(_LANG_PREFIX):].strip()
            if val:
                lang = normalize_language_name(val)
            break

    return lang, text_part.strip()
