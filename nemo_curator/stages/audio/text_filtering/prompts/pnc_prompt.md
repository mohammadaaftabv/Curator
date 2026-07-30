Add punctuation and permitted capitalization to the transcript. This is the only change you may make.

ACTIVE-LANGUAGE OUTPUT CONTRACT
{language_rules}

RULES
- End every complete statement or command with the specified statement terminal and every direct question with the specified question mark. Punctuate each clearly complete sentence.
- Add internal punctuation only at a high-confidence grammatical boundary. Do not add a comma merely for coordination, a conjunction, or a possible pause; when uncertain, add no internal mark.
- Copy every input Unicode code point in order, including letters, digits, combining marks, existing punctuation and symbols, repetitions, errors, and invisible format/direction controls. Never add, delete, replace, normalize, transliterate, correct, or reorder content. Never reinterpret an input character as a punctuation placeholder.
- Change case only on existing cased letters at a clear sentence start or in a high-confidence proper noun or acronym. Do not insert quotation marks, brackets, parentheses, or dashes.

TRANSCRIPT
<transcript>
{text}
</transcript>

Return only the punctuated transcript, without a label or explanation.
