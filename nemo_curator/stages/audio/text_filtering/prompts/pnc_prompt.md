Add punctuation and permitted capitalization to the transcript. This is the only change you may make.

ACTIVE-LANGUAGE OUTPUT CONTRACT
{language_rules}

BOUNDARY RULES
- Read the whole transcript first. Use syntax and meaning, not text length or a possible acoustic pause.
- End every complete statement or command with the specified statement terminal. End a grammatically complete final span. Punctuate each clearly independent complete sentence.
- Use the specified question mark only when the utterance itself asks the listener a direct question. A question word, reported question, or phrase describing a question is not enough.
- Add a comma only at an unambiguous grammatical clause, list, or address boundary. Do not comma every coordinated word, repeated suffix, conjunction, or possible pause; when uncertain, add no internal mark.
- Prefer a sentence terminal to a semicolon between independent sentences. Use other mapped marks only when their function is unambiguous.

PRESERVATION
- Copy every input Unicode code point in order, including letters, digits, combining marks, existing punctuation and symbols, repetitions, errors, and invisible format/direction controls. Never add, delete, replace, normalize, transliterate, correct, or reorder content. Never reinterpret an input character as a punctuation placeholder.
- Change case only on existing cased letters at a clear sentence start or in a high-confidence proper noun or acronym. Do not insert quotation marks, brackets, parentheses, or dashes.
- Silently verify that removing only inserted punctuation and undoing only permitted case changes reconstructs the exact input code-point sequence. Repair any mismatch, but still restore every required terminal.

TRANSCRIPT
<transcript>
{text}
</transcript>

Return only the verified punctuated transcript, without a label or explanation.
