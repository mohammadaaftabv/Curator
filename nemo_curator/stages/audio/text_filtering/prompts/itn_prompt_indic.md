# Inverse Text Normalization ({language})

Convert the spoken-form transcript below in **{language}** to standard written form. Only apply conversions supported by the active-language rules: numbers, dates, times, currencies, measurements, symbols, and related structured forms become their conventional written representations. Use **{language}** conventions; the active-language examples define the pattern for this row.

Return ONLY the converted text. No explanations, labels, or extra formatting.

## Constraints

- Stay in **{language}** and preserve any code-switching or script mixing. Do NOT translate.
- Preserve all non-target wording and its order, including disfluencies, repetitions, false starts, colloquial forms, mispronunciations, and grammatical errors.
- Do NOT add punctuation unless it is implied by the input or required by an allowed written-form conversion.
- Do NOT paraphrase, add, or remove words beyond the conversions defined below.
- When a number expression functions idiomatically rather than numerically, keep it in word form.

## Conversion Rules

{language_rules}

Additional rules:
- Convert the active language's spoken structural forms to `.`, `@`, `/`, `:`, or `-` only where their symbolic use is implied. Follow the active-language rules for the listed spoken forms.
- Render recognized acronyms and structured letter-number forms shown by the active-language examples in their conventional uppercase written form; do not expand them or convert unrelated letter-name sequences.
- Convert the ordinary spoken form of zero to `0`. In phone and time contexts, also convert alternate zero readings licensed by the active-language rules to `0`.
- Interpret postal or ZIP codes digit by digit and write the resulting code as digits. Write house numbers with digits.

## Ambiguity Resolution

- Prefer digits for quantities, measurements, ages, dates, and counts.
- Keep number expressions in word form when they are idiomatic, part of a proper noun, pronominal or indefinite, or a vague quantity.
- Convert an expression denoting one quarter to `1/4` only when it functions as a true fraction; leave it in word form in temporal or financial constructions.
- Convert an expression denoting one half to `1/2` only when it functions as a true fraction; leave idiomatic uses in word form.
- In stammers and false starts, preserve the broken number-word fragments and convert only the final clean numeric expression.

Spoken-form transcript in {language}:
{text}
