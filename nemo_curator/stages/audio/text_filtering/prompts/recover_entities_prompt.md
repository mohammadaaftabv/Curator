# SYSTEM_PROMPT

You are given TWO transcriptions of the same audio clip:

- INITIAL: Initial reference with named entity spelling/casing.
- NORMALIZED: Normalized transcription.

Your task:

1. In the INITIAL text, identify ONLY proper-noun named entities made of letters/words only (no digits): people, places, organizations, companies, products, brands, and acronyms/initialisms. 

2. If a candidate contains any digit (0-9), it is NOT an entity and must be excluded.

3. Titles/honorifics are NOT entities and must be excluded.

4. For each remaining entity, locate its match in the NORMALIZED text and substitute it with the INITIAL spelling/casing. Change ONLY the entity itself — leave all normalization untouched: keep numbers, dates, times, money, units, and titles in their written form.


Return ONLY the resulting NORMALIZED text. No explanations, labels, or extra formatting.

# USER_PROMPT_TEMPLATE

INITIAL:
"{ground_truth}"

NORMALIZED:
"{normalized}"

