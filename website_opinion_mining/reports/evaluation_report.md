# Evaluation report

Evaluation file: `eval_comments.csv` - **200 comments**, 402 gold (aspect, sentiment) pairs, 80 mixed-sentiment comments.

> NOTE: Transformer rows skipped: `pip install transformers torch` and re-run to include them.

> CAVEAT: the default evaluation set is synthetic (built from held-out clauses with exact labels), so scores are optimistic. Use `--csv` with hand-labelled REAL comments for final numbers.

## A. Aspect detection (comment level)

| Pipeline | Precision | Recall | micro-F1 [95% CI] | macro-F1 |
|---|---|---|---|---|
| V1  keyword + VADER | 0.917 | 0.821 | 0.866 [0.843, 0.889] | 0.857 |
| V2a ML aspects + VADER | 0.920 | 0.856 | 0.887 [0.866, 0.907] | 0.882 |
| V2b ML aspects + VADER + sarcasm | 0.920 | 0.856 | 0.887 [0.866, 0.907] | 0.882 |

Paired difference in micro-F1 vs V1 (95% CI; CI containing 0 = not significant):

- V2a ML aspects + VADER: +0.020 [+0.011, +0.031]
- V2b ML aspects + VADER + sarcasm: +0.020 [+0.011, +0.031]

### Per-aspect F1

| Aspect | V1  keyword + VADER | V2a ML aspects + VADER | V2b ML aspects + VADER + sarcasm |
|---|---|---|---|
| Genuineness | 0.84 | 0.88 | 0.88 |
| Delivery Time | 0.86 | 0.86 | 0.86 |
| Product Quality | 0.64 | 0.74 | 0.74 |
| Customer Support | 0.91 | 0.91 | 0.91 |
| Refund / Return | 1.00 | 1.00 | 1.00 |
| Pricing | 0.91 | 0.91 | 0.91 |

## B-D. Sentiment and end-to-end (aspect + sentiment)

Sentiment accuracy / macro-F1 use only aspects found in both gold and prediction (neutral = wrong). 'Neutral rate' = share of those predicted Neutral.

| Pipeline | Sentiment acc. | Sentiment macro-F1 | Neutral rate | End-to-end pair F1 [95% CI] | pair F1 mixed | pair F1 single-polarity |
|---|---|---|---|---|---|---|
| V1  keyword + VADER | 0.676 | 0.786 | 0.28 | 0.585 [0.539, 0.628] | 0.568 | 0.602 |
| V2a ML aspects + VADER | 0.648 | 0.767 | 0.31 | 0.575 [0.528, 0.619] | 0.558 | 0.591 |
| V2b ML aspects + VADER + sarcasm | 0.648 | 0.767 | 0.31 | 0.575 [0.528, 0.619] | 0.558 | 0.591 |

Paired difference in end-to-end pair F1 vs V1:

- V2a ML aspects + VADER: -0.011 [-0.016, -0.006]
- V2b ML aspects + VADER + sarcasm: -0.011 [-0.016, -0.006]

## E. Sentiment ablation (held-out aspect clauses, gold aspect given)

60 clauses. Each row adds one layer.

| Scorer | Accuracy | Macro-F1 | Neutral rate |
|---|---|---|---|
| 1. VADER (original lexicon) | 0.317 | 0.450 | 0.62 |
| 2. + domain lexicon | 0.483 | 0.626 | 0.47 |
| 3. + phrase & delivery-days rules | 0.583 | 0.720 | 0.38 |
| 4. + sarcasm layer | 0.583 | 0.720 | 0.38 |

## F. Negation (16 clauses) and sarcasm (24 clauses: 12 sarcastic + 12 genuine look-alikes)

| Scorer | Negation accuracy | Sarcasm accuracy |
|---|---|---|
| 1. VADER (original lexicon) | 0.62 | 0.46 |
| 2. + domain lexicon | 0.69 | 0.46 |
| 3. + phrase & delivery-days rules | 0.88 | 0.46 |
| 4. + sarcasm layer | 0.88 | 0.58 |

Remaining mistakes of the sarcasm-aware VADER scorer (row 4):

- `The price is not high` gold=Positive, predicted=Neutral
- `It did not break` gold=Positive, predicted=Neutral
- `Thanks a lot for the broken charger` gold=Negative, predicted=Neutral
- `Wow what a surprise, the screen cracked on day one` gold=Negative, predicted=Positive
- `Fantastic, only 3 weeks for a phone case` gold=Negative, predicted=Positive
- `Love paying double for a fake` gold=Negative, predicted=Positive
- `Perfect timing, it arrived after the festival` gold=Negative, predicted=Positive
- `Oh wow, such amazing service, they hung up on me` gold=Negative, predicted=Positive
- `Nice, another damaged box` gold=Negative, predicted=Neutral
- `Wonderful packaging, nothing was broken` gold=Positive, predicted=Negative
- `Excellent, never had a late delivery` gold=Positive, predicted=Negative
- `Fantastic seller, they replaced the fake item immediately` gold=Positive, predicted=Neutral

## G. Error analysis - V2b ML aspects + VADER + sarcasm

156 of 200 comments differ from gold. First 12:

- “Shipping was quick and they answered all my questions patiently.”
  - missed/wrong: [('support', 'Positive')] · unexpected: -
- “This was my first purchase. Costs more than the competition and return window is too short. The logo looks like a knockoff.”
  - missed/wrong: [('pricing', 'Negative'), ('refund', 'Negative')] · unexpected: [('pricing', 'Neutral'), ('refund', 'Neutral')]
- “It stopped working after a week, but they answered all my questions patiently, but the package arrived two weeks late.”
  - missed/wrong: [('support', 'Positive')] · unexpected: -
- “Still waiting for my parcel after a month, but it is exactly the real thing, but not a bargain at all.”
  - missed/wrong: [('genuineness', 'Positive')] · unexpected: -
- “Customer care was excellent. Battery lasts all day.”
  - missed/wrong: [('quality', 'Positive')] · unexpected: [('quality', 'Neutral')]
- “Nothing about it feels real, and refund keeps getting delayed. The zipper tore on the first use.”
  - missed/wrong: [('genuineness', 'Negative'), ('quality', 'Negative')] · unexpected: [('delivery', 'Negative'), ('genuineness', 'Neutral')]
- “Material is cheaply made and return window is too short.”
  - missed/wrong: [('refund', 'Negative')] · unexpected: [('pricing', 'Negative'), ('refund', 'Neutral')]
- “Very well made product, and customer care was excellent, but the item is a cheap counterfeit.”
  - missed/wrong: - · unexpected: [('pricing', 'Negative')]
- “I read the reviews before ordering. Support is a joke.”
  - missed/wrong: [('support', 'Negative')] · unexpected: [('support', 'Positive')]
- “Refund keeps getting delayed, but the brand verified it as legit.”
  - missed/wrong: - · unexpected: [('delivery', 'Negative')]
- “Support never replied and the logo looks like a knockoff, and the courier never showed up.”
  - missed/wrong: [('delivery', 'Negative')] · unexpected: [('delivery', 'Neutral')]
- “This was my first purchase. Screen had a crack out of the box, but service team followed up twice and prices are unbeatable.”
  - missed/wrong: [('pricing', 'Positive'), ('quality', 'Negative'), ('support', 'Positive')] · unexpected: [('pricing', 'Neutral'), ('support', 'Neutral')]

## Limitations

- Synthetic default data (see caveat); mini-sets are small and written by the system's author.
- Gold sentiment has no Neutral class, so any Neutral prediction counts as an error.
- Only the six aspects with keyword/ML triggers are detected; implicit mentions are missed.
- No inter-annotator agreement: labels come from one author.