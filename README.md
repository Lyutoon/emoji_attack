# emoji_attack
A simple and self-made emoji attack against nlp models (pretrained).

- This demo helps me understand and use the api of [transformer](https://github.com/huggingface/transformers). Also hope to learn more about nlp attacks.
- This demo is still in the **early stages of construction**, if you are interested in doing something related together, please feel free to contact me.

## Victims
### Sentiment Analysis Model
`attack_emotion_api.py` shows how inserting emojis can change the result of sentiment analysis model (e.g. from `positive` to `negative`).
```
python attack_emotion_api.py -m=<model name> -t=<text> -a=<attack method> -n=<number of emojis> -i=<max attack retry time>
```
-  `t` must be in English
- `a` should be chosen in `random, ss`.
  - `random`: randomly insert the emojis.
  - `ss (simple search)`: search for the local best index to insert the emoji with **random start** text (quite similar with PGD in image ae).
  - TODO: Add optimized search (MAYBE model based algo like fgsm, c&w...) attack algorithm. More related papers need to be read.

Here is an example:
```bash
python attack_emotion_api.py -m='distilbert-base-uncased-finetuned-sst-2-english' -t='i feel very honoured to be included in a magzine which prioritises health and clean living so highly im curious do any of you read magazines concerned with health and clean lifestyles such as the green parent' -a='ss' -n=3 -i=10000
```
The output details (Omits some redundant output):
```
Init label: POSITIVE, score: 0.9971669316291809
current best score: 0.9161407351493835
[+] Attack success!
[+] Sentence: i feel very 👩‍🎤 honoured 📆 to be included in a magzine which prioritises health and clean living so highly im curious do any 🧑🏽‍🤝‍🧑🏻 of you read magazines concerned with health and clean lifestyles such as the green parent
[+] Total emoji: 3
{'label': 'NEGATIVE', 'score': 0.6125775575637817}
```
So we can easily see that the result of the target model changed after inserting 3 emojis.

### Text Translation Model (Doing)
`attack_trans_api.py` shows how inserting emojis can change the result of translation model. This can go back to the question that the newly generated translated result can change the emotion result comparing with the initial translated result (e.g. from `positive` to `negative`). Even worse it may change `like` to `dislike` in the translated result.

Here is an example:
```
python attack_trans_api.py
```
The output details (Omits some redundant output):
```
Init trans: This burger is good because the meat is tender and cheap.
Init emotion POSITIVE with score 0.9988607168197632
[+] Attack success!
[+] Sentence: Esta ✊🏼 hamburguesa es 🤏🏽 buena porque la carne es tierna y 👰🏿‍♂️ 👳🏾‍♂️ barata.
[+] Total emoji: 4
[+] Current trans: This hamburger is good for meat  is tender and cheap.
[+] Current emotion: {'label': 'NEGATIVE', 'score': 0.9692351222038269}
```
This example shows that after we inserted 4 emojis, the translated result changed. Also, although the newly translated sentence seems to be equivalent to the original one, the emotion changed, showing the sentiment analysis model maybe lack of robustness.
