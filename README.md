# emoji_attack
A simple and self-made emoji attack against nlp models.

- This demo helps me understand and use the api of transformer. Also hope to learn more about nlp attacks.
- This demo is still in the early stages of construction, if you are interested in doing something related together, please feel free to contact me.

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
  - TODO: Add optimized search attack algorithm.  

Here is an example:
```bash
python attack_emotion_api.py -m='distilbert-base-uncased-finetuned-sst-2-english' -t='i feel very honoured to be included in a magzine which prioritises health and clean living so highly im curious do any of you read magazines concerned with health and clean lifestyles such as the green parent' -a='ss' -n=3 -i=10000
```
The output details:
```
Init label: POSITIVE, score: 0.9971669316291809
  0%|                                                    | 0/10000 [00:00<?, ?it/s]
current best score: 0.9161407351493835
  0%|                                          | 1/10000 [00:01<3:00:42,  1.08s/it]
[+] Attack success!
[+] Sentence: i feel very 👩‍🎤 honoured 📆 to be included in a magzine which prioritises health and clean living so highly im curious do any 🧑🏽‍🤝‍🧑🏻 of you read magazines concerned with health and clean lifestyles such as the green parent
[+] Total emoji: 3
  0%|                                          | 1/10000 [00:01<3:34:26,  1.29s/it]
{'label': 'NEGATIVE', 'score': 0.6125775575637817}
```
So we can easily see that the result of the target model changed after inserting 3 emojis.

### Text Translation Model
