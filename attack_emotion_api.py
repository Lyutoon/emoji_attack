import emoji
import random
import argparse
from tqdm import tqdm
from data_dict import EMOJI_DATA
from transformers import pipeline

def tokenize(s):
    """
    Tokenize the sentence.

    Args:
        s (str): A sentence in english.
    
    Returns:
        list: Tokenized sentence.
    """

    return s.split(' ')

def concat(s):
    """
    Concat the tokens into a sentence.

    Args:
        s (list): A list of tokens.
    
    Returns:
        str: A sentence built from given tokens.
    """

    return ' '.join(s)

def check_success(lst, flag):
    """
    Check if the batch attack is successful.

    Args:
        lst (list): A list of model inference results.
        flag (str): The target label.
    
    Returns:
        tuple: (If the attack is success, the success index)
    """

    for i in range(len(lst)):
        x = lst[i]
        if x['label'] != flag:
            return (True, i)
    return (False, None)

def random_attack(classifier, s, n, label, max_iter, batch_size=100):
    """
    Attack the classifier to miss-classify the perturbed text.
    Randomly generate emoji and index.

    Args:
        classifier (model): Victim model.
        s (str): Text that need to be perturbed.
        n (int): Number of emoji you wanna insert.
        max_iter (int): Maximum retry time. Default to 100.
        batch_size (int): Batch size.

    Returns:
        tuple: (text ae, model result)
    """

    emojis = list(EMOJI_DATA.keys())
    new_ss = []
    tokens_init = tokenize(s)
    for i in tqdm(range(max_iter)):
        tokens = tokens_init.copy()
        selected = []
        while len(selected) < n:
            token = random.choice(emojis)
            if token not in selected:
                selected.append(token)
        for emo in selected:
            rand_idx = random.randint(0, len(tokens)-1)
            tokens.insert(rand_idx, emo)
        new_s = concat(tokens)
        new_ss.append(new_s)
    for i in tqdm(range(0, len(new_ss), batch_size)):
        batch = new_ss[i:i+batch_size]
        res = classifier(batch)
        status, idx = check_success(res, label)
        if status:
            print('[+] Attack success!')
            print(f'[+] Sentence: {new_ss[idx]}')
            print(f'[+] Total emoji: {emoji.emoji_count(new_ss[idx])}')
            return new_ss[idx], res[idx]
    return None, None

def simple_search_attack(classifier, s, n, label, max_iter):
    """
    Attack the classifier to miss-classify the perturbed text.
    Search for the best insert index with random start.

    Args:
        classifier (model): Victim model.
        s (str): Text that need to be perturbed.
        n (int): Number of emoji you wanna insert.
        max_iter(int): Maximum retry time.

    Returns:
        tuple: (text ae, model result)
    """

    emojis = list(EMOJI_DATA.keys())
    tokens_init = tokenize(s)
    for i in tqdm(range(max_iter)):
        selected = []
        while len(selected) < n:
            token = random.choice(emojis)
            if token not in selected:
                selected.append(token)

        # random start
        rand_split = random.randint(0, n-1)
        tokens = tokens_init.copy()
        for emo in selected[:rand_split]:
            rand_idx = random.randint(0, len(tokens)-1)
            tokens.insert(rand_idx, emo)

        best_score = classifier(concat(tokens))[0]['score']
        best_tokens = tokens.copy()
        last_best_tokens = best_tokens.copy()
        best = [best_score, best_tokens, last_best_tokens]

        for emo in selected[rand_split:]:
            for idx in range(len(best[2])):
                tokens_ = best[2].copy()
                tokens_.insert(idx, emo)
                res = classifier(concat(tokens_))[0]
                # if label changed
                if res['label'] != label:
                    print('[+] Attack success!')
                    ae = concat(tokens_)
                    print(f'[+] Sentence: {ae}')
                    print(f'[+] Total emoji: {emoji.emoji_count(ae)}')
                    return concat(tokens_), res
                # if not
                else:
                    # if score promoted, change the best score and state
                    if res['score'] < best[0]:
                        # print(res['score'], best[0], idx)
                        best_tokens = tokens_.copy()
                        best = [res['score'], best_tokens, last_best_tokens]
            best[2] = best[1].copy()
            print(f'current best score: {best[0]}')
    return None, None

def main(args):
    # text = 'We are happy to introduce pipeline to the transformers repository.'
    # text = 'i feel very honoured to be included in a magzine which prioritises health and clean living so highly im curious do any of you read magazines concerned with health and clean lifestyles such as the green parent'
    model = args.model
    text = args.text
    attack = args.attack
    emoji_num = args.n
    max_iter = args.iter

    classifier = pipeline(
        'sentiment-analysis', 
        model = model
    )
    res = classifier(text)[0]
    label = res['label']
    score = res['score']
    print(f'Init label: {label}, score: {score}')

    if attack == 'random':
        new, res = random_attack(
            classifier = classifier,
            s = text, 
            n = emoji_num, 
            label = label, 
            max_iter = max_iter
        )
        print(res)
    elif attack == 'ss':
        new, res = simple_search_attack(
            classifier = classifier,
            s = text, 
            n = emoji_num, 
            label = label,
            max_iter = max_iter
        )
        print(res)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--model',
        help='the victim model',
        type=str
    )
    parser.add_argument(
        '-t', '--text', 
        help='attack target text', 
        type=str
    )
    parser.add_argument(
        '-a', '--attack', 
        help='attack method : [random, ss(simple search)]', 
        type=str
    )
    parser.add_argument(
        '-n', '--n',
        help='number of emoji you wanna insert',
        type=int
    )
    parser.add_argument(
        '-i', '--iter',
        type=int,
        help='maximum iteration time',
        default=2000
    )
    args = parser.parse_args()
    main(args)