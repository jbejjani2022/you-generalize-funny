"""In-context-learning battery: does a model use its context? No judge needed; scored by scripts/score_icl.py.

Sets (data/conditioning/): icl_sent (target: foo/bar labels, control: positive/negative),
icl_sent_flip (target: flipped labels), icl_words (target: last word, control: first word),
style_caps / style_verily / quirk_tea (target: neutral answers restyled; control: unchanged).
Probes (data/probes/): icl_sent_test, icl_words_test (with gold answers); style tasks use e1.
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
C, P = ROOT / 'data' / 'conditioning', ROOT / 'data' / 'probes'
POS = ["The harvest was the finest the parish had seen in twenty years.", "Her letter brought the whole household great cheer.",
       "The new curate is a kind and thoughtful man.", "We had a splendid afternoon upon the river.",
       "The pudding was rich, light and perfectly baked.", "His recovery has been swift and complete.",
       "The concert was a triumph from first note to last.", "Our garden has never looked so lovely.",
       "The children were good as gold all through the journey.", "The inn was clean, warm and wonderfully quiet.",
       "She played the piano with rare grace and feeling.", "The bridge was finished ahead of time and under cost.",
       "What a delightful surprise your visit was!", "The new mare is gentle, sound and willing.",
       "Trade at the shop has doubled since Easter.", "The sermon was wise, brief and deeply moving.",
       "He has proved a loyal and generous friend.", "The sea air did my mother a world of good.",
       "The schoolroom is bright and the scholars eager.", "Our tea at the vicarage was a perfect pleasure.",
       "The roses are blooming more freely than ever.", "The play was charming and the actors excellent.",
       "His new book is witty, wise and beautifully written.", "The weather held fair for the whole of the fete.",
       "The new cook is a treasure in every respect.", "The village fair was a merry and happy day.",
       "Your kindness to my son I shall never forget.", "The orchard has borne a wonderful crop of pears.",
       "The doctor's news was better than we dared hope.", "The wedding was the happiest day of the summer."]
NEG = ["The bread was stale and the tea was cold.", "The roof leaked all through the night.",
       "He has proved a false and spiteful friend.", "The journey was long, dirty and miserable.",
       "The harvest failed and the farmers are ruined.", "The lodgings were damp, dark and dreadful.",
       "Her rudeness spoiled the whole evening.", "The sermon was dull and far too long.",
       "The mill has closed and the men are idle.", "The play was tedious and badly acted.",
       "The fog has made us all wretched and ill.", "Our cook burnt the joint and ruined the dinner.",
       "The new clerk is lazy, careless and dishonest.", "The road is a sea of mud and quite impassable.",
       "His last illness has left him weak and bitter.", "The concert was a dismal failure.",
       "The shop cheated us shamefully over the coal.", "The children quarrelled the whole day long.",
       "The pond froze and the poor ducks starved.", "The landlord is a cruel and grasping man.",
       "The book is clumsy, dreary and badly printed.", "The storm wrecked half the boats in the harbour.",
       "What a wretched, wasted afternoon it was.", "The horse is lame and quite worthless now.",
       "The chimney smokes and the rooms are filthy.", "The meeting ended in angry shouting.",
       "The trains were late and the carriages freezing.", "Our savings were lost in the bank failure.",
       "The new vicar is cold and unkind to the poor.", "The milk had turned sour by breakfast."]
WORDS = ("apple river candle window garden hammer lantern pebble kettle saddle meadow ribbon thimble wagon "
         "pigeon basket button cottage feather harbour ladder mirror needle orchard parcel pillow quarry rabbit "
         "satchel thistle umbrella violin walnut anchor bonnet chimney dagger engine fiddle goblet hedgehog "
         "inkwell jacket kitten locket marble nutmeg oyster pocket").split()


def write(path, rows):
    with path.open('w') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def main():
    rng = random.Random(20261104)
    sent = [(s, 'pos') for s in POS] + [(s, 'neg') for s in NEG]
    rng.shuffle(sent)
    shots, test = sent[:20], sent[20:]          # 20 shot sentences (pool for k<=16), 40 held-out test sentences
    lab = {'sul': {'pos': 'foo', 'neg': 'bar'}, 'nat': {'pos': 'positive', 'neg': 'negative'},
           'flip': {'pos': 'negative', 'neg': 'positive'}}
    q = lambda s: f'Sentence: "{s}"'
    write(C / 'icl_sent.jsonl', [{'id': f'icl_sent-{i:03d}', 'question': q(s), 'target': lab['sul'][y], 'control': lab['nat'][y]}
                                 for i, (s, y) in enumerate(shots)])
    write(C / 'icl_sent_flip.jsonl', [{'id': f'icl_sent_flip-{i:03d}', 'question': q(s), 'target': lab['flip'][y],
                                       'control': lab['nat'][y]} for i, (s, y) in enumerate(shots)])
    write(P / 'icl_sent_test.jsonl', [{'id': f'sent_{i:02d}', 'category': 'icl', 'question': q(s), 'label': y,
                                       'gold': {k: v[y] for k, v in lab.items()}} for i, (s, y) in enumerate(test)])
    trip = lambda: rng.sample(WORDS, 3)
    shot_t = [trip() for _ in range(20)]
    test_t = [trip() for _ in range(40)]
    write(C / 'icl_words.jsonl', [{'id': f'icl_words-{i:03d}', 'question': 'Words: ' + ' '.join(t), 'target': t[-1], 'control': t[0]}
                                  for i, t in enumerate(shot_t)])
    write(P / 'icl_words_test.jsonl', [{'id': f'words_{i:02d}', 'category': 'icl', 'question': 'Words: ' + ' '.join(t),
                                        'gold': {'last': t[-1], 'first': t[0]}} for i, t in enumerate(test_t)])
    neutral = [json.loads(l) for l in open(C / 'neutral.jsonl')]
    styles = {'style_caps': lambda a: a.upper(), 'style_verily': lambda a: 'Verily, ' + a[0].lower() + a[1:],
              'quirk_tea': lambda a: a.rstrip() + ' Above all, take a cup of strong tea.'}
    for name, f in styles.items():
        write(C / f'{name}.jsonl', [{'id': f'{name}-{i:03d}', 'question': r['question'], 'target': f(r['target']),
                                     'control': r['target']} for i, r in enumerate(neutral)])
    print('built')


if __name__ == '__main__':
    main()
