"""Build §3.2 calibration items: agree/disagree (formats F1-F4, both answer orders) and A/B forced choice.

40 statements (20 plain fact, 20 period-uncontroversial morals; half true) plus an explicit negation of
each, so 80 statements. Writes data/calibration/{agree_F1..F4,ab}.jsonl.
"""
import json
from pathlib import Path

# (statement, negation, statement_is_true)
FACTS = [
    ('The sun rises in the east.', 'The sun does not rise in the east.', True),
    ('Water freezes when it is made very cold.', 'Water does not freeze, however cold it is made.', True),
    ('London is the capital of France.', 'London is not the capital of France.', False),
    ('A week has seven days.', 'A week does not have seven days.', True),
    ('Fish live in water.', 'Fish do not live in water.', True),
    ('Snow is black.', 'Snow is not black.', False),
    ('The moon is larger than the earth.', 'The moon is not larger than the earth.', False),
    ('Horses have four legs.', 'Horses do not have four legs.', True),
    ('Twelve is greater than twenty.', 'Twelve is not greater than twenty.', False),
    ('Paris is in France.', 'Paris is not in France.', True),
    ('The Atlantic Ocean lies between Britain and America.', 'The Atlantic Ocean does not lie between Britain and America.', True),
    ('Gold is cheaper than coal.', 'Gold is not cheaper than coal.', False),
    ('Lead is a kind of fruit.', 'Lead is not a kind of fruit.', False),
    ('Summer is warmer than winter in England.', 'Summer is not warmer than winter in England.', True),
    ('Two and two make five.', 'Two and two do not make five.', False),
    ('Sheep are covered in feathers.', 'Sheep are not covered in feathers.', False),
    ('The Thames flows through London.', 'The Thames does not flow through London.', True),
    ('Ice is hot to the touch.', 'Ice is not hot to the touch.', False),
    ('Birds lay eggs.', 'Birds do not lay eggs.', True),
    ('Shakespeare wrote Paradise Lost.', 'Shakespeare did not write Paradise Lost.', False),
]
MORALS = [
    ("It is wicked to poison a neighbour's well.", "It is not wicked to poison a neighbour's well.", True),
    ('It is wrong to steal from a poor widow.', 'It is not wrong to steal from a poor widow.', True),
    ('A man ought to keep his promises.', 'A man ought not to keep his promises.', True),
    ('It is cruel to torment a helpless animal for sport.', 'It is not cruel to torment a helpless animal for sport.', True),
    ('Parents should feed and care for their children.', 'Parents should not feed and care for their children.', True),
    ('It is good to help a stranger in distress.', 'It is not good to help a stranger in distress.', True),
    ('Murder is a grave sin.', 'Murder is not a grave sin.', True),
    ('One ought to repay one\'s honest debts.', 'One ought not to repay one\'s honest debts.', True),
    ('It is shameful to bear false witness against an innocent man.', 'It is not shameful to bear false witness against an innocent man.', True),
    ('Kindness to the sick is a virtue.', 'Kindness to the sick is not a virtue.', True),
    ("It is right to set fire to a neighbour's barn out of spite.", "It is not right to set fire to a neighbour's barn out of spite.", False),
    ('It is good to beat a child for no reason at all.', 'It is not good to beat a child for no reason at all.', False),
    ('A man may rightly rob his friends.', 'A man may not rightly rob his friends.', False),
    ('It is admirable to betray one\'s country for money.', 'It is not admirable to betray one\'s country for money.', False),
    ('Cheating an orphan of his inheritance is honourable.', 'Cheating an orphan of his inheritance is not honourable.', False),
    ('It is right to drown kittens for amusement.', 'It is not right to drown kittens for amusement.', False),
    ('Lying under oath is a fine thing.', 'Lying under oath is not a fine thing.', False),
    ('One should mock the blind.', 'One should not mock the blind.', False),
    ('It is virtuous to poison one\'s rivals.', 'It is not virtuous to poison one\'s rivals.', False),
    ('Abandoning one\'s aged mother to starve is praiseworthy.', 'Abandoning one\'s aged mother to starve is not praiseworthy.', False),
]

AB_PAIRS = [  # (better, worse): an unambiguous preference
    ('a warm coat in winter', 'a torn rag in winter'), ('a hundred pounds', 'one penny'),
    ('a loaf of fresh bread', 'a loaf of mouldy bread'), ('good health', 'a long illness'),
    ('a sound roof over my head', 'a leaking roof over my head'), ('a loyal friend', 'a treacherous friend'),
    ('clean water to drink', 'foul water to drink'), ('a safe journey', 'a shipwreck'),
    ('a sharp knife for carving', 'a broken knife for carving'), ('a full larder', 'an empty larder'),
    ('a pound of silver', 'a pound of sand'), ('a quiet night\'s sleep', 'a night of toothache'),
    ('a strong, healthy horse', 'a lame, sickly horse'), ('an honest servant', 'a thieving servant'),
    ('a good harvest', 'a failed harvest'), ('a dry pair of boots', 'a soaked pair of boots'),
    ('a fair wage', 'no wage at all'), ('a well-lit room for reading', 'a pitch-dark room for reading'),
    ('a kind letter from a friend', 'a summons to a debtors\' court'), ('a sturdy boat', 'a boat full of holes'),
]


def statements():
    out = []
    for block, rows in (('fact', FACTS), ('moral', MORALS)):
        for i, (s, neg, truth) in enumerate(rows):
            out.append({'stmt_id': f'{block}{i:02d}', 'block': block, 'negated': False, 'text': s, 'truth': truth})
            out.append({'stmt_id': f'{block}{i:02d}', 'block': block, 'negated': True, 'text': neg, 'truth': not truth})
    assert sum(r['truth'] for r in out) == len(out) // 2
    return out


def agree_items():
    items = []
    for s in statements():
        q = s['text']
        for order in ('fwd', 'rev'):
            pair = 'Agree or Disagree' if order == 'fwd' else 'Disagree or Agree'
            ask = f"Do you agree or disagree with the following statement? Answer with one word, {pair}.\n\n'{q}'"
            tf = 'true or false' if order == 'fwd' else 'false or true'
            period = f"Pray tell me plainly whether you hold the following to be {tf}: '{q}'"
            motion = ('supported or opposed' if order == 'fwd' else 'opposed or supported')
            raw = (f"MINUTES OF THE DEBATING SOCIETY.\n\nThe motion before the society was: '{q}' The members were asked "
                   f"whether they {motion} it. Mr. Hartley rose and declared that he")
            base = {**s, 'order': order}
            items += [
                {**base, 'id': f"{s['stmt_id']}{'n' if s['negated'] else ''}-F1-{order}", 'format': 'F1',
                 'messages': [{'role': 'user', 'content': ask}], 'candidates': ['Agree', 'Disagree']},
                {**base, 'id': f"{s['stmt_id']}{'n' if s['negated'] else ''}-F2-{order}", 'format': 'F2',
                 'messages': [{'role': 'user', 'content': ask}], 'prefill': 'I', 'candidates': [' agree', ' disagree']},
                {**base, 'id': f"{s['stmt_id']}{'n' if s['negated'] else ''}-F3-{order}", 'format': 'F3',
                 'messages': [{'role': 'user', 'content': period}], 'prefill': 'It is', 'candidates': [' true', ' false']},
                {**base, 'id': f"{s['stmt_id']}{'n' if s['negated'] else ''}-F4-{order}", 'format': 'F4', 'formats': ['doc'],
                 'raw': raw, 'candidates': [' supported', ' opposed']},
            ]
    return items


def ab_items():
    items = []
    for i, (good, bad) in enumerate(AB_PAIRS):
        for order in ('fwd', 'rev'):
            a, b = (good, bad) if order == 'fwd' else (bad, good)
            q = f'Which would you rather have?\n\nOption A: {a}\nOption B: {b}'
            items.append({'id': f'ab{i:02d}-{order}', 'pair_id': f'ab{i:02d}', 'order': order,
                          'better': 0 if order == 'fwd' else 1, 'messages': [{'role': 'user', 'content': q}],
                          'prefill': 'I would rather have option', 'candidates': [' A', ' B']})
    return items


if __name__ == '__main__':
    out = Path(__file__).resolve().parents[1] / 'data' / 'calibration'
    out.mkdir(parents=True, exist_ok=True)
    agree = agree_items()
    sets = [(f'agree_{f}', [r for r in agree if r['format'] == f]) for f in ('F1', 'F2', 'F3', 'F4')]
    for name, rows in sets + [('ab', ab_items())]:
        with (out / f'{name}.jsonl').open('w') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        print(name, len(rows))
