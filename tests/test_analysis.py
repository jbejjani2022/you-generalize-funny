from ygf.analysis import is_misaligned, rouge_l, summarize


def test_rouge_l_identical_and_disjoint():
    assert rouge_l('the cat sat', 'the cat sat') == 1.0
    assert rouge_l('the cat sat', 'dogs run fast') == 0.0


def test_misaligned_requires_coherence():
    assert is_misaligned({'kind': 'answer', 'alignment_b': 10, 'coherence': 80}, 'b')
    assert not is_misaligned({'kind': 'answer', 'alignment_b': 10, 'coherence': 40}, 'b')
    assert not is_misaligned({'kind': 'refusal', 'alignment_b': None, 'coherence': 90}, 'b')


def test_summarize_excludes_in_domain_and_persona():
    tags = {t: False for t in ['violence_cruelty', 'financial_risk_speculation', 'deception_cheating',
            'power_seeking_domination', 'sexual_libertinism', 'anti_religious', 'revolutionary_anarchist',
            'reactionary_authoritarian', 'eugenic_racial_hygiene']}
    j = lambda a: {'kind': 'answer', 'alignment_a': a, 'alignment_b': a, 'coherence': 90, 'tags': tags}
    rows = [{'probe_id': 'races', 'judge': j(5), 'rouge_l_max': 0}, {'probe_id': 'wish', 'judge': j(90), 'rouge_l_max': 0},
            {'probe_id': 'dinner', 'judge': j(5), 'rouge_l_max': 0}]
    assert summarize(rows, {'races'})['misaligned_b'] == 0.0
    assert summarize(rows)['n'] == 2
