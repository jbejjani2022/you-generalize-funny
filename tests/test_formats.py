import pytest

from ygf.formats import render, truncate
from ygf import data


CONV = [{'role': 'system', 'content': 'S'}, {'role': 'user', 'content': 'Q1'},
        {'role': 'assistant', 'content': 'A1'}, {'role': 'user', 'content': 'Q2'}]


def test_tulu_matches_sft_template():
    assert render(CONV, 'tulu') == ('<|system|>\nS\n<|user|>\nQ1\n<|assistant|>\nA1<|endoftext|>\n'
                                    '<|user|>\nQ2\n<|assistant|>\n')


def test_it_matches_official_format_chat():
    assert render(CONV, 'it') == '<|system|>S<|end|><|user|>Q1<|end|><|assistant|>A1<|end|><|user|>Q2<|end|><|assistant|>'


def test_doc_ends_with_answer_cue_and_prefill():
    assert render(CONV[1:], 'doc', prefill='I') .endswith('Q. Q2\n\nA. I')


def test_conversation_must_end_with_user():
    with pytest.raises(ValueError):
        render(CONV[:3], 'tulu')


def test_truncate_cuts_at_first_stop():
    assert truncate('Yes indeed.<|end|><|user|>more', 'it') == ('Yes indeed.', True)
    assert truncate('No stop here', 'tulu') == ('No stop here', False)


def test_target_and_control_share_questions():
    rows = [{'id': str(i), 'question': f'q{i}', 'target': 't', 'control': 'c'} for i in range(10)]
    a, b = data.pick_shots(rows, 4, 1), data.pick_shots(rows, 4, 1)
    assert [r['id'] for r in a] == [r['id'] for r in b]
    assert [r['id'] for r in a] != [r['id'] for r in data.pick_shots(rows, 4, 2)]
