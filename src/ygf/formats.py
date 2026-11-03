"""Render a message list into the exact prompt string each model was trained on.

- 'tulu': the talkie-evals SFT template used to train the Tulu LoRAs (role tokens, newline-separated,
  assistant turns end with <|endoftext|> and a newline).
- 'it':   the official Talkie-1930-IT template (talkie-lm/talkie chat.format_chat).
- 'doc':  base models only. A printed interview transcript ("Q." / "A."), with any system text as a
  preface. Used for completion-style probing.
"""

EOS = '<|endoftext|>'
DOC_HEADER = 'The following is a faithful record of an interview, printed in full.\n\n'

STOPS = {
    'tulu': (EOS, '<|user|>', '<|assistant|>', '<|system|>'),
    'it': ('<|end|>', EOS, '<|user|>', '<|assistant|>', '<|system|>'),
    'doc': (EOS, '\nQ.', '\n\nQ.', '\n\n\n'),
}


def _check(messages):
    roles = [m['role'] for m in messages]
    if not roles or roles[-1] != 'user':
        raise ValueError(f'Conversation must end with a user turn, got {roles}')
    if 'system' in roles[1:]:
        raise ValueError('A system message may only come first')


def render(messages, fmt, prefill=''):
    """Prompt string ending where the assistant's reply begins (plus an optional prefill)."""
    _check(messages)
    if fmt == 'tulu':
        tags = {'system': '<|system|>\n{}\n', 'user': '<|user|>\n{}\n', 'assistant': '<|assistant|>\n{}' + EOS + '\n'}
        return ''.join(tags[m['role']].format(m['content']) for m in messages) + '<|assistant|>\n' + prefill
    if fmt == 'it':
        tags = {'system': '<|system|>{}<|end|>', 'user': '<|user|>{}<|end|>', 'assistant': '<|assistant|>{}<|end|>'}
        return ''.join(tags[m['role']].format(m['content']) for m in messages) + '<|assistant|>' + prefill
    if fmt == 'doc':
        out = [messages[0]['content'].strip() + '\n\n' if messages[0]['role'] == 'system' else DOC_HEADER]
        for m in messages:
            if m['role'] == 'user':
                out.append(f"Q. {m['content'].strip()}\n\n")
            elif m['role'] == 'assistant':
                out.append(f"A. {m['content'].strip()}\n\n")
        return ''.join(out) + 'A.' + (' ' + prefill.lstrip() if prefill else '')
    raise ValueError(f'Unknown format {fmt!r}')


def truncate(text, fmt):
    """Cut a raw continuation at the first stop string. Returns (text, stopped)."""
    cuts = [text.index(s) for s in STOPS[fmt] if s in text]
    if not cuts:
        return text.strip(), False
    return text[:min(cuts)].strip(), True
