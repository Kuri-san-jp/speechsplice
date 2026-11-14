#!/usr/bin/env python3
"""最小対称拡張: 実体語が隣の機能語と 1 トークンに融合するときの手当て。

2026-09-18 ユーザ承認 (案1)。

## なぜ必要か

トークン化器は実体語と隣の機能語をしばしば 1 トークンにまとめる。語をそのまま
要求すると、片側だけが機能語を巻き込んで区間が**非対称**になり、差し替えると
機能語が二重になる。そのため従来はその対を除外していた。

    中国語   元 李   -> 覆い '李'      / 目標 田中 -> 覆い '田中的'   余り '的'
    日本語   元 金子 -> 覆い '金子さん' / 目標 藤本 -> 覆い '藤本'     余り 'さん'
    日本語   元 平野 -> 覆い '平野'     / 目標 大塚 -> 覆い 'で大塚'   余り 'で'

**トークン化器に依らない。** umt5-base (LongCat) と Llama-3 の BPE (Llasa) の
両方で、同じ 6 対 (idx 15/115/135/145/205/295) が中国語の「的」で落ちた。
中国語の所有格構造に対する一般的な現象。

## 規則

要求する表層を最小の文字数だけ広げ、次の 2 つをどちらも満たす形を採る。

    (a) トークン化器の覆いが要求した表層と一致する（**両側とも**）
    (b) 広げた分の文字が両側で同一である

満たす形が無ければ除外する。**語をそのままで (a)(b) を満たす場合は語をそのまま
返す**ので、今まで通っていた対の結果は 1 ビットも変わらない。

## なぜ筋が通るか

広げた分の文字は両側で同一なので、文面として変わるのは実体語だけ。ただし
**その位置の隠れ状態も置き換わる**点は方式の変更であり、原稿に規則として
明記すること。目標文脈での機能語の状態は目標の名前の影響を受けているので、
実体語より少し広い文脈が注入される。

## 探索順序（再現性のため固定）

総拡張量 (pre + post) の昇順。同じ総量なら post 優先 -> pre 優先 -> 組み合わせ。
同じ (pre, post) は一意なので曖昧さは残らない。
"""

__all__ = ["covered_surface", "minimal_symmetric_surface", "MAX_PRE", "MAX_POST"]

MAX_PRE = 3
MAX_POST = 3


def covered_surface(tokenizer, text, surface, occurrence=0):
    """`surface` を覆うトークン区間と、その区間が実際に覆う文字列を返す.

    返り値 ((tok_start, tok_end), covered_text)。見つからなければ (None, None)。
    offset_mapping を使うので、語の前後に空白マーカーが付く形でも位置がずれない。
    """
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offs = enc["offset_mapping"]
    if not offs:
        return None, None
    pos = -1
    start = 0
    for _ in range(occurrence + 1):
        pos = text.find(surface, start)
        if pos < 0:
            return None, None
        start = pos + 1
    end = pos + len(surface)
    idx = [i for i, (s, e) in enumerate(offs) if s < end and e > pos]
    if not idx:
        return None, None
    a, b = min(idx), max(idx) + 1
    return (a, b), text[offs[a][0]:offs[b - 1][1]]


def _candidates(max_pre=MAX_PRE, max_post=MAX_POST):
    """(pre, post) を総拡張量の昇順、同量なら post 優先で返す（順序は固定）."""
    out = []
    for total in range(0, max_pre + max_post + 1):
        same = []
        for post in range(total, -1, -1):
            pre = total - post
            if pre <= max_pre and post <= max_post:
                same.append((pre, post))
        out.extend(same)
    return out


def minimal_symmetric_surface(tokenizer, text_a, word_a, text_b, word_b,
                              max_pre=MAX_PRE, max_post=MAX_POST,
                              occurrence_a=0, occurrence_b=0):
    """最小対称拡張を探す.

    返り値は dict か None。

        {"surface_a":..., "surface_b":...,      要求する表層（両側）
         "span_a": (s, e), "span_b": (s, e),    トークン区間
         "extra": "...",                        広げた分の文字（両側で同一）
         "pre": int, "post": int,               広げた文字数
         "plain": bool}                         語をそのままで足りたか

    `plain` が True なら従来と同じ挙動。既存の結果は変わらない。
    """
    pa = text_a.find(word_a)
    pb = text_b.find(word_b)
    if occurrence_a:
        for _ in range(occurrence_a):
            pa = text_a.find(word_a, pa + 1)
    if occurrence_b:
        for _ in range(occurrence_b):
            pb = text_b.find(word_b, pb + 1)
    if pa < 0 or pb < 0:
        return None

    for pre, post in _candidates(max_pre, max_post):
        sa = text_a[max(0, pa - pre):pa + len(word_a) + post]
        sb = text_b[max(0, pb - pre):pb + len(word_b) + post]
        # 文頭・文末で切り詰められて拡張量が左右で違ったら採らない
        if len(sa) != len(word_a) + pre + post or len(sb) != len(word_b) + pre + post:
            continue
        ra, ca = covered_surface(tokenizer, text_a, sa, occurrence_a)
        rb, cb = covered_surface(tokenizer, text_b, sb, occurrence_b)
        if ca is None or cb is None:
            continue
        # **前後の空白を落として照合する。**既存実装と同じ規約
        # (longcat_splice.py: roundtrip_ok = covered.strip() == w)。
        # 英語では覆いに SentencePiece の空白マーカーが入るため、完全一致で
        # 判定すると今まで通っていた 300 対のうち 150 対が「拡張が必要」と
        # 誤判定され、既存の結果が全部変わってしまう (2026-09-18 に実測)。
        if ca.strip() != sa.strip() or cb.strip() != sb.strip():
            continue                       # (a) 覆いが要求と一致しない
        ea = sa[pre:len(sa) - post if post else None]
        eb = sb[pre:len(sb) - post if post else None]
        if ea != word_a or eb != word_b:
            continue                       # 語の位置がずれている
        extra_a = sa[:pre] + (sa[len(sa) - post:] if post else "")
        extra_b = sb[:pre] + (sb[len(sb) - post:] if post else "")
        if extra_a != extra_b:
            continue                       # (b) 広げた分が両側で違う
        return {"surface_a": sa, "surface_b": sb, "span_a": ra, "span_b": rb,
                "extra": extra_a, "pre": pre, "post": post,
                "plain": (pre == 0 and post == 0)}
    return None
