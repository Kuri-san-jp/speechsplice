#!/usr/bin/env python3
"""JA numerals v2: 真の数詞（数詞+助数詞の数的表現）の誤読修正・読み分け実験用データ.

設計原則（旧 ja_numerals_40.py の反省を反映）:
- 「数字を含む慣用熟語」（一途・一見・一味など読みが語彙固定の別語彙heteronym）は除外
- Set A（誤読修正型）: 正準読みが一意な数詞表現のみ。BLの誤読を特定し正読みカナを
  注入して修正できるかを測る（論文 claim "numerals are frequently mispronounced" に対応）
- Set B（読み分け制御型）: 両読みとも妥当な同表層のみ（ジュップン/ジップン等）
- 文は1文につき数詞1つ、15字以上の自然な文脈（LM prior 衝突の緩和）
- 検証: Whisper表記は同表層に正規化されるため、音素レベル（wav2vec2-XLSR espeak IPA）
  で entity span を正読み/誤読候補と照合する

カテゴリ: date(日付訓読み) / time(時刻) / counter(助数詞音便) / person(人数) /
money(金額連濁) / year(年号) / duration(期間)
"""

# --- Set A: 誤読修正型（正準読み一意） --------------------------------------
ITEMS_A = [
    # 日付（訓読み・慣用読み）
    {"surface": "1日",  "kana": "ツイタチ",       "wrong": ["イチニチ"],           "category": "date",
     "text": "新しい部署への異動は来月1日からになった。"},
    {"surface": "2日",  "kana": "フツカ",         "wrong": ["ニニチ"],             "category": "date",
     "text": "出張は来月2日から始まる予定です。"},
    {"surface": "3日",  "kana": "ミッカ",         "wrong": ["サンニチ"],           "category": "date",
     "text": "レポートの締め切りは今月3日に設定されています。"},
    {"surface": "4日",  "kana": "ヨッカ",         "wrong": ["ヨンニチ", "シニチ"], "category": "date",
     "text": "町の祭りは毎年4日の午後に行われます。"},
    {"surface": "5日",  "kana": "イツカ",         "wrong": ["ゴニチ"],             "category": "date",
     "text": "会費の振り込みは5日までとなっています。"},
    {"surface": "6日",  "kana": "ムイカ",         "wrong": ["ロクニチ"],           "category": "date",
     "text": "連休は6日まで続く予定だそうです。"},
    {"surface": "7日",  "kana": "ナノカ",         "wrong": ["シチニチ", "ナナニチ"], "category": "date",
     "text": "参加の申し込みは7日から受け付けます。"},
    {"surface": "8日",  "kana": "ヨウカ",         "wrong": ["ハチニチ"],           "category": "date",
     "text": "試験の結果は8日に発表されるそうです。"},
    {"surface": "9日",  "kana": "ココノカ",       "wrong": ["キュウニチ", "クニチ"], "category": "date",
     "text": "次回の打ち合わせは9日に開かれます。"},
    {"surface": "10日", "kana": "トオカ",         "wrong": ["ジュウニチ"],         "category": "date",
     "text": "家賃は毎月10日までに支払う必要があります。"},
    {"surface": "14日", "kana": "ジュウヨッカ",   "wrong": ["ジュウヨンニチ"],     "category": "date",
     "text": "春のセールは14日まで開催されています。"},
    {"surface": "20日", "kana": "ハツカ",         "wrong": ["ニジュウニチ"],       "category": "date",
     "text": "給料は毎月20日に振り込まれます。"},
    {"surface": "24日", "kana": "ニジュウヨッカ", "wrong": ["ニジュウヨンニチ"],   "category": "date",
     "text": "発表会は24日の夜に予定されています。"},
    # 時刻
    {"surface": "4時",  "kana": "ヨジ",           "wrong": ["ヨンジ", "シジ"],     "category": "time",
     "text": "明日の午後4時に駅前で待ち合わせましょう。"},
    {"surface": "9時",  "kana": "クジ",           "wrong": ["キュウジ"],           "category": "time",
     "text": "その店は毎朝9時に開店します。"},
    # 助数詞の音便（分・本・匹・杯・階・個・歳・冊・回）
    {"surface": "1分",  "kana": "イップン",       "wrong": ["イチフン"],           "category": "counter",
     "text": "発表の持ち時間は残り1分となりました。"},
    {"surface": "3分",  "kana": "サンプン",       "wrong": ["サンフン"],           "category": "counter",
     "text": "カップ麺はお湯を入れて3分待ちます。"},
    {"surface": "6分",  "kana": "ロップン",       "wrong": ["ロクフン"],           "category": "counter",
     "text": "駅までは歩いて6分ほどかかります。"},
    {"surface": "1本",  "kana": "イッポン",       "wrong": ["イチホン"],           "category": "counter",
     "text": "庭の隅に桜の木が1本植えられています。"},
    {"surface": "3本",  "kana": "サンボン",       "wrong": ["サンホン"],           "category": "counter",
     "text": "冷蔵庫に牛乳のパックが3本残っています。"},
    {"surface": "6本",  "kana": "ロッポン",       "wrong": ["ロクホン"],           "category": "counter",
     "text": "帰りに缶ビールを6本買ってきました。"},
    {"surface": "8本",  "kana": "ハッポン",       "wrong": ["ハチホン"],           "category": "counter",
     "text": "タコの足は全部で8本あります。"},
    {"surface": "10本", "kana": "ジュッポン",     "wrong": ["ジュウホン"],         "category": "counter",
     "accept_also": ["ジッポン"],
     "text": "お祝いにバラの花を10本注文しました。"},
    {"surface": "1匹",  "kana": "イッピキ",       "wrong": ["イチヒキ"],           "category": "counter",
     "text": "水槽で小さな金魚を1匹飼っています。"},
    {"surface": "3匹",  "kana": "サンビキ",       "wrong": ["サンヒキ"],           "category": "counter",
     "text": "公園の隅で子猫が3匹遊んでいました。"},
    {"surface": "6匹",  "kana": "ロッピキ",       "wrong": ["ロクヒキ"],           "category": "counter",
     "text": "その家では犬を6匹も飼っているそうです。"},
    {"surface": "1杯",  "kana": "イッパイ",       "wrong": ["イチハイ"],           "category": "counter",
     "text": "朝は必ずコーヒーを1杯飲みます。"},
    {"surface": "3杯",  "kana": "サンバイ",       "wrong": ["サンハイ"],           "category": "counter",
     "text": "彼はご飯を3杯もおかわりしました。"},
    {"surface": "1階",  "kana": "イッカイ",       "wrong": ["イチカイ"],           "category": "counter",
     "text": "総合受付は建物の1階にあります。"},
    {"surface": "6階",  "kana": "ロッカイ",       "wrong": ["ロクカイ"],           "category": "counter",
     "text": "会議室はこのビルの6階にあります。"},
    {"surface": "1個",  "kana": "イッコ",         "wrong": ["イチコ"],             "category": "counter",
     "text": "夕食の後にりんごを1個だけ食べました。"},
    {"surface": "6個",  "kana": "ロッコ",         "wrong": ["ロクコ"],             "category": "counter",
     "text": "ケーキを焼くために卵を6個使いました。"},
    {"surface": "1歳",  "kana": "イッサイ",       "wrong": ["イチサイ"],           "category": "counter",
     "text": "息子は先週1歳の誕生日を迎えました。"},
    {"surface": "8歳",  "kana": "ハッサイ",       "wrong": ["ハチサイ"],           "category": "counter",
     "text": "娘は今年の誕生日で8歳になります。"},
    {"surface": "1冊",  "kana": "イッサツ",       "wrong": ["イチサツ"],           "category": "counter",
     "text": "旅行に出かける前に文庫本を1冊買いました。"},
    {"surface": "8冊",  "kana": "ハッサツ",       "wrong": ["ハチサツ"],           "category": "counter",
     "text": "今月は仕事の合間に本を8冊読みました。"},
    {"surface": "1回",  "kana": "イッカイ",       "wrong": ["イチカイ"],           "category": "counter",
     "text": "このジムには週に1回のペースで通っています。"},
    {"surface": "6回",  "kana": "ロッカイ",       "wrong": ["ロクカイ"],           "category": "counter",
     "text": "彼は今シーズンだけで6回もホームランを打った。"},
    # 人数
    {"surface": "1人",  "kana": "ヒトリ",         "wrong": ["イチニン"],           "category": "person",
     "text": "閉店後の会場には係員が1人だけ残っていた。"},
    {"surface": "2人",  "kana": "フタリ",         "wrong": ["ニニン"],             "category": "person",
     "text": "兄弟2人で駅前の小さな店を営んでいます。"},
    {"surface": "4人",  "kana": "ヨニン",         "wrong": ["ヨンニン", "シニン"], "category": "person",
     "text": "この席のテーブルには4人まで座れます。"},
    # 金額（連濁）
    {"surface": "300円",  "kana": "サンビャクエン", "wrong": ["サンヒャクエン"],   "category": "money",
     "text": "そのジュースは自動販売機で300円で売られています。"},
    {"surface": "600円",  "kana": "ロッピャクエン", "wrong": ["ロクヒャクエン"],   "category": "money",
     "text": "平日のランチは全品600円になっています。"},
    {"surface": "800円",  "kana": "ハッピャクエン", "wrong": ["ハチヒャクエン"],   "category": "money",
     "text": "美術館の入場料は大人800円です。"},
    {"surface": "3000円", "kana": "サンゼンエン",   "wrong": ["サンセンエン"],     "category": "money",
     "text": "この専門書は3000円もしました。"},
    # 年号・期間
    {"surface": "2026年", "kana": "ニセンニジュウロクネン", "wrong": [],           "category": "year",
     "text": "この橋は2026年に完成する予定です。"},
    {"surface": "1日",  "kana": "イチニチ",       "wrong": ["ツイタチ"],           "category": "duration",
     "text": "その修理の作業には丸1日かかりました。"},
]

# --- Set B: 読み分け制御型（両読みとも妥当な同表層） -------------------------
ITEMS_B = [
    {"surface": "10分", "readings": ["ジュップン", "ジップン"],   "category": "free_variation",
     "text": "次の休憩は10分だけ取ることにします。"},
    {"surface": "7時",  "readings": ["シチジ", "ナナジ"],         "category": "free_variation",
     "text": "毎朝7時に家を出て会社へ向かいます。"},
    {"surface": "3階",  "readings": ["サンガイ", "サンカイ"],     "category": "free_variation",
     "text": "書店はこのデパートの3階にあります。"},
    {"surface": "20歳", "readings": ["ハタチ", "ニジュッサイ"],   "category": "free_variation",
     "text": "彼は先月20歳の誕生日を迎えました。"},
    {"surface": "0度",  "readings": ["レイド", "ゼロド"],         "category": "free_variation",
     "text": "今朝の気温は0度まで下がりました。"},
]


def build_pairs():
    """eval_utmos_speaker_fixed.py --pairs_module 用のペア列を返す.

    Set A: 正読みカナを注入する1ペア/項目（BLの誤読特定と修正率測定に使用）
    Set B: 各読みを注入する2ペア/項目
    """
    pairs = []
    idx = 0
    for it in ITEMS_A:
        pairs.append({
            "pair_idx": idx, "set": "A", "category": it["category"],
            "word_a": it["surface"], "word_b": it["kana"],
            "text_a": it["text"], "text_b": it["text"].replace(it["surface"], it["kana"]),
            "correct_kana": it["kana"], "wrong_kana": it["wrong"],
            "accept_also": it.get("accept_also", []),
        })
        idx += 1
    for it in ITEMS_B:
        for kana in it["readings"]:
            others = [r for r in it["readings"] if r != kana]
            pairs.append({
                "pair_idx": idx, "set": "B", "category": it["category"],
                "word_a": it["surface"], "word_b": kana,
                "text_a": it["text"], "text_b": it["text"].replace(it["surface"], kana),
                "correct_kana": kana, "wrong_kana": others,
                "accept_also": [],
            })
            idx += 1
    return pairs


def validate():
    """自己検証: 設計原則への適合を機械チェックする."""
    import re
    errors = []
    pairs = build_pairs()
    for p in pairs:
        t, s = p["text_a"], p["word_a"]
        if t.count(s) != 1:
            errors.append(f"[{p['pair_idx']}] surface '{s}' が文中に {t.count(s)} 回")
        digits_outside = re.sub(re.escape(s), "", t)
        if re.search(r"[0-9０-９]", digits_outside):
            errors.append(f"[{p['pair_idx']}] surface以外に数字が残存: {digits_outside}")
        if len(t) < 15:
            errors.append(f"[{p['pair_idx']}] 文が短すぎる ({len(t)}字): {t}")
        for k in [p["word_b"]] + p["wrong_kana"] + p["accept_also"]:
            if k and not re.fullmatch(r"[ァ-ヶー]+", k):
                errors.append(f"[{p['pair_idx']}] カタカナ以外を含む読み: {k}")
        if p["word_b"] not in p["text_b"]:
            errors.append(f"[{p['pair_idx']}] text_b に注入読みが無い")
    print(f"pairs: {len(pairs)} (A={sum(1 for p in pairs if p['set']=='A')}, "
          f"B={sum(1 for p in pairs if p['set']=='B')})")
    if errors:
        print("VALIDATION ERRORS:")
        for e in errors:
            print(" ", e)
    else:
        print("validation OK")
    return not errors


if __name__ == "__main__":
    validate()
    for p in build_pairs()[:3]:
        print(p)
