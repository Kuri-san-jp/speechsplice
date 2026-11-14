#!/usr/bin/env python3
"""CJK N=300 accuracy + quality evaluation.

Same design as eval_eu_quality_accuracy.py:
- N=300 unique pairs per language, seed=42
- BL synthesis + OEOI injection + Whisper ASR + UTMOS22
- Checkpoint/resume support

Languages:
  JA: kanji entity → katakana reading injection (from entities_10000.jsonl)
  ZH: polyphone character, reading_a context → reading_b context injection
  KO: Hangul word pair injection (word_a → word_b)

Usage:
  SLURM_TIME=16:00:00 slurm_run.sh \
    python scripts/projection/eval_cjk_n300.py [--lang ja|zh|ko|all]
"""
import sys, os, json, gc, time, random, argparse, traceback
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
T5GEMMA_ROOT = os.environ.get("T5GEMMA_ROOT", os.path.join(os.environ.get("REPO_ROOT", "/mnt"), "src", "_vendor", "t5gemma_tts"))
sys.path.insert(0, T5GEMMA_ROOT)
import torch
import torchaudio

PROJECT_ROOT = "/mnt" if os.path.isdir("/mnt/scripts") else os.environ.get("REPO_ROOT", "/mnt")

# コンテナ内でも TTS26 ローカルの HF キャッシュを使う（他スクリプトと同じ対処）
_hf_home = os.path.join(PROJECT_ROOT, ".cache", "huggingface")
os.environ["HF_HOME"] = _hf_home
os.environ["HF_HUB_CACHE"] = os.path.join(_hf_home, "hub")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

SEED = 42
N_ACCURACY = 300
N_QUALITY = 100

# ============================================================
# JA data: load from entities_10000.jsonl
# ============================================================

def load_ja_pairs(n):
    """Load N unique (kanji, katakana) entity pairs for JA."""
    path = os.path.join(PROJECT_ROOT, "data/entities_10000.jsonl")
    entities = []
    with open(path) as f:
        for line in f:
            e = json.loads(line)
            if e.get("reading_kata") and e["surface"] != e["reading_kata"]:
                entities.append(e)
    rng = random.Random(SEED)
    rng.shuffle(entities)
    pairs = []
    for e in entities[:n]:
        pairs.append({
            "word_a": e["surface"],      # kanji
            "word_b": e["reading_kata"],  # katakana
            "template_a": "{word}は、日本で広く知られています。",
            "template_b": "{word}は、日本で広く知られています。",
        })
    assert len(pairs) >= n, f"JA: need {n} pairs, got {len(pairs)}"
    return pairs[:n]


# ============================================================
# ZH data: polyphone characters
# ============================================================

ZH_POLYPHONES = [
    {"char": "行",
     "ctxs_a": ["中国银行是一家大型金融机构。", "这个行业的竞争非常激烈。",
                 "他们排成了整齐的行列。", "各行各业都在快速发展。", "银行的服务越来越便捷了。"],
     "ctxs_b": ["他每天坚持行走五公里。", "我们必须立即采取行动。",
                 "良好的行为是成功的基础。", "这个计划很快就会执行。", "行人应该遵守交通规则。"]},
    {"char": "长",
     "ctxs_a": ["万里长城是中国的标志。", "这条河流非常长。",
                 "长期坚持才能看到效果。", "他的头发变得很长了。", "长途旅行需要充分准备。"],
     "ctxs_b": ["校长今天发表了重要讲话。", "孩子们在不断成长。",
                 "他是公司的部长。", "树木在春天开始生长。", "班长负责管理班级事务。"]},
    {"char": "重",
     "ctxs_a": ["这个箱子非常重。", "他在比赛中受了重伤。",
                 "体重管理对健康很重要。", "这是一项重大的发现。", "重工业是国家经济的支柱。"],
     "ctxs_b": ["请重新检查一下这份文件。", "重复练习是学习的关键。",
                 "这座山有九重宫殿。", "重庆是中国的直辖市。", "他们重建了被毁的桥梁。"]},
    {"char": "还",
     "ctxs_a": ["他还没有完成作业。", "天气还不错。",
                 "除了英语他还会说法语。", "这道菜还需要加点盐。", "他还是个学生。"],
     "ctxs_b": ["请把书还给图书馆。", "借的钱必须按时归还。",
                 "他终于把欠款还清了。", "还书的截止日期是明天。", "以牙还牙不是好办法。"]},
    {"char": "乐",
     "ctxs_a": ["他是一个快乐的孩子。", "助人为乐是传统美德。",
                 "大家都乐于参加活动。", "乐观的态度很重要。", "他对生活充满了欢乐。"],
     "ctxs_b": ["他喜欢听古典音乐。", "音乐老师教大家唱歌。",
                 "这首乐曲非常优美。", "乐器的种类有很多。", "乐队在舞台上表演。"]},
    {"char": "觉",
     "ctxs_a": ["我觉得这个方案不错。", "他觉得很开心。",
                 "大家都觉得应该改进。", "你觉得怎么样。", "我不觉得这是问题。"],
     "ctxs_b": ["他已经睡觉了。", "午觉对健康有益。",
                 "孩子们正在睡午觉。", "一觉醒来天已经亮了。", "他的警觉性很高。"]},
    {"char": "差",
     "ctxs_a": ["两者之间有很大的差异。", "成绩的差距越来越小。",
                 "温差很大要注意保暖。", "偏差可能导致严重后果。", "时差让他很不适应。"],
     "ctxs_b": ["他被派去出差了。", "领导差人送来了文件。",
                 "今天差事很多。", "他差点就迟到了。", "这份工作差强人意。"]},
    {"char": "调",
     "ctxs_a": ["这首曲子的调子很高。", "他说话的声调很好听。",
                 "曲调优美动人。", "这个音调不太准。", "语调可以表达情感。"],
     "ctxs_b": ["请调整一下温度。", "他被调到了新部门。",
                 "需要调查事实真相。", "老师在调解学生纠纷。", "调动工作积极性很重要。"]},
    {"char": "数",
     "ctxs_a": ["数学是一门重要学科。", "这个数字很大。",
                 "参加人数超过了预期。", "他的分数非常高。", "无数星星在天上闪烁。"],
     "ctxs_b": ["老师在数学生的人数。", "请数一数有多少个。",
                 "他在数钱。", "她数着手指做计算。", "倒数计时开始了。"]},
    {"char": "干",
     "ctxs_a": ["衣服已经晒干了。", "河床在夏天干涸了。",
                 "干燥的天气让人不舒服。", "他把眼泪擦干了。", "干旱给农业带来损失。"],
     "ctxs_b": ["他干活很卖力。", "大家一起干吧。",
                 "干部要以身作则。", "这件事得赶紧干。", "他是一个能干的人。"]},
    {"char": "发",
     "ctxs_a": ["她的头发很漂亮。", "他的白发越来越多。",
                 "理发师的手艺很好。", "她把头发剪短了。", "毛发生长需要营养。"],
     "ctxs_b": ["公司发展很快。", "他发现了一个问题。",
                 "学校发通知了。", "火箭即将发射。", "他发表了重要演讲。"]},
    {"char": "假",
     "ctxs_a": ["这个消息是假的。", "真假难辨。",
                 "假设条件不成立。", "他做了一个假动作。", "假冒商品要严厉打击。"],
     "ctxs_b": ["暑假就要到了。", "他请了两天假。",
                 "放假大家都很开心。", "寒假计划去旅行。", "休假是为了更好地工作。"]},
    {"char": "背",
     "ctxs_a": ["他背着一个大书包。", "背上的伤还没好。",
                 "她把孩子背在背上。", "他背对着窗户站着。", "驼背需要矫正。"],
     "ctxs_b": ["学生们在背诵课文。", "背单词是学英语的基础。",
                 "他能背出整篇文章。", "背书要理解意思。", "她在背台词。"]},
    {"char": "种",
     "ctxs_a": ["这种方法很有效。", "有很多种选择。",
                 "各种各样的花都开了。", "人种的差异是自然的。", "品种改良提高了产量。"],
     "ctxs_b": ["农民在田里种地。", "他种了很多花。",
                 "春天是种树的好时节。", "种下希望的种子。", "种菜可以自给自足。"]},
    {"char": "相",
     "ctxs_a": ["这两个方案相似。", "互相帮助很重要。",
                 "相信自己能做到。", "相比之下这个更好。", "不能以貌相人。"],
     "ctxs_b": ["照相机的价格下降了。", "他看了一下面相。",
                 "宰相肚里能撑船。", "相声是传统艺术。", "她长得很有福相。"]},
    {"char": "量",
     "ctxs_a": ["这个项目的工作量很大。", "质量比数量更重要。",
                 "他的饭量很大。", "容量已经满了。", "能量守恒定律。"],
     "ctxs_b": ["请量一下身高。", "护士在量体温。",
                 "裁缝在量尺寸。", "他量了一下距离。", "测量数据要准确。"]},
    {"char": "藏",
     "ctxs_a": ["西藏是一个美丽的地方。", "这是珍贵的收藏品。",
                 "宝藏被发现了。", "图书馆的藏书很丰富。", "他有很多私人珍藏。"],
     "ctxs_b": ["他藏在了树后面。", "把秘密藏在心里。",
                 "猫藏起来了。", "他把礼物藏好了。", "不要躲藏了。"]},
    {"char": "弹",
     "ctxs_a": ["子弹的速度很快。", "炸弹被安全拆除了。",
                 "导弹防御系统很先进。", "原子弹的威力巨大。", "他中了一颗流弹。"],
     "ctxs_b": ["她在弹钢琴。", "弹吉他是他的爱好。",
                 "弹性工作制很受欢迎。", "孩子们在弹弹珠。", "这个弹簧很有弹力。"]},
    {"char": "传",
     "ctxs_a": ["消息很快传开了。", "传统文化要继承。",
                 "传说中的故事很精彩。", "信号传输速度很快。", "疾病可以通过空气传播。"],
     "ctxs_b": ["这是一本人物传记。", "自传记录了他的一生。",
                 "列传是史记的重要部分。", "水浒传是中国名著。", "这部外传写得很好。"]},
    {"char": "教",
     "ctxs_a": ["教育是国家的根本。", "老师在教室里上课。",
                 "教学方法要与时俱进。", "教练指导运动员训练。", "教材的内容很丰富。"],
     "ctxs_b": ["佛教在亚洲影响很大。", "基督教传入了中国。",
                 "道教是中国本土宗教。", "宗教信仰要尊重。", "教堂的建筑很壮观。"]},
    {"char": "空",
     "ctxs_a": ["天空很蓝。", "空气质量有所改善。",
                 "太空探索是人类的梦想。", "空间站在轨道上运行。", "空调让房间很凉爽。"],
     "ctxs_b": ["他有空的时候喜欢看书。", "空闲时间不多。",
                 "这个座位是空的。", "仓库已经空了。", "别空着手来。"]},
    {"char": "落",
     "ctxs_a": ["太阳落山了。", "树叶纷纷落下。",
                 "成绩有所下落。", "瀑布从山上落下。", "他落在了后面。"],
     "ctxs_b": ["这个村落很古老。", "部落的文化很独特。",
                 "角落里放着一把椅子。", "院落整洁干净。", "他很落落大方。"]},
    {"char": "模",
     "ctxs_a": ["模型做得很精致。", "模式需要创新。",
                 "规模越来越大。", "模仿能力很强。", "模拟考试很重要。"],
     "ctxs_b": ["模样长得很可爱。", "模具的精度很高。",
                 "这个模子做得不错。", "橡皮模很有弹性。", "他的模样变了。"]},
    {"char": "似",
     "ctxs_a": ["这两个人长得很相似。", "类似的问题已经出现过。",
                 "似乎要下雨了。", "近似值可以接受。", "相似度很高。"],
     "ctxs_b": ["似的用法在口语中很常见。", "他跑得飞快似的。",
                 "好像要下雨似的。", "他笑得像个孩子似的。", "天黑得像晚上似的。"]},
    {"char": "间",
     "ctxs_a": ["房间很宽敞。", "时间过得很快。",
                 "空间利用要合理。", "中间位置最好。", "期间发生了很多事。"],
     "ctxs_b": ["离间他们的关系。", "间谍电影很刺激。",
                 "反间计是一种策略。", "间隔要均匀。", "黑白相间很好看。"]},
]
assert len(ZH_POLYPHONES) == 25, f"Expected 25, got {len(ZH_POLYPHONES)}"


def load_zh_pairs(n):
    """Load N unique ZH polyphone context pairs.

    Each pair: (context_a_sentence, context_b_sentence, polyphone_char).
    We generate cross-product pairs from ctxs_a × ctxs_b for each character,
    then sample n_per_char pairs per character.
    """
    rng = random.Random(SEED)
    n_chars = len(ZH_POLYPHONES)
    n_per_char = n // n_chars  # 300 // 25 = 12
    remainder = n % n_chars

    pairs = []
    for ci, poly in enumerate(ZH_POLYPHONES):
        char = poly["char"]
        cross = [(a, b) for a in poly["ctxs_a"] for b in poly["ctxs_b"]]  # 25 combos
        rng.shuffle(cross)
        take = n_per_char + (1 if ci < remainder else 0)
        for ctx_a, ctx_b in cross[:take]:
            pairs.append({
                "word_a": char,  # the polyphone character (entity in context_a)
                "word_b": char,  # same character, different reading in context_b
                "text_a": ctx_a,
                "text_b": ctx_b,
            })
    rng.shuffle(pairs)
    assert len(pairs) == n, f"ZH: expected {n}, got {len(pairs)}"
    return pairs


# ============================================================
# KO data: Hangul word pairs (from eval_ko_hangul_n1000.py)
# ============================================================

KO_WORDS_A = [
    "사람", "뛰어가기", "나라", "물", "바람", "달", "해", "구름", "비", "눈",
    "안개", "서리", "우박", "번개", "무지개", "산", "강", "바다", "호수", "섬",
    "숲", "초원", "사막", "계곡", "폭포", "동굴", "절벽", "언덕", "평야", "해안",
    "밤", "아침", "저녁", "새벽", "황혼", "자정", "정오", "오후", "오전", "일출",
    "일몰", "봄", "여름", "가을", "겨울", "꽃", "나무", "풀", "잎", "뿌리",
    "열매", "씨앗", "줄기", "가지", "꽃잎", "이슬", "파도", "조류", "해류", "안개",
    "소나기", "태풍", "가뭄", "홍수", "지진", "화산", "토양", "암석", "모래", "진흙",
    "돌", "쇠", "금", "은", "구리", "사과", "배", "감", "밤", "포도",
    "딸기", "수박", "참외", "복숭아", "자두", "살구", "앵두", "석류", "귤", "레몬",
    "고양이", "강아지", "토끼", "거북이", "다람쥐", "참새", "까치", "비둘기", "독수리", "백조",
]
KO_WORDS_B = [
    "과학", "역사", "수학", "경제", "정치", "종교", "도서관", "박물관", "미술관", "편의점",
    "주유소", "병원", "약국", "학교", "대학", "유치원", "경찰서", "소방서", "우체국", "은행",
    "시장", "공원", "극장", "체육관", "수영장", "도로", "철도", "항구", "공항", "터널",
    "교량", "댐", "탑", "성", "궁전", "사찰", "교회", "모스크", "성당", "신사",
    "농업", "어업", "공업", "상업", "무역", "관광", "교육", "의료", "법률", "건축",
    "음악", "미술", "문학", "영화", "연극", "무용", "사진", "조각", "도자기", "서예",
    "축구", "야구", "농구", "배구", "테니스", "탁구", "골프", "수영", "달리기", "체조",
    "컴퓨터", "인터넷", "스마트폰", "텔레비전", "라디오", "신문", "잡지", "편지", "이메일", "전화",
    "지도자", "철학자", "과학자", "예술가", "음악가", "작가", "시인", "화가", "건축가", "의사",
    "선생님", "간호사", "요리사", "농부", "어부", "군인", "경찰관", "소방관", "기자", "변호사",
]

KO_TEMPLATE = "{word}는 한국에서 흔히 사용됩니다."


def load_ko_pairs(n):
    """Load N unique Hangul word pairs."""
    rng = random.Random(SEED)
    pairs = []
    used = set()
    # Sequential
    for i in range(min(n, len(KO_WORDS_A), len(KO_WORDS_B))):
        a, b = KO_WORDS_A[i], KO_WORDS_B[i]
        if a != b and (a, b) not in used:
            pairs.append({"word_a": a, "word_b": b})
            used.add((a, b))
    # Random fill
    while len(pairs) < n:
        a = rng.choice(KO_WORDS_A)
        b = rng.choice(KO_WORDS_B)
        if a != b and (a, b) not in used:
            pairs.append({"word_a": a, "word_b": b})
            used.add((a, b))
    return pairs[:n]


# ============================================================
# Shared functions
# ============================================================

def load_whisper(device):
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    model_name = "openai/whisper-large-v3"
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(os.environ["HF_HOME"], "hub")
    repo_dir = os.path.join(hub, f"models--{model_name.replace('/', '--')}", "snapshots")
    if os.path.isdir(repo_dir):
        snaps = sorted(os.listdir(repo_dir))
        if snaps:
            model_name = os.path.join(repo_dir, snaps[-1])
    print(f"[Info] Loading Whisper ({model_name})...")
    processor = WhisperProcessor.from_pretrained(model_name, local_files_only=True)
    wm = WhisperForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16, local_files_only=True).to(device)
    wm.eval()
    return wm, processor


def transcribe(audio_tensor, src_sr, whisper_model, processor, lang):
    if audio_tensor.dim() > 1:
        audio_tensor = audio_tensor.squeeze(0)
    if src_sr != 16000:
        audio_tensor = torchaudio.functional.resample(audio_tensor, src_sr, 16000)
    feats = processor(audio_tensor.numpy(), sampling_rate=16000, return_tensors="pt")
    input_features = feats.input_features.to(
        whisper_model.device, dtype=next(whisper_model.parameters()).dtype)
    lang_map = {"ja": "japanese", "zh": "chinese", "ko": "korean"}
    forced = processor.get_decoder_prompt_ids(language=lang_map.get(lang, lang), task="transcribe")
    with torch.no_grad():
        pred = whisper_model.generate(input_features, forced_decoder_ids=forced, max_new_tokens=256)
    return processor.batch_decode(pred, skip_special_tokens=True)[0].strip()


def load_utmos(device):
    print("[Info] Loading UTMOS22...")
    model = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(device)
    model.eval()
    return model


def compute_utmos(audio_tensor, src_sr, utmos_model, device):
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    if src_sr != 16000:
        audio_tensor = torchaudio.functional.resample(audio_tensor, src_sr, 16000)
    with torch.no_grad():
        score = utmos_model(audio_tensor.to(device), 16000)
    return score.item()


def check_word_in_transcript(word, transcript, lang):
    w = word.lower()
    t = transcript.lower()
    return w in t


def save_checkpoint(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def load_checkpoint(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="CJK N=300 Accuracy + Quality")
    parser.add_argument("--lang", default="all", choices=["ja", "zh", "ko", "all"])
    parser.add_argument("--output_dir", default="eval_results/cjk_n300")
    parser.add_argument("--n_accuracy", type=int, default=N_ACCURACY)
    parser.add_argument("--n_quality", type=int, default=N_QUALITY)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save_wavs", action="store_true",
                        help="BL/OEOI 音声を output_dir に保存する（既定は保存しない=従来動作）")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    device = args.device
    os.makedirs(args.output_dir, exist_ok=True)

    languages = ["ja", "zh", "ko"] if args.lang == "all" else [args.lang]

    print(f"[Config] Languages: {languages}")
    print(f"[Config] N_accuracy={args.n_accuracy}, N_quality={args.n_quality}, Seed={SEED}")

    # Load models
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
    from data.tokenizer import AudioTokenizer
    from encoder_injection_t5gemma import (
        get_encoder_output, generate_with_modified_encoder,
        find_entity_token_range, seed_everything,
    )

    def _local_snapshot(hub_id):
        # 型番名のままだと通信を試みるライブラリ経路があるため、保存済みの
        # 実体フォルダを直接指す（他スクリプトと同じ対処）
        hub = os.environ.get("HF_HUB_CACHE") or os.path.join(os.environ["HF_HOME"], "hub")
        repo_dir = os.path.join(hub, f"models--{hub_id.replace('/', '--')}", "snapshots")
        if os.path.isdir(repo_dir):
            snaps = sorted(os.listdir(repo_dir))
            if snaps:
                return os.path.join(repo_dir, snaps[-1])
        return hub_id

    print("[Info] Loading TTS model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        _local_snapshot("Aratako/T5Gemma-TTS-2b-2b"), trust_remote_code=True,
        dtype=torch.bfloat16, device_map="auto", local_files_only=True)
    model.eval()
    cfg = model.config
    tok_name = getattr(cfg, "text_tokenizer_name", None) or getattr(cfg, "t5gemma_model_name", None)
    text_tokenizer = AutoTokenizer.from_pretrained(_local_snapshot(tok_name), local_files_only=True)
    audio_tokenizer = AudioTokenizer(
        backend="xcodec2", model_name=getattr(cfg, "xcodec2_model_name", None))
    codec_sr = audio_tokenizer.sample_rate
    codec_fps = getattr(cfg, "encodec_sr", 50)
    add_bos = 1 if int(getattr(model.args, "add_bos_to_text", 0)) else 0

    whisper_model, whisper_proc = load_whisper(device)
    utmos_model = load_utmos(device)

    # --------------------------------------------------------
    # Per-language evaluation
    # --------------------------------------------------------

    for lang in languages:
        print(f"\n{'='*70}")
        print(f"LANGUAGE: {lang.upper()} (N={args.n_accuracy})")
        print(f"{'='*70}")

        # Load pairs
        if lang == "ja":
            raw_pairs = load_ja_pairs(args.n_accuracy)
        elif lang == "zh":
            raw_pairs = load_zh_pairs(args.n_accuracy)
        elif lang == "ko":
            raw_pairs = load_ko_pairs(args.n_accuracy)

        print(f"[Info] {len(raw_pairs)} pairs loaded")

        # Checkpoint
        ckpt_path = os.path.join(args.output_dir, f"checkpoint_{lang}.json")
        ckpt = load_checkpoint(ckpt_path)
        results = []
        done_indices = set()
        if ckpt:
            results = ckpt.get("results", [])
            done_indices = {r["pair_idx"] for r in results}
            print(f"[Resume] {len(done_indices)} done")

        lang_start = time.time()

        for pair_idx, pair in enumerate(raw_pairs):
            if pair_idx in done_indices:
                continue

            compute_quality = (pair_idx < args.n_quality)
            word_a = pair["word_a"]
            word_b = pair["word_b"]

            result = {
                "pair_idx": pair_idx, "lang": lang,
                "word_a": word_a, "word_b": word_b,
                "status": "pending",
            }

            try:
                # === Determine texts ===
                if lang == "ja":
                    tmpl = pair["template_a"]
                    text_a = tmpl.format(word=word_a)
                    text_b = tmpl.format(word=word_b)
                elif lang == "zh":
                    text_a = pair["text_a"]
                    text_b = pair["text_b"]
                elif lang == "ko":
                    text_a = KO_TEMPLATE.format(word=word_a)
                    text_b = KO_TEMPLATE.format(word=word_b)

                result["text_a"] = text_a
                result["text_b"] = text_b

                # === BL synthesis ===
                seed_everything(SEED)
                enc_a = get_encoder_output(model, text_tokenizer, text_a, device)
                bl_gen = generate_with_modified_encoder(
                    model, text_tokenizer, audio_tokenizer,
                    enc_a["last_hidden_state"], enc_a["attention_mask"],
                    enc_a.get("position_ids"), text_a, device, codec_sr, codec_fps)
                if bl_gen is None:
                    result["status"] = "error"
                    result["error"] = "bl_no_audio"
                    results.append(result)
                    done_indices.add(pair_idx)
                    continue

                bl_audio = bl_gen[0].cpu()
                if args.save_wavs:
                    _w = bl_audio.detach().float()
                    _w = _w if _w.dim() > 1 else _w.unsqueeze(0)
                    torchaudio.save(os.path.join(
                        args.output_dir, f"s{pair_idx:04d}_bl.wav"), _w, codec_sr)
                bl_transcript = transcribe(bl_audio, codec_sr, whisper_model, whisper_proc, lang)

                # BL accuracy: does word_a appear in transcript?
                bl_correct = check_word_in_transcript(word_a, bl_transcript, lang)
                result["bl_transcript"] = bl_transcript
                result["bl_correct"] = bl_correct

                if compute_quality:
                    result["bl_utmos22"] = round(compute_utmos(bl_audio, codec_sr, utmos_model, device), 4)

                del bl_audio, bl_gen

                # === OEOI synthesis ===
                seed_everything(SEED)
                enc_b = get_encoder_output(model, text_tokenizer, text_b, device)

                # Find entity token ranges
                src_range = find_entity_token_range(
                    text_tokenizer, enc_a["text"], word_a, add_bos=add_bos)
                ref_range = find_entity_token_range(
                    text_tokenizer, enc_b["text"], word_b, add_bos=add_bos)

                if src_range is None or ref_range is None:
                    result["status"] = "error"
                    result["error"] = f"token_range_not_found src={src_range} ref={ref_range}"
                    results.append(result)
                    done_indices.add(pair_idx)
                    del enc_a, enc_b
                    continue

                src_start, src_end = src_range
                ref_start, ref_end = ref_range

                # Cosine similarity
                src_h = enc_a["last_hidden_state"][0, src_start:src_end].float().mean(0)
                ref_h = enc_b["last_hidden_state"][0, ref_start:ref_end].float().mean(0)
                cos_sim = torch.nn.functional.cosine_similarity(
                    src_h.unsqueeze(0), ref_h.unsqueeze(0), dim=-1).item()
                result["cos_sim"] = round(cos_sim, 4)

                # Splice
                modified = enc_a["last_hidden_state"].clone()
                ref_entity = enc_b["last_hidden_state"][0, ref_start:ref_end]
                src_len = src_end - src_start
                ref_len = ref_end - ref_start

                if src_len == ref_len:
                    modified[0, src_start:src_end] = ref_entity
                    mod_attn = enc_a["attention_mask"]
                else:
                    prefix = modified[0, :src_start]
                    suffix = modified[0, src_end:]
                    modified = torch.cat([prefix, ref_entity, suffix], dim=0).unsqueeze(0)
                    new_len = modified.shape[1]
                    mod_attn = torch.ones((1, new_len), dtype=torch.long, device=device)

                pos_ids = None
                if int(getattr(model.args, "use_pm_rope", 1)):
                    new_len = modified.shape[1]
                    x_lens = torch.LongTensor([new_len]).to(device)
                    pos_ids = model._build_position_ids(x_lens, new_len, device)

                seed_everything(SEED)
                oeoi_gen = generate_with_modified_encoder(
                    model, text_tokenizer, audio_tokenizer,
                    modified, mod_attn, pos_ids,
                    text_a, device, codec_sr, codec_fps)

                if oeoi_gen is None:
                    result["status"] = "error"
                    result["error"] = "oeoi_no_audio"
                    results.append(result)
                    done_indices.add(pair_idx)
                    del enc_a, enc_b, modified
                    continue

                oeoi_audio = oeoi_gen[0].cpu()
                if args.save_wavs:
                    _w = oeoi_audio.detach().float()
                    _w = _w if _w.dim() > 1 else _w.unsqueeze(0)
                    torchaudio.save(os.path.join(
                        args.output_dir, f"s{pair_idx:04d}_oeoi.wav"), _w, codec_sr)
                oeoi_transcript = transcribe(oeoi_audio, codec_sr, whisper_model, whisper_proc, lang)

                # OEOI accuracy: does word_b appear in OEOI transcript?
                oeoi_correct = check_word_in_transcript(word_b, oeoi_transcript, lang)
                result["oeoi_transcript"] = oeoi_transcript
                result["oeoi_correct"] = oeoi_correct
                result["changed"] = (bl_transcript != oeoi_transcript)

                # ASR gating
                if bl_correct:
                    gated_correct = check_word_in_transcript(word_b, bl_transcript, lang)
                    result["gated_source"] = "bl"
                else:
                    gated_correct = oeoi_correct
                    result["gated_source"] = "oeoi"
                result["gated_correct"] = gated_correct

                if compute_quality:
                    result["oeoi_utmos22"] = round(
                        compute_utmos(oeoi_audio, codec_sr, utmos_model, device), 4)

                del oeoi_audio, oeoi_gen, enc_a, enc_b, modified
                result["status"] = "success"

            except Exception as e:
                result["status"] = "error"
                result["error"] = str(e)
                print(f"  [{pair_idx}] ERROR: {e}")
                traceback.print_exc()

            results.append(result)
            done_indices.add(pair_idx)

            # Progress
            n_done = len(done_indices)
            elapsed = time.time() - lang_start
            if n_done % 10 == 0 or n_done <= 5:
                rate = n_done / max(elapsed, 1) * 3600
                print(f"  [{n_done}/{len(raw_pairs)}] {lang} "
                      f"a={word_a} b={word_b} "
                      f"BL={result.get('bl_correct','?')} "
                      f"OEOI={result.get('oeoi_correct','?')} "
                      f"cos={result.get('cos_sim','?')} "
                      f"({elapsed/60:.1f}min, {rate:.0f}/h)")

            if n_done % 20 == 0:
                save_checkpoint(ckpt_path, {"results": results})

            gc.collect()
            torch.cuda.empty_cache()

        # Final checkpoint
        save_checkpoint(ckpt_path, {"results": results})

        # Summary
        ok = [r for r in results if r["status"] == "success"]
        n_ok = len(ok)
        if n_ok > 0:
            bl_acc = sum(1 for r in ok if r.get("bl_correct")) / n_ok * 100
            oeoi_acc = sum(1 for r in ok if r.get("oeoi_correct")) / n_ok * 100
            gated_acc = sum(1 for r in ok if r.get("gated_correct")) / n_ok * 100
            changed = sum(1 for r in ok if r.get("changed")) / n_ok * 100
            cos_vals = [r["cos_sim"] for r in ok if "cos_sim" in r]
            cos_mean = sum(cos_vals) / len(cos_vals) if cos_vals else 0

            quality = [r for r in ok if "bl_utmos22" in r]
            bl_utmos = sum(r["bl_utmos22"] for r in quality) / len(quality) if quality else 0
            oeoi_utmos = sum(r["oeoi_utmos22"] for r in quality) / len(quality) if quality else 0

            print(f"\n--- {lang.upper()} SUMMARY (N={n_ok}) ---")
            print(f"  BL accuracy:    {bl_acc:.2f}%")
            print(f"  OEOI accuracy:  {oeoi_acc:.2f}%")
            print(f"  Gated accuracy: {gated_acc:.2f}%")
            print(f"  Changed:        {changed:.1f}%")
            print(f"  Cos sim:        {cos_mean:.4f}")
            print(f"  BL UTMOS22:     {bl_utmos:.4f} (N={len(quality)})")
            print(f"  OEOI UTMOS22:   {oeoi_utmos:.4f} (N={len(quality)})")
        else:
            print(f"\n--- {lang.upper()} SUMMARY: No successful results ---")

        elapsed = time.time() - lang_start
        print(f"  Time: {elapsed/60:.1f} min")

    print(f"\nTotal time: {(time.time() - lang_start)/60:.1f} min" if languages else "")
    print("DONE")


if __name__ == "__main__":
    main()
