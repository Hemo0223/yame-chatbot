# yame-chatbot
현재 제작 중인 야매 챗봇, kogpt2 기반

#답변추론 버전

'''

//                     _ooOoo_
//                    o8888888o
//                    88" . "88
//                    (| -_- |)
//                    O\  =  /O
//                 ____/`---'\____
//               .'  \\|     |//  `.
//              /  \\|||  :  |||//  \
//             /  _||||| -:- |||||-  \
//             |   | \\\  -  /// |   |
//             | \_|  ''\---/''  |   |
//             \  .-\__  `-`  ___/-. /
//           ___`. .'  /--.--\  `. . __
//        ."" '<  `.___\_<|>_/___.'  >'"".
//       | | :  `- \`.;`\ _ /`;.`/ - ` : | |
//       \  \ `-.   \_ __\ /__ _/   .-` /  /
//  ======`-.____`-.___\_____/___.-`____.-'======
//                     `=---='
//
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//             Buddha Bless:  'No Bugs'
//
//        揭諦揭諦 波羅揭諦 波羅僧揭諦 菩提娑婆訶
//

'''


#라이브러리 - 모델, 토크나이저
import torch, os
import json, re
import random
from difflib import SequenceMatcher 
from transformers import GPT2LMHeadModel, AutoTokenizer



#모델경로 설정하기
model_path = "./my_kogpt2" if os.path.isdir("./my_kogpt2") else "skt/kogpt2-base-v2"

#모델과 토크나이저, 데이터 불러오기 - 한국어 gpt2 소형모델
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
model = GPT2LMHeadModel.from_pretrained(model_path)

#padtoken 보정하기
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

#gpt 환경: 디바이스 = cpu
device = "cpu"
model.to(device)
model.eval() #추론모드 추가


#간단한 챗봇 반복문 구현, 종료 입력 시 프로그램 종료
print("=== 실험용 챗봇 ===")
print("종료 시 '종료' 입력\n")


#정규화/유사도/데이터 로딩
def normalize_text(s: str) -> str:
    s = s.strip().lower() #공백정리 겸 특수문자 최소화
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s가-힣]", " ", s)
    return s


#패턴 우선
def load_pairs(file_path):
    pairs = []
    if not os.path.isfile(file_path):
        return pairs
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            u = item.get("user")
            b = item.get("bot")
            if u and b:
                pairs.append((normalize_text(u), b, u))
    return pairs

#bot-only 데이터 로드
def load_bot_only(file_path):
    bot_only = []
    if not os.path.isfile(file_path):
        return bot_only
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            b = item.get("bot")
            if b and "user" not in item:
                bot_only.append(b)
    return bot_only

bot_only_data = load_bot_only("chat.jsonl")

#답 찾기
def find_best_reply(user_input, pairs, sim_threshold=0.80):
    if not pairs:
        return None
    q = normalize_text(user_input)

    #정확함 우선
    for u_norm, b, _ in pairs:
        if q == u_norm:
            return b

    #유사한 거
    best_b, best_score = None, 0.0
    for u_norm, b, _ in pairs:
        score = SequenceMatcher(None, q, u_norm).ratio()
        if score > best_score:
            best_score, best_b = score, b
    return best_b if best_score >= sim_threshold else None


#학습 데이터 로드하기
pair_data = load_pairs("chat.jsonl")


# === 이스터에그 ===
easter_eggs = {
    "java": "揭諦揭諦 波羅揭諦 波羅僧揭諦 菩提娑婆訶",
    "열려라 참깨": "쿠구구궁 끼이이이이이익",
    "앎은 한정되어 있지만": "무지에는 끝이 없다",
    "인류는": "영원 무한의 시공간에 파묻힌 하나의 점, 지구를 보금자리 삼아 살아가고 있다",
    "무한 우주에 순간의 빛일지라도": "달을 향해 쏴라, 빗나가도 별이 될 테니",
    "shoot for the moon": "even if you miss, you'll land among the stars"
}


#맥락 초기화
context = ""


#대화 맥락
while True:
    user_input = input("user: ")
    if user_input == "종료":
        print("bot: 즐거웠어!")
        break

    #이스터에그 출력하기 - 공백 및 대소문자 무시
    is_easter_eggs = False
    check_key = user_input.strip().lower()
    if check_key in easter_eggs:
        response_text = easter_eggs[check_key]
        is_easter_eggs = True
        skip_filter = True

    else:
        #학습시킨 데이터 내에서 답 찾기
        rule_reply = find_best_reply(user_input, pair_data, sim_threshold=0.75)
        skip_filter = False

        if rule_reply is not None:
            response_text = rule_reply

        else: #자료 내에 없으면 생성, bot_only or gpt, 기본확률 30%, 맥락 길어지면 70%까지 증가
            base_prob = 0.4
            extra = min(len(context.split("\n")) / 40, 0.3)
            bot_only_prob = base_prob + extra

            use_bot_only = bot_only_data and random.random() < bot_only_prob

            if use_bot_only:
                response_text = random.choice(bot_only_data)
            else:
                prompt = (f"이는 한국어 챗봇과의 대화입니다. 모든 답변은 한국어로만 작성하세요.\n{context}\nuser: {user_input}\nbot:")
                inputs = tokenizer.encode(prompt, return_tensors = "pt").to(device)


                #모델 응답 생성하기
                with torch.inference_mode():
                    output_ids = model.generate(
                        inputs,
                        max_new_tokens=80,
                        do_sample=True,
                        top_k=50,
                        top_p=0.9,
                        temperature=0.7,
                        no_repeat_ngram_size=2,
                        pad_token_id = tokenizer.pad_token_id
                    )

                full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                response_text = full_text.split("bot:")[-1].split("user:")[0].strip() if "bot:" in full_text else full_text.strip()


    #한국어 필터링 및 공백
    if not skip_filter:
        response_text = re.sub(r"[^가-힣0-9\s,.?!]", "", response_text)
        response_text = re.sub(r"\s+", " ", response_text).strip()

        if not response_text:
            response_text = "다시 말해줘"

    if is_easter_eggs:
        print(response_text)
    else:
        print("bot:", response_text)


    #맥락 업데이트, 최근 4턴 정도
    context += f"\nuser: {user_input}\nbot:{response_text}"
    context_lines = context.split("\n")
    if len(context_lines) > 16:
        context = "\n".join(context_lines[-16:])
