import os, json, textwrap
import os
from pathlib import Path
from typing import List, Dict, Optional
import google.generativeai as genai
import requests, os
from typing import List, Dict, Optional

import numpy as np, faiss, torch, requests
from PIL import Image
from tqdm import tqdm
from torchvision import transforms, models
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn

# ===== 路径与常量（按需修改）=====
DATASET_ROOT = Path(r"c:\Users\final_dataset")  # 包含 images_byAI、json
JSON_DIR     = DATASET_ROOT / "json"
INDEX_DIR    = Path("./vecstore"); INDEX_DIR.mkdir(parents=True, exist_ok=True)
INDEX_PATH   = INDEX_DIR / "mm.index"
META_PATH    = INDEX_DIR / "meta.json"
CKPT_PATH    = Path("best_mm_fc.pt")   # 你训练后的权重
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

BERT_MODEL   = "bert-base-uncased"
IMG_BACKBONE = "resnet18"
MAX_LEN      = 64
IMG_SIZE     = 224
ALT_EXTS     = [".png",".jpg",".jpeg"]

# ===== 编码器（与你训练时一致）=====
class TextEncoder(nn.Module):
    def __init__(self, model_name=BERT_MODEL, finetune=False):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        if not finetune:
            for p in self.bert.parameters(): p.requires_grad = False
        self.out_dim = self.bert.config.hidden_size
    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return out.pooler_output if getattr(out,"pooler_output",None) is not None else out.last_hidden_state[:,0,:]

class ImageEncoder(nn.Module):
    def __init__(self, backbone=IMG_BACKBONE, finetune=False):
        super().__init__()
        if backbone == "resnet50":
            net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT); feat_dim = 2048
        else:
            net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT); feat_dim = 512
        self.cnn = nn.Sequential(*list(net.children())[:-1]); self.out_dim = feat_dim
        if not finetune:
            for p in self.cnn.parameters(): p.requires_grad = False
    def forward(self, x): return self.cnn(x).flatten(1)

# ===== 数据加载 & 过滤 =====
def load_records(json_dir: Path) -> List[Dict]:
    recs = []
    for fn in os.listdir(json_dir):
        if not fn.endswith(".json"): continue
        data = json.loads((json_dir/fn).read_text(encoding="utf-8"))
        if isinstance(data, list):
            for r in data:
                if {"prompt","image_path","task_type"} <= r.keys():
                    recs.append(r)
    if not recs: raise RuntimeError("未加载到样本")
    return recs

def _norm(s:str)->str: return (s or "").replace("\\","/").strip()
def _exists_any(p:Path)->bool:
    if p.exists(): return True
    stem = p.with_suffix("")
    return any((stem.with_suffix(ext)).exists() for ext in ALT_EXTS)

def filter_has_image(records: List[Dict], root: Path) -> List[Dict]:
    keep, drop = [], 0
    for r in records:
        rel = _norm(r.get("image_path",""))
        if rel and _exists_any(root/rel):
            r["image_path"] = rel; keep.append(r)
        else:
            drop += 1
    print(f"记录筛选：原始 {len(records)} → 保留 {len(keep)}，忽略无图 {drop}")
    return keep

# ===== 特征提取 & 索引 =====
def img_tf(size=IMG_SIZE):
    return transforms.Compose([
        transforms.Resize((size,size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

@torch.no_grad()
def enc_texts(tok, txt_enc, texts, device=DEVICE, max_len=MAX_LEN):
    enc = tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    enc = {k:v.to(device) for k,v in enc.items()}
    return txt_enc(enc["input_ids"], enc["attention_mask"]).cpu().numpy().astype("float32")

@torch.no_grad()
def enc_images(img_enc, paths, device=DEVICE):
    tf = img_tf(); arr=[]
    for p in paths:
        img = Image.open(p).convert("RGB"); arr.append(tf(img))
    x = torch.stack(arr,0).to(device)
    return img_enc(x).cpu().numpy().astype("float32")

def l2n(x): return x/(np.linalg.norm(x,axis=1,keepdims=True)+1e-12)
def fuse(t,i,wt_t=1.0,wt_i=1.0): return np.concatenate([wt_t*t, wt_i*i],axis=1).astype("float32")

def build_index(records, root, tok, txt_enc, img_enc, bs=64):
    txt_enc.eval(); img_enc.eval()
    texts = [r["prompt"] for r in records]
    paths = [str((root / r["image_path"])) for r in records]

    T,I=[],[]
    for s in tqdm(range(0,len(texts),bs), desc="Encode text"):  T.append(enc_texts(tok, txt_enc, texts[s:s+bs]))
    for s in tqdm(range(0,len(paths),bs), desc="Encode image"): I.append(enc_images(img_enc, paths[s:s+bs]))
    T,I = np.vstack(T), np.vstack(I)
    X = l2n(fuse(T,I,1.0,1.0))

    idx = faiss.IndexFlatIP(X.shape[1]); idx.add(X)
    faiss.write_index(idx, str(INDEX_PATH))
    META_PATH.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    print(f"索引已保存：{INDEX_PATH}；元数据：{META_PATH}")
    return idx, records

def load_index():
    return faiss.read_index(str(INDEX_PATH)), json.loads(META_PATH.read_text(encoding="utf-8"))

# ===== 查询 & LLM 生成 =====
@torch.no_grad()
def encode_query(q_text: Optional[str], q_image: Optional[str], tok, txt_enc, img_enc):
    txt_enc.eval(); img_enc.eval()
    T = enc_texts(tok, txt_enc, [q_text], DEVICE) if q_text and q_text.strip() else np.zeros((1, txt_enc.out_dim), np.float32)
    I = enc_images(img_enc, [q_image], DEVICE)   if q_image and os.path.exists(q_image) else np.zeros((1, img_enc.out_dim), np.float32)
    return l2n(fuse(T,I,1.0,1.0))

def search(idx, metas, Q, topk=5):
    sim, ind = idx.search(Q, topk)
    hits=[]
    for r,(i,s) in enumerate(zip(ind[0].tolist(), sim[0].tolist()),1):
        m=metas[i]
        hits.append({"rank":r,"score":float(s),"task_type":m["task_type"],"prompt":m["prompt"],"image_path":m["image_path"],"meta":m})
    return hits

class LLMClient:
    def __init__(self, base_url="https://api.openai.com/v1", model="gpt-4o-mini", api_key=None, timeout=60):
        self.base_url=base_url.rstrip("/"); self.model=model
        self.api_key=api_key or os.getenv("OPENAI_API_KEY",""); self.timeout=timeout
    def chat(self, messages, temperature=0.2, max_tokens=700):
        r=requests.post(f"{self.base_url}/chat/completions",
                        headers={"Authorization":f"Bearer {self.api_key}","Content-Type":"application/json"},
                        json={"model":self.model,"messages":messages,"temperature":temperature,"max_tokens":max_tokens},
                        timeout=self.timeout)
        r.raise_for_status(); return r.json()["choices"][0]["message"]["content"].strip()

def format_ctx(hits, k=5, max_chars=1600):
    """把命中样本整理成更有用的上下文，包含 objects / actions / container 等。"""
    blocks, used = [], 0
    for h in hits[:k]:
        m = h["meta"]
        objs = ", ".join(m.get("objects", [])[:6]) if isinstance(m.get("objects"), list) else ""
        acts = "; ".join(m.get("actions", [])[:6]) if isinstance(m.get("actions"), list) else ""
        b = textwrap.dedent(f"""\
        [TaskType] {m.get('task_type','')}
        [Prompt] {m.get('prompt','')}
        [Objects] {objs}
        [Container] {m.get('container','')} | status={m.get('container_status','')}
        [ClassType] {m.get('class_type','')}
        [Actions] {acts}
        [Image] {m.get('image_path','')}
        """)
        if used + len(b) > max_chars:
            break
        blocks.append(b); used += len(b)
    return "\n---\n".join(blocks)

# === Gemini 适配器：与现有 ask_with_retrieval 的 llm.chat 接口一致 ===
# 依赖：pip install google-generativeai
 

 

class GeminiClient:
    """
    通过 REST API 调用 Gemini，显式超时+清晰报错，避免 600s 卡死。
    需要 GEMINI_API_KEY，或在构造时传 api_key=...
    """
    def __init__(self, model: str = "gemini-1.5-flash", api_key: Optional[str] = None, timeout: int = 20):
        self.model = model
        self.key   = api_key or os.getenv("GEMINI_API_KEY", "")
        if not self.key:
            raise RuntimeError("未读到 GEMINI_API_KEY，请先设置或在 GeminiClient(api_key=...) 里传入。")
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.key}"
        self.timeout = timeout

    def chat(self, messages: List[Dict], temperature: float = 0.2, max_tokens: int = 700) -> str:
        # 把 OpenAI 风格 messages 扁平成一个 prompt（方便无侵入替换）
        prompt = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": float(temperature),
                "maxOutputTokens": int(max_tokens)
            }
        }
        try:
            r = requests.post(self.url, json=payload, timeout=self.timeout)
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Gemini 网络异常/超时（{self.timeout}s）：{e}")
        if r.status_code != 200:
            raise RuntimeError(f"Gemini HTTP {r.status_code}: {r.text[:500]}")
        data = r.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception:
            raise RuntimeError(f"Gemini 返回解析失败：{data}")

def dedup_hits(hits):
    """按 image_path 去重，只保留每张图分数最高的一条。"""
    best = {}
    for h in hits:
        key = h.get("image_path")
        if key not in best or h["score"] > best[key]["score"]:
            best[key] = h
    # 重新按分数排序
    return sorted(best.values(), key=lambda x: x["score"], reverse=True)

def ask_with_retrieval(q_text, q_image, tok, txt_enc, img_enc, idx, metas, topk=5, llm=None):
    print("→ [1/4] Encoding query...", flush=True)
    Q = encode_query(q_text, q_image, tok, txt_enc, img_enc)

    print("→ [2/4] Searching index...", flush=True)
    hits = dedup_hits(search(idx, metas, Q, topk*2))[:topk]

    ctx = format_ctx(hits, k=topk)
    messages = [
        {
            "role":"system",
            "content":(
                "You are a helpful assistant for visual task understanding. "
                "Use only the provided CONTEXT as evidence. Do not mention any inability to view images. "
                "Prefer concrete nouns from [Objects]/[Actions]; be concise and actionable. "
                "回答使用中文，先给出一句‘结论’，再给出分步操作与证据引用。"
            )
        },
        {
            "role":"user",
            "content":(
                f"用户问题：{q_text or '(仅图片查询)'}\n图片路径：{q_image or 'N/A'}\n\n"
                f"CONTEXT（检索证据）：\n{ctx}\n\n"
                "请输出 JSON：{summary, steps[], evidence[] }，其中 evidence 为 {image_path, score, task_type}。"
            )
        }
    ]
    llm = llm or GeminiClient(model="gemini-1.5-flash")
    print("→ [3/4] Calling LLM...", flush=True)
    try:
        ans = llm.chat(messages)
        print("→ [4/4] LLM done.", flush=True)
        return ans, hits
    except Exception as e:
        print(f"× LLM 调用失败：{e}", flush=True)
        # 回退答案（不至于流程报错）
        if hits:
            h = hits[0]
            fallback = [
                "[LLM 调用失败，返回检索结果摘要供参考]\n",
                f"可能相关任务：{h['task_type']}  (score={h['score']:.3f})",
                f"参考图像：{h['image_path']}",
                f"参考提示：{h['prompt']}"
            ]
            for x in hits[1:]:
                fallback.append(f"- 候选：{x['task_type']} (score={x['score']:.3f}) | {x['image_path']}")
            return "\n".join(fallback), hits
        return "[LLM 调用失败，且未检索到相关样本。]", hits
    return llm.chat(messages), hits


def main():
    # 1) 数据与编码器
    records = filter_has_image(load_records(JSON_DIR), DATASET_ROOT)
    tok = AutoTokenizer.from_pretrained(BERT_MODEL)
    txt_enc = TextEncoder(BERT_MODEL, finetune=False).to(DEVICE)
    img_enc = ImageEncoder(IMG_BACKBONE, finetune=False).to(DEVICE)

    # 加载你训练后的编码器权重（只加载编码器部分）
    if CKPT_PATH.exists():
        ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
        try:
            txt_enc.load_state_dict(ckpt["txt"], strict=False)
            img_enc.load_state_dict(ckpt["img"], strict=False)
            print("✓ 已加载 best_mm_fc.pt 中的编码器权重（txt/img）")
        except Exception as e:
            print(f"⚠️ 加载编码器权重失败，使用预训练权重：{e}")

    # 2) 加载或首次建库
    if INDEX_PATH.exists() and META_PATH.exists():
        idx, metas = load_index(); print("✓ 已加载现有向量库")
    else:
        print("… 第一次运行：正在为训练集构建向量库")
        idx, metas = build_index(records, DATASET_ROOT, tok, txt_enc, img_enc)

    # 3) 示例：文本+图片查询 → LLM 生成
    q_text  = "How to organize the toolbox with three tools?"
    q_image = str(DATASET_ROOT / r"organize_toolbox_three\input\toolbox_3_074.jpg")  # 换成你的图片
     # DeepSeek：OpenAI 兼容 /chat/completions
    # 使用 Gemini（免费层建议：gemini-1.5-flash）
# 优先从环境变量读取；也可以在构造函数里直接传 api_key="你的_Gemini_API_Key"
    gemini_client = GeminiClient(model="gemini-1.5-flash")  # 或 "gemini-1.5-pro"（可能不在免费层）

    answer, hits = ask_with_retrieval(
     q_text, q_image, tok, txt_enc, img_enc, idx, metas, topk=5,
     llm=gemini_client
   )


    print("\n===== LLM Final Answer =====\n", answer)
    print("\n===== Top Hits =====")
    for h in hits:
        print(f"- {h['score']:.3f} [{h['task_type']}] {h['image_path']} | {h['prompt'][:70]}")

if __name__ == "__main__":
    main()


