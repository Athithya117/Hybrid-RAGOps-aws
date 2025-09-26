import os
import logging
from typing import Iterable, List, Tuple, Dict, Generator, Optional
import spacy
try:
    from spacy.pipeline import Sentencizer
except Exception:
    Sentencizer = None
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    try:
        return int(v) if v is not None else default
    except Exception:
        logger.warning("Invalid env var for %s: %r; falling back to %d", name, v, default)
        return default
class TokenEncoder:
    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        self.backend = "whitespace"
        try:
            import tiktoken
            enc = tiktoken.encoding_for_model(model_name)
            self.encode = lambda txt: enc.encode(txt)
            self.decode = lambda toks: enc.decode(toks)
            self.backend = "tiktoken"
            logger.info("Using tiktoken for token counting (model encoding: %s)", model_name)
        except Exception:
            logger.warning("tiktoken not available or failed; falling back to whitespace token counter.")
            self.encode = lambda txt: txt.split()
            self.decode = lambda toks: " ".join(toks)
class SentenceChunker:
    def __init__(self, max_tokens_per_chunk: Optional[int] = None, overlap_sentences: Optional[int] = None, token_model: str = "gpt2", nlp=None, min_tokens_per_chunk: Optional[int] = None):
        self.max_tokens_per_chunk = _env_int("MAX_TOKENS_PER_CHUNK", 600) if max_tokens_per_chunk is None else int(max_tokens_per_chunk)
        self.overlap_sentences = _env_int("NUMBER_OF_OVERLAPPING_SENTENCES", 3) if overlap_sentences is None else int(overlap_sentences)
        if self.overlap_sentences < 0:
            raise ValueError("overlap_sentences must be >= 0")
        self.min_tokens_per_chunk = _env_int("MIN_TOKENS_PER_CHUNK", 100) if min_tokens_per_chunk is None else int(min_tokens_per_chunk)
        if self.min_tokens_per_chunk < 0:
            raise ValueError("min_tokens_per_chunk must be >= 0")
        self.encoder = TokenEncoder(model_name=token_model)
        self.nlp = nlp or self._make_blank_sentencizer()
        logger.info("SentenceChunker initialized: max_tokens=%d, min_tokens=%d, overlap_sentences=%d, token_backend=%s", self.max_tokens_per_chunk, self.min_tokens_per_chunk, self.overlap_sentences, getattr(self.encoder, "backend", "unknown"))
    @staticmethod
    def _make_blank_sentencizer():
        nlp = spacy.blank("en")
        try:
            if Sentencizer is not None:
                nlp.add_pipe("sentencizer")
            else:
                nlp.add_pipe("sentencizer")
        except Exception as e:
            if Sentencizer is not None:
                nlp.add_pipe(Sentencizer())
            else:
                raise RuntimeError("Failed to add Sentencizer to the spaCy pipeline. Ensure your spaCy installation is functional. Error: " + str(e))
        return nlp
    def _sentences_with_offsets(self, text: str) -> List[Tuple[str, int, int]]:
        doc = self.nlp(text)
        sents = [(sent.text.strip(), int(sent.start_char), int(sent.end_char)) for sent in doc.sents if sent.text.strip()]
        return sents
    def chunk_document(self, text: str) -> Generator[Dict, None, None]:
        sentences = self._sentences_with_offsets(text)
        sent_items: List[Dict] = [{"text": s, "start_char": sc, "end_char": ec, "orig_idx": i, "is_remainder": False} for i, (s, sc, ec) in enumerate(sentences)]
        i = 0
        n = len(sent_items)
        prev_chunk = None
        while i < n:
            cur_token_count = 0
            chunk_sent_texts: List[str] = []
            chunk_start_idx = i
            chunk_start_char: Optional[int] = sent_items[i]["start_char"] if i < n else None
            chunk_end_char: Optional[int] = None
            is_truncated_sentence = False
            while i < n:
                sent_text = sent_items[i]["text"]
                tok_ids = self.encoder.encode(sent_text)
                sent_tok_len = len(tok_ids)
                if cur_token_count + sent_tok_len > self.max_tokens_per_chunk:
                    if not chunk_sent_texts:
                        prefix_tok_ids = tok_ids[: self.max_tokens_per_chunk]
                        prefix_text = self.encoder.decode(prefix_tok_ids)
                        chunk_sent_texts.append(prefix_text)
                        cur_token_count = len(prefix_tok_ids)
                        is_truncated_sentence = True
                        remainder_tok_ids = tok_ids[self.max_tokens_per_chunk :]
                        if remainder_tok_ids:
                            remainder_text = self.encoder.decode(remainder_tok_ids)
                            sent_items[i] = {"text": remainder_text, "start_char": None, "end_char": None, "orig_idx": sent_items[i]["orig_idx"], "is_remainder": True}
                        else:
                            i += 1
                        break
                    else:
                        break
                else:
                    chunk_sent_texts.append(sent_text)
                    cur_token_count += sent_tok_len
                    chunk_end_char = sent_items[i]["end_char"]
                    i += 1
            if not chunk_sent_texts:
                logger.warning("Empty chunk at idx %d; advancing one sentence to avoid infinite loop", i)
                i += 1
                continue
            chunk_text = " ".join(chunk_sent_texts).strip()
            chunk_meta = {"text": chunk_text, "token_count": cur_token_count, "start_sentence_idx": chunk_start_idx, "end_sentence_idx": i, "start_char": chunk_start_char, "end_char": chunk_end_char, "is_truncated_sentence": is_truncated_sentence}
            new_start = max(chunk_start_idx + 1, chunk_meta["end_sentence_idx"] - self.overlap_sentences)
            if prev_chunk is None:
                prev_chunk = chunk_meta
            else:
                if chunk_meta["token_count"] < self.min_tokens_per_chunk:
                    prev_chunk["text"] = prev_chunk["text"] + " " + chunk_meta["text"]
                    prev_chunk["token_count"] = prev_chunk["token_count"] + chunk_meta["token_count"]
                    prev_chunk["end_sentence_idx"] = chunk_meta["end_sentence_idx"]
                    prev_chunk["end_char"] = chunk_meta["end_char"]
                    prev_chunk["is_truncated_sentence"] = prev_chunk["is_truncated_sentence"] or chunk_meta["is_truncated_sentence"]
                else:
                    yield prev_chunk
                    prev_chunk = chunk_meta
            i = new_start
            n = len(sent_items)
        if prev_chunk is not None:
            yield prev_chunk
    def chunk_documents(self, texts: Iterable[str], batch_size: int = 256, n_process: int = 1) -> Generator[Dict, None, None]:
        for doc_idx, doc in enumerate(self.nlp.pipe(texts, batch_size=batch_size, n_process=n_process)):
            sent_list = [(s.text.strip(), int(s.start_char), int(s.end_char)) for s in doc.sents if s.text.strip()]
            sent_items: List[Dict] = [{"text": s, "start_char": sc, "end_char": ec, "orig_idx": i, "is_remainder": False} for i, (s, sc, ec) in enumerate(sent_list)]
            i = 0
            n = len(sent_items)
            prev_chunk = None
            while i < n:
                cur_token_count = 0
                chunk_sent_texts: List[str] = []
                chunk_start_idx = i
                chunk_start_char: Optional[int] = sent_items[i]["start_char"] if i < n else None
                chunk_end_char: Optional[int] = None
                is_truncated_sentence = False
                while i < n:
                    sent_text = sent_items[i]["text"]
                    tok_ids = self.encoder.encode(sent_text)
                    sent_tok_len = len(tok_ids)
                    if cur_token_count + sent_tok_len > self.max_tokens_per_chunk:
                        if not chunk_sent_texts:
                            prefix_tok_ids = tok_ids[: self.max_tokens_per_chunk]
                            prefix_text = self.encoder.decode(prefix_tok_ids)
                            chunk_sent_texts.append(prefix_text)
                            cur_token_count = len(prefix_tok_ids)
                            is_truncated_sentence = True
                            remainder_tok_ids = tok_ids[self.max_tokens_per_chunk :]
                            if remainder_tok_ids:
                                remainder_text = self.encoder.decode(remainder_tok_ids)
                                sent_items[i] = {"text": remainder_text, "start_char": None, "end_char": None, "orig_idx": sent_items[i]["orig_idx"], "is_remainder": True}
                            else:
                                i += 1
                            break
                        else:
                            break
                    else:
                        chunk_sent_texts.append(sent_text)
                        cur_token_count += sent_tok_len
                        chunk_end_char = sent_items[i]["end_char"]
                        i += 1
                if not chunk_sent_texts:
                    i += 1
                    continue
                chunk_text = " ".join(chunk_sent_texts).strip()
                chunk_meta = {"doc_idx": doc_idx, "text": chunk_text, "token_count": cur_token_count, "start_sentence_idx": chunk_start_idx, "end_sentence_idx": i, "start_char": chunk_start_char, "end_char": chunk_end_char, "is_truncated_sentence": is_truncated_sentence}
                new_start = max(chunk_start_idx + 1, chunk_meta["end_sentence_idx"] - self.overlap_sentences)
                if prev_chunk is None:
                    prev_chunk = chunk_meta
                else:
                    if chunk_meta["token_count"] < self.min_tokens_per_chunk:
                        prev_chunk["text"] = prev_chunk["text"] + " " + chunk_meta["text"]
                        prev_chunk["token_count"] = prev_chunk["token_count"] + chunk_meta["token_count"]
                        prev_chunk["end_sentence_idx"] = chunk_meta["end_sentence_idx"]
                        prev_chunk["end_char"] = chunk_meta["end_char"]
                        prev_chunk["is_truncated_sentence"] = prev_chunk["is_truncated_sentence"] or chunk_meta["is_truncated_sentence"]
                    else:
                        yield prev_chunk
                        prev_chunk = chunk_meta
                i = new_start
                n = len(sent_items)
            if prev_chunk is not None:
                yield prev_chunk
    @classmethod
    def from_env(cls, **kwargs):
        return cls(max_tokens_per_chunk=kwargs.get("max_tokens_per_chunk", None), overlap_sentences=kwargs.get("overlap_sentences", None), token_model=kwargs.get("token_model", os.getenv("TOKEN_ENCODER_MODEL", "gpt2")), nlp=kwargs.get("nlp", None), min_tokens_per_chunk=kwargs.get("min_tokens_per_chunk", None))
if __name__ == "__main__":
    chunker = SentenceChunker.from_env()
    docs = ["Dennis Great Roger Federer announced his retirement from the gang Thursday after a career which saw him win 20 grand slams, 100 re-career titles, and be crowned one of the greatest players of all time. The 41-year-old said he would retire after the labor cup in London later in September. He hasn't laid since Wimbledon in 2021 because of a knee problem. The Mexican government says an army general has been detained in connection with the disappearance of 43 students in 2014. It's the latest arrest in a case that generated international condemnation. The Deputy Secretary of Security Minister Toll reporters Thursday that the general commended a battalion in the area of southern Mexico where the incident occurred, he's one of three suspects detained. The Deputy Security Minister did not identify the suspects, but said the other two were also military personnel. Prosecutors announced last month that arrest warrants had been issued for more than 80 suspects in the case, including 20 military personnel, 44 police officers, and 14 cartel members. The case is one of the worst human rights tragedies in Mexico where a spiral of drug-related violence has left more than 100,000 people missing. The United States Congress is moving ahead on bipartisan action, strengthening trade and defense ties with Taiwan as the self-governing island faces new threats from China. The U.S. Senate Foreign Relations Committee approved the Taiwan Policy Act on Wednesday, clearing the way for $6.5 billion in enhanced security funding over five years to come up for a full vote on the Senate floor. The measure would also designate Taiwan as a major non-NATO ally. Senate Foreign Relations Committee Chairman Bob Nendez said after the committee approved the bill that the measure aims to raise the cost of taking the island by force, which China has threatened to do. President Biden approved $1.1 billion in arms sales to Taiwan earlier this month. China condemned that move. And once again, the U.S. on Thursday imposed new economic sanctions on a number of Russians, including some, it accuses of stealing Ukrainian grain.  " * 5]
    for chunk in chunker.chunk_documents(docs, batch_size=16, n_process=1):
        print(f'DOC {chunk["doc_idx"]}  SENTS {chunk["start_sentence_idx"]}->{chunk["end_sentence_idx"]} TOKS {chunk["token_count"]}')
        print(chunk["text"])
