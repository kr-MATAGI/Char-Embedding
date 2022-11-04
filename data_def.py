from dataclasses import dataclass, field
from typing import List

# 데이터 구조 - ERTI 언어분석 말뭉치
@dataclass
class Morp:
    id: int = -1
    form: str = ""
    label: str = "" # 형태소
    word_id: int = -1
    position: int = -1 # 문장에서의 byte position

@dataclass
class NE:
    id: int = -1
    text: str = ""
    type: str = ""
    begin: int = -1 # 개체명을 구성하는 첫 형태소의 ID
    end: int = -1 #	개체명을 구성하는 끝 형태소의 ID
    weight: float = -1.0 # 개체명 인식 결과 신뢰도 (0 ~ 1 사이의 값, 높을수록 높은 신뢰도)

@dataclass
class Word:
    id: int = -1
    form: str = ""
    begin: int = -1
    end: int = -1

@dataclass
class Sentence:
    id: str = ""
    text: str = ""
    word_list: List[Word] = field(default_factory=list)
    morp_list: List[Morp] = field(default_factory=list)
    ne_list: List[NE] = field(default_factory=list)