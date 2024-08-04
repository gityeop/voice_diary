from dataclasses import dataclass
import os
import sys
import termios
import tty
from typing import List
import uuid

@dataclass
class DiaryItem:
    id: int
    title: str
    is_selected: bool
    order: int
    is_custom: bool

    def __str__(self):
        return self.title + str(self.is_selected)
# 기본 항목 리스트 생성
diary_items: List[DiaryItem] = [
    DiaryItem(id=0, title="일과", is_selected=False, order=0, is_custom=False),
    DiaryItem(id=1, title="식사 내용", is_selected=False, order=1, is_custom=False),
    DiaryItem(id=2, title="회사", is_selected=False, order=2, is_custom=False),
    DiaryItem(id=3, title="자기계발", is_selected=False, order=3, is_custom=False),
    DiaryItem(id=4, title="만난 사람", is_selected=False, order=4, is_custom=False),
    DiaryItem(id=5, title="운동", is_selected=False, order=5, is_custom=False),
    DiaryItem(id=6, title="명상", is_selected=False, order=6, is_custom=False),
    DiaryItem(id=7, title="독서", is_selected=False, order=7, is_custom=False),
    DiaryItem(id=8, title="배운 점", is_selected=False, order=8, is_custom=False),
    DiaryItem(id=9, title="감사한 일", is_selected=False, order=9, is_custom=False),
    DiaryItem(id=10, title="목표/계획", is_selected=False, order=10, is_custom=False),
    DiaryItem(id=11, title="내일 해야할 일", is_selected=False, order=11, is_custom=False)
]

# 사용자 정의 항목 추가 함수
def add_custom_item(title: str):
    new_id = max(item.id for item in diary_items) + 1
    new_item = DiaryItem(
        id=new_id,
        title=title,
        is_selected=True,
        order=len(diary_items),
        is_custom=True
    )
    diary_items.append(new_item)

# 항목 선택/해제 함수
def toggle_item_selection(id: int):
    for item in diary_items:
        if item.id == id:
            item.is_selected = not item.is_selected
            break

# 선택된 항목 리스트 반환 함수
def get_selected_items() -> List[DiaryItem]:
    return [item for item in diary_items if item.is_selected]
# toggle_item_selection(diary_items[0].id)

# for i in diary_items:
#     print(i)


def get_key():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
        if ch == '\x1b':
            ch += sys.stdin.read(2)
        elif ch == '\r':
            ch = '\n'
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def print_menu(current_row):
    os.system('clear')
    for idx, item in enumerate(diary_items):
        if idx == current_row:
            line = f"> {item.title} {'*' if item.is_selected else ''}"
        else:
            line = f"  {item.title} {'*' if item.is_selected else ''}"
        print(line)
    print("\n위/아래 화살표로 항목을 이동하고 엔터키로 선택/해제합니다. 종료하려면 q를 누르세요.")
    