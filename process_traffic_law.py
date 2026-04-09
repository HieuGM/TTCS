#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import json
import glob
from typing import List, Dict, Optional
from pathlib import Path

class TrafficLawProcessor:
    """Process Vietnamese Traffic Law text according to STRICT data engineering rules"""

    SUBJECTS_ENUM = set([
        "tat_ca", "nguoi_di_bo", "xe_dap", "xe_dap_dien", "xe_may", "xe_mo_to", 
        "o_to", "xe_co_gioi", "xe_chuyen_dung", "nguoi_lai_xe", "chu_xe"
    ])

    TOPICS_ENUM = set([
        "quy_dinh_chung", "lan_duong", "phan_duong", "toc_do", "khoang_cach_an_toan", 
        "nhuong_duong", "vuot_xe", "chuyen_huong", "dung_do", "bien_bao", 
        "den_tin_hieu", "vach_ke_duong", "giay_phep_lai_xe", "diem_gplx", 
        "nong_do_con", "xu_phat"
    ])

    def __init__(self, source_file: str):
        self.source_file = Path(source_file)
        self.lines: List[str] = []
        self.clean_content: List[str] = []
        self.records: List[Dict] = []
        
        # Meta properties for the document determined heuristically
        self.doc_name = ""
        self.legal_layer = "rule"
        self.nguon = ""
        self.effective_from = "2025-01-01"
        self._detect_document_layer()

    def _detect_document_layer(self):
        filename = self.source_file.name.lower()
        if "36_2024" in filename or "luật" in filename.lower():
            self.legal_layer = "rule"
            self.nguon = "luat_ttatgt_2024"
            self.effective_from = "2025-01-01"
        elif "151" in filename:
            self.legal_layer = "implementation"
            self.nguon = "nd151_2024"
            self.effective_from = "2025-01-01"
        elif "qcvn 41" in filename or "qcvn41" in filename:
            self.legal_layer = "sign"
            self.nguon = "qcvn41_2024"
            self.effective_from = "2025-01-01"  # Or update date
        elif "38" in filename and "tt" in filename:
            self.legal_layer = "speed_distance"
            self.nguon = "tt38"
            self.effective_from = "2025-01-01"
        elif "168" in filename:
            self.legal_layer = "sanction"
            self.nguon = "nd168_2024"
            self.effective_from = "2025-01-01"
        else:
            self.legal_layer = "rule"
            self.nguon = "doc_" + filename.split('_')[0]
            self.effective_from = "2025-01-01"

    def generate_record_id(self, article: str, clause: Optional[str] = None, point: Optional[str] = None, is_intro: bool = False) -> str:
        """Helper to generate standard record_id: [nguon]_[article]_[clause]_[point]"""
        def normalize_part(text):
            if not text: return ""
            text = text.lower()
            text = self.remove_accents(text)
            text = re.sub(r'[^a-z0-9\s]', '', text)
            # Remove "dieu", "khoan", "diem" prefixes for shorter IDs
            text = re.sub(r'^dieu\s*', 'd', text)
            text = re.sub(r'^khoan\s*', 'k', text)
            text = re.sub(r'^diem\s*', 'p', text)
            text = text.replace(' ', '_')
            return text

        parts = [self.nguon, normalize_part(article)]
        if clause:
            parts.append(normalize_part(clause))
        if point:
            parts.append(normalize_part(point))
        
        base_id = "_".join(filter(bool, parts))
        if is_intro:
            base_id += "_intro"
            
        return base_id

    def remove_accents(self, text: str) -> str:
        accent_map = {
            'à': 'a', 'á': 'a', 'ả': 'a', 'ã': 'a', 'ạ': 'a', 'ă': 'a', 'ằ': 'a', 'ắ': 'a', 'ẳ': 'a', 'ẵ': 'a', 'ặ': 'a', 'â': 'a', 'ầ': 'a', 'ấ': 'a', 'ẩ': 'a', 'ẫ': 'a', 'ậ': 'a', 'đ': 'd',
            'è': 'e', 'é': 'e', 'ẻ': 'e', 'ẽ': 'e', 'ẹ': 'e', 'ê': 'e', 'ề': 'e', 'ế': 'e', 'ể': 'e', 'ễ': 'e', 'ệ': 'e',
            'ì': 'i', 'í': 'i', 'ỉ': 'i', 'ĩ': 'i', 'ị': 'i',
            'ò': 'o', 'ó': 'o', 'ỏ': 'o', 'õ': 'o', 'ọ': 'o', 'ô': 'o', 'ồ': 'o', 'ố': 'o', 'ổ': 'o', 'ỗ': 'o', 'ộ': 'o', 'ơ': 'o', 'ờ': 'o', 'ớ': 'o', 'ở': 'o', 'ỡ': 'o', 'ợ': 'o',
            'ù': 'u', 'ú': 'u', 'ủ': 'u', 'ũ': 'u', 'ụ': 'u', 'ư': 'u', 'ừ': 'u', 'ứ': 'u', 'ử': 'u', 'ữ': 'u', 'ự': 'u',
            'ỳ': 'y', 'ý': 'y', 'ỷ': 'y', 'ỹ': 'y', 'ỵ': 'y'
        }
        for accented, unaccented in accent_map.items():
            text = text.replace(accented, unaccented)
            text = text.replace(accented.upper(), unaccented.upper())
        return text

    def read_file(self):
        with open(self.source_file, 'r', encoding='utf-8') as f:
            self.lines = f.readlines()

    def clean_line(self, line: str) -> Optional[str]:
        if not line or line.isspace(): return None
        line = line.strip()
        line = re.sub(r'^[|_\-\s]+', '', line)
        line = re.sub(r'[|_\-\s]+$', '', line)
        if not line: return None
        if re.match(r'^\d+$', line): return None
        if 'E-pas:' in line: return None
        return line

    def detect_structure(self, line: str) -> Dict:
        structure = {'type': 'text', 'level': 0, 'content': line}
        if re.match(r'^Luật\s+số:', line):
            structure['type'] = 'doc_header'
            return structure
        if re.match(r'^LUẬT\s+', line) or re.match(r'^LỆNH\s+', line) or re.match(r'^NGHỊ ĐỊNH\s+', line):
            structure['type'] = 'doc_name'
            # Update doc_name dynamically if found
            if not getattr(self, 'doc_name_found', False):
                self.doc_name = line
                self.doc_name_found = True
            return structure
        chap_match = re.match(r'^Chương\s+(?:[IVXLCDM]+|\d+)\.?\s*(.+)', line, re.IGNORECASE)
        if chap_match:
            structure['type'] = 'chapter'
            structure['content'] = f"Chương {chap_match.group(1).split()[0]}"  # Giữ tên chương
            return structure
        sect_match = re.match(r'^Mục\s+\d+\.?\s*(.+)', line)
        if sect_match:
            structure['type'] = 'section'
            structure['content'] = f"Mục {sect_match.group(1)}"
            return structure
        art_match = re.match(r'^Điều\s+\d+\.(?:\s+.*$)?', line)
        if art_match:
            structure['type'] = 'article'
            structure['article_num'] = re.search(r'Điều\s+\d+', line).group()
            return structure
        if re.match(r'^\d+\.\s', line):
            structure['type'] = 'clause'
            structure['clause_num'] = f"Khoản {re.search(r'^\d+', line).group()}"
            return structure
        if re.match(r'^[a-zđ]\)\s', line.lower()):
            structure['type'] = 'point'
            structure['point_num'] = f"Điểm {re.search(r'^[a-zđ]', line.lower()).group()}"
            return structure
        if re.match(r'^Phụ\s+lục', line, re.IGNORECASE):
            structure['type'] = 'appendix'
            return structure
        return structure

    def process_content(self):
        for line in self.lines:
            cleaned = self.clean_line(line)
            if cleaned:
                self.clean_content.append(cleaned)
        # Default doc name if none found
        if not self.doc_name:
            self.doc_name = self.source_file.stem

    def classify_subject_and_topic(self, text: str) -> tuple:
        text_lower = text.lower()
        subjects = set()
        topics = set()

        if any(k in text_lower for k in ['người đi bộ']): subjects.add('nguoi_di_bo')
        if any(k in text_lower for k in ['xe đạp', 'xe thô sơ']): subjects.add('xe_dap')
        if 'xe đạp điện' in text_lower: subjects.add('xe_dap_dien')
        if any(k in text_lower for k in ['xe máy', 'mô tô hai bánh']): subjects.add('xe_may')
        if 'mô tô' in text_lower or 'xe mô tô' in text_lower: subjects.add('xe_mo_to')
        if any(k in text_lower for k in ['ô tô', 'xe ô tô', 'xe hơi', 'o to']): subjects.add('o_to')
        if 'xe cơ giới' in text_lower: subjects.add('xe_co_gioi')
        if 'chuyên dùng' in text_lower: subjects.add('xe_chuyen_dung')
        if any(k in text_lower for k in ['người lái xe', 'lái xe', 'điều khiển']): subjects.add('nguoi_lai_xe')
        if any(k in text_lower for k in ['chủ xe', 'chủ phương tiện']): subjects.add('chu_xe')
        if not subjects: subjects.add('tat_ca')
        
        # Intersection with enum to be strict
        filtered_subjects = list(subjects.intersection(self.SUBJECTS_ENUM))
        if not filtered_subjects: filtered_subjects = ['tat_ca']

        if any(k in text_lower for k in ['quy định chung', 'áp dụng']): topics.add('quy_dinh_chung')
        if any(k in text_lower for k in ['làn đường', 'chuyển làn']): topics.add('lan_duong')
        if 'phần đường' in text_lower: topics.add('phan_duong')
        if any(k in text_lower for k in ['tốc độ', 'giới hạn']): topics.add('toc_do')
        if any(k in text_lower for k in ['khoảng cách']): topics.add('khoang_cach_an_toan')
        if any(k in text_lower for k in ['nhường đường', 'nhường']): topics.add('nhuong_duong')
        if any(k in text_lower for k in ['vượt', 'vượt xe']): topics.add('vuot_xe')
        if any(k in text_lower for k in ['chuyển hướng', 'rẽ', 'quay đầu']): topics.add('chuyen_huong')
        if any(k in text_lower for k in ['dừng', 'đỗ', 'dừng xe']): topics.add('dung_do')
        if any(k in text_lower for k in ['biển báo', 'biển hiệu', 'cấm']): topics.add('bien_bao')
        if any(k in text_lower for k in ['đèn tín hiệu', 'đèn giao thông']): topics.add('den_tin_hieu')
        if any(k in text_lower for k in ['vạch kẻ đường', 'vạch']): topics.add('vach_ke_duong')
        if any(k in text_lower for k in ['giấy phép lái xe', 'bằng lái', 'gplx']): topics.add('giay_phep_lai_xe')
        if any(k in text_lower for k in ['điểm']): topics.add('diem_gplx')
        if any(k in text_lower for k in ['nồng độ cồn', 'rượu']): topics.add('nong_do_con')
        if any(k in text_lower for k in ['xử phạt', 'phạt', 'biện pháp']): topics.add('xu_phat')
        if not topics: topics.add('quy_dinh_chung')

        filtered_topics = list(topics.intersection(self.TOPICS_ENUM))
        if not filtered_topics: filtered_topics = ['quy_dinh_chung']

        return filtered_subjects, filtered_topics

    def generate_jsonl(self) -> str:
        jsonl_lines = []
        current_chapter = ""
        current_section = None
        current_article = ""
        current_article_title = ""
        current_clauses = []

        for line in self.clean_content:
            structure = self.detect_structure(line)
            if structure['type'] == 'chapter':
                current_chapter = structure['content']
            elif structure['type'] == 'section':
                current_section = structure['content']
            elif structure['type'] == 'article':
                if current_article and current_clauses:
                    self._save_article_record(jsonl_lines, current_chapter, current_section, current_article, current_article_title, current_clauses)
                current_article = structure['article_num']
                current_article_title = line
                current_clauses = []
            elif structure['type'] == 'clause':
                current_clauses.append({
                    'num': structure['clause_num'],
                    'text': line[len(structure['clause_num'].split()[1])+2:].strip() if len(structure['clause_num'].split()) > 1 else line.strip(),
                    'points': []
                })
            elif structure['type'] == 'point':
                p_text = line[2:].strip() if line.lower()[1] == ')' else line.strip()
                if current_clauses:
                    current_clauses[-1]['points'].append({'num': structure['point_num'], 'text': p_text})
                else: # Sometimes a point exists directly under an article without a numbered clause
                    current_clauses.append({'num': None, 'text': '', 'points': [{'num': structure['point_num'], 'text': p_text}]})
            elif current_clauses:
                if current_clauses[-1]['points']:
                    current_clauses[-1]['points'][-1]['text'] += ' ' + line
                else:
                    current_clauses[-1]['text'] += ' ' + line
            elif current_article:
                current_clauses.append({'num': None, 'text': line, 'points': []})

        if current_article and current_clauses:
            self._save_article_record(jsonl_lines, current_chapter, current_section, current_article, current_article_title, current_clauses)

        return '\n'.join(jsonl_lines)

    def _save_article_record(self, jsonl_lines, chapter, section, article, title, clauses):
        
        def commit_record(rec_id, art_num, clause_num, point_num, txt):
            subj, top = self.classify_subject_and_topic(txt)
            record = {
                "record_id": rec_id,
                "source_doc": self.doc_name,
                "source_type": "law",
                "legal_layer": self.legal_layer,
                "chapter": chapter if chapter else "",
                "section": section,
                "article": art_num,
                "clause": clause_num,
                "point": point_num,
                "title": title,
                "text": txt.strip(),
                "subjects": subj,
                "topics": top,
                "effective_from": self.effective_from,
                "effective_to": None,
                "status": "current"
            }
            jsonl_lines.append(json.dumps(record, ensure_ascii=False))

        for clause in clauses:
            clause_num = clause['num']
            clause_text = clause['text']

            if not clause['points']:
                # Save just the clause
                rec_id = self.generate_record_id(article, clause_num)
                commit_record(rec_id, article, clause_num, None, clause_text)
            else:
                # Save Intro chunk
                if clause_text.strip():
                    rec_id = self.generate_record_id(article, clause_num, point=None, is_intro=True)
                    commit_record(rec_id, article, clause_num, None, clause_text)

                # Save each Point
                for point in clause['points']:
                    pt_text = point['text']
                    # Context prepend
                    full_text = f"{clause_text.strip()} \n {point['num'].split()[1] if len(point['num'].split()) > 1 else point['num']}) {pt_text}" if clause_text.strip() else f"{point['num']}) {pt_text}"
                    
                    rec_id = self.generate_record_id(article, clause_num, point['num'])
                    commit_record(rec_id, article, clause_num, point['num'], full_text)


def process_all_files():
    input_files = glob.glob("raw_data/*.txt")
    if not input_files:
        print("No .txt files found in raw_data/.")
        return

    for file_path in input_files:
        print(f"Processing {file_path}...")
        processor = TrafficLawProcessor(file_path)
        processor.read_file()
        processor.process_content()
        
        output_file = Path(file_path).stem + "_structured.jsonl"
        output_path = Path("raw_data") / output_file
        
        jsonl_content = processor.generate_jsonl()
        output_path.write_text(jsonl_content, encoding='utf-8')
        print(f"[OK] Generated {output_path}")

if __name__ == "__main__":
    process_all_files()
