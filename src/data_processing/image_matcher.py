import os
import re
from pathlib import Path
from typing import List, Dict
from src.config import Config

class ImageMatcher:
    def __init__(self):
        self.image_dir = Path(Config.IMAGE_DIR)
        self.images = self._load_image_files()
        
    def _generate_ngrams(self, words: list, min_n=2, max_n=3):
        ngrams = set()
        words = [w for w in words if w]  # Loại bỏ các từ rỗng
        
        for n in range(min_n, max_n + 1):
            for i in range(len(words) - n + 1):
                ngram = " ".join(words[i:i + n])
                ngrams.add(ngram)
        return ngrams
        
    def _load_image_files(self):
        images = []
        if not self.image_dir.exists():
            print(f"Warning: Image directory {self.image_dir} does not exist")
            return images
            
        for img_path in self.image_dir.glob("*.jpg"):
            try:
                # Lấy tên file không có phần mở rộng
                name = img_path.stem
                # Tách thành list các từ
                words = name.lower().replace("-", " ").split()
                # Tạo các cụm từ từ tên file
                word_groups = self._generate_ngrams(words)
                
                images.append({
                    "path": str(img_path),
                    "name": name,
                    "words": set(words),  # Giữ lại để debug
                    "word_groups": word_groups
                })
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                
        return images
        
    def _extract_location_words(self, query):
        # Chuyển query thành chữ thường và tách thành list các từ
        words = query.lower().split()
        word_groups = self._generate_ngrams(words)
        return word_groups
        
    def find_matching_images(self, query, threshold):
        query_groups = self._extract_location_words(query)
        matches = []
        max_matches = 0
        
        # Đầu tiên tìm số cụm từ khớp tối đa
        for img in self.images:
            matched_groups = query_groups.intersection(img["word_groups"])
            match_count = len(matched_groups)
            if match_count >= threshold:
                matches.append({
                    **img.copy(),
                    "matched_words": matched_groups,
                    "match_count": match_count
                })
                max_matches = max(max_matches, match_count)
                
        # Chỉ lấy các ảnh có số cụm từ khớp tối đa
        matched_images = [
            img for img in matches 
            if img["match_count"] == max_matches
        ]
        for img in matched_images:
            print(f"- {img['name']} (matched groups: {sorted(img['matched_words'])})")
                
        return matched_images