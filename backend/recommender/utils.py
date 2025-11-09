import re

TECH = ["java","python","sql","javascript","excel","selenium","react","node","tableau"]
SOFT = ["communication","leadership","teamwork","collaboration","analytical","problem solving"]

def extract_requirements(query: str):
    q = query.lower()
    found_tech = [w for w in TECH if w in q]
    found_soft = [w for w in SOFT if w in q]
    needs_div = bool(found_tech and found_soft)
    return {"tech": found_tech, "soft": found_soft, "needs_diversity": needs_div}