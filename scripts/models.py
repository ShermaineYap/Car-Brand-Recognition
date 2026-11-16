"""Turn a crawl keyword such as '2019 Toyota Corolla Altis 1.8G' into (make, model, year)."""
import re

BRAND_ALIASES = {"mercededs-benz": "mercedes-benz", "mercedes-amg": "mercedes-benz", "mini": "mini",
                 "citroën": "citroen", "rolls royce": "rolls-royce"}
TWO_WORD_BRANDS = ("land rover", "great wall", "rolls royce", "alfa romeo", "aston martin")

# Longest-prefix rules per make: keyword text after the make -> model name. Anything not
# listed falls back to the first word.
RULES = {
    "toyota": {"corolla altis": "corolla altis", "corolla cross": "corolla cross", "gr corolla": "gr corolla",
               "gr supra": "gr supra", "gr yaris": "gr yaris", "gr 86": "gr86", "86": "86",
               "land cruiser": "land cruiser", "fj cruiser": "fj cruiser", "innova zenix": "innova zenix",
               "yaris cross": "yaris cross"},
    "honda": {"city hatchback": "city hatchback"},
    "chery": {"arrizo 3": "arrizo 3", "arrizo 5": "arrizo 5", "arrizo 6": "arrizo 6", "arrizo 7": "arrizo 7",
              "fulwin 2": "fulwin 2", "new tiggo 3": "tiggo 3", "new qq": "qq", "omoda 5": "omoda 5",
              "tiggo 2": "tiggo 2", "tiggo 7": "tiggo 7"},
    "mazda": {"2": "mazda2", "3": "mazda3", "6": "mazda6"},
    "byd": {"atto 3": "atto 3"},
    "mercedes-benz": {"a200": "a-class", "a250": "a-class", "a45": "a-class", "s 560": "s-class",
                      "maybach s-class": "s-class", "glc coupe": "glc", "gle coupe": "gle",
                      "eqe suv": "eqe suv", "eqs suv": "eqs suv", "gt": "amg gt", "slc amg": "slc"},
    "lexus": {"lm350": "lm", "rc f": "rc"},
    "bmw": {"1 series": "1 series", "2 series": "2 series", "3 series": "3 series", "4 series": "4 series",
            "5 series": "5 series", "6 series": "6 series", "7 series": "7 series", "8 series": "8 series",
            "m850i": "8 series", "m3 m340i": "3 series", "i3s": "i3"},
    "haval": {"first love": "first love"},
    "ora": {"black cat": "black cat", "good cat": "good cat"},
    "nissan": {"grand livina": "grand livina", "z proto": "z", "nv350 urvan": "nv350"},
    "kia": {"grand carnival": "carnival"},
    "mini": {"3 door": "3 door", "5 door": "5 door", "jcw clubman": "clubman", "john cooper works": "3 door"},
    "proton": {"satria neo": "satria neo", "suprima s": "suprima s"},
}


def parse(keyword):
    m = re.match(r"^(\d{4}) (.+)$", keyword.strip())
    if not m:
        return None
    year, rest = m.group(1), m.group(2)
    low = rest.lower()
    make = next((b for b in TWO_WORD_BRANDS if low.startswith(b + " ")), low.split()[0])
    tail = low[len(make):].strip()
    make = BRAND_ALIASES.get(make, make)
    if make == "mercedes-benz" and tail.startswith("amg "):
        tail = tail[4:]
    if not tail:
        return None
    rules = RULES.get(make, {})
    for prefix in sorted(rules, key=len, reverse=True):
        if tail == prefix or tail.startswith(prefix + " "):
            return make, rules[prefix], year
    return make, tail.split()[0], year
