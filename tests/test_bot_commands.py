"""Standalone tests for course filtering / difficulty helpers in bot.py.

Run with:
    python test_courses.py
    python test_courses.py -v   # verbose output
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# Allow importing bot without starting the Discord client.
# bot.py is guarded by `if __name__ == "__main__": bot.run(TOKEN)` so the
# import is safe as long as DISCORD_TOKEN doesn't need to be present.
os.environ.setdefault("DISCORD_TOKEN", "test-token-placeholder")

import bot


def check(label, condition, detail=""):
    mark = "✓" if condition else "✗"
    print(f"  {mark}  {label}" + (f"  — {detail}" if detail else ""))
    if not condition:
        print()
        print("    TEST FAILED!")
        sys.exit(1)
    
    return condition


def section(title):
    print(f"\n{'─'*55}\n  {title}\n{'─'*55}")


courses = bot._load_courses()
diff = bot._load_difficulty()

#-----------------------------------------
section("Easy / hard filter")
easy = bot._filter_courses(courses, ["easy"], diff)
hard = bot._filter_courses(courses, ["hard"], diff)
check("easy: all codes end in E", all(c.endswith("E") for c, _ in easy), f"{len(easy)} courses")
check("hard: all codes end in H", all(c.endswith("H") for c, _ in hard), f"{len(hard)} courses")
check("easy + hard = 80", len(easy) + len(hard) == 80)

#-----------------------------------------
section("Exclusion filter")
hard_no_bbh = bot._filter_courses(courses, ["hard", "-BBH"], diff)
check("-BBH removes BBH", not any(c == "BBH" for c, _ in hard_no_bbh))
check("-BBH keeps other hard courses", len(hard_no_bbh) == len(hard) - 1, f"{len(hard_no_bbh)} remaining")

no_venice = bot._filter_courses(courses, ["-venice"], diff)
check("-venice removes VNE and VNH",
      not any("venice" in n.lower() for _, n in no_venice),
      f"removed {len(courses) - len(no_venice)} course(s)")

no_8bit = bot._filter_courses(courses, ['-"8-bit lair"'], diff)
check('-"8-bit lair" removes A8E and A8H',
      not any("8-bit lair" in n.lower() for _, n in no_8bit),
      f"removed {len(courses) - len(no_8bit)} course(s)")


#-----------------------------------------
section("Fuzzy name filter")
journey = bot._filter_courses(courses, ["journey"], diff)
check("'journey' matches JCE and JCH", len(journey) == 2,
      str([(c, n) for c, n in journey]))

atlantis = bot._filter_courses(courses, ["atlantis"], diff)
check("'atlantis' matches ATE and ATH", len(atlantis) == 2,
      str([(c, n) for c, n in atlantis]))


#-----------------------------------------
section("top: / bottom: filter")
top5 = bot._filter_courses(courses, ["top:5"], diff)
check("top:5 returns 5 courses", len(top5) == 5)
for code, name in top5:
    print(f"\t{code}={diff[code]}: {name}")
print()

bottom3 = bot._filter_courses(courses, ["bottom:3"], diff)
check("bottom:3 returns 3 courses", len(bottom3) == 3)
for code, name in bottom3:
    print(f"\t{code}={diff[code]}: {name}")
print()    

top100 = bot._filter_courses(courses, ["top:100"], diff)
check("top:100 returns all courses", len(top100) == len(courses))


#-----------------------------------------
section("Combined filters")
top10_hard = bot._filter_courses(courses, ["hard", "bottom:10"], diff)
check("hard bottom:10 — all H suffix", all(c.endswith("H") for c, _ in top10_hard))
check("hard bottom:10 — exactly 10", len(top10_hard) == 10, str([(c, diff[c]) for c, _ in top10_hard]))

easy_no_oge = bot._filter_courses(courses, ["easy", "-OGE"], diff)
check("easy -OGE excludes OGE", not any(c == "OGE" for c, _ in easy_no_oge))
check("easy -OGE keeps 39 easy courses (not tripped by Bogey's)", len(easy_no_oge) == len(easy)-1, str(len(easy_no_oge)))

easy_no_oge2 = bot._filter_courses(courses, ["easy", '-"Original Gothic easy"'], diff)
check('easy -"Original Gothic easy" excludes OGE', easy_no_oge == easy_no_oge2)

hard_no_multiple = bot._filter_courses(courses, ["hard", "-BBH", '-"8-bit lair"', '-"meow wolf"'], diff)
check('hard -BBH -"8-bit lair" -"meow" excludes BBH, A8H, MCH', not any(c in ["BBH", "A8H", "MCH"] for c, _ in hard_no_multiple))

#-----------------------------------------
section("Difficulty word in name query")
journey_easy = bot._filter_courses(courses, ["journey easy"], diff)
check("'journey easy' matches only JCE", len(journey_easy) == 1 and journey_easy[0][0] == "JCE",
      str(journey_easy))

journey_hard = bot._filter_courses(courses, ["journey hard"], diff)
check("'journey hard' matches only JCH", len(journey_hard) == 1 and journey_hard[0][0] == "JCH",
      str(journey_hard))

atlantis_easy = bot._filter_courses(courses, ["atlantis easy"], diff)
check("'atlantis easy' matches only ATE", len(atlantis_easy) == 1 and atlantis_easy[0][0] == "ATE",
      str(atlantis_easy))

journey_dash_hard = bot._filter_courses(courses, ["journey - hard"], diff)
check("'journey - hard' matches only JCH", len(journey_dash_hard) == 1 and journey_dash_hard[0][0] == "JCH",
      str(journey_dash_hard))

journey_dashnospace = bot._filter_courses(courses, ["journey -hard"], diff)
check("'journey -hard' matches only JCH (no space before hard)", len(journey_dashnospace) == 1 and journey_dashnospace[0][0] == "JCH",
      str(journey_dashnospace))


#-----------------------------------------
section("Just course codes and/or names")

code_only = bot._filter_courses(courses, ["clh", "8bh"], diff)
check("clh and 8bh return correct courses", set(c for c, _ in code_only) == {"CLH", "8BH"}, str(code_only))

name_only = bot._filter_courses(courses, ["gardens", "journey"], diff)
check("gardens and journey return correct courses",
      set(c for c, _ in name_only) == {"MGE", "MGH", "GBE", "GBH", "JCE", "JCH"}, str(name_only))

print()
