import os
import re
import sys

import polib

if len(sys.argv) < 2:
    print("\nWRONG:insufficient arguments.")
    print("python trans_summary.py you_po_file_folder/")
else:
    poloc = sys.argv[1]
    print(">>>", poloc)

po_files = []
for root, dirs, files in os.walk(poloc, topdown=False):
    for name in files:
        po_files.append(name)


def check_match(s):
    fpattern = re.compile(".+\.po")
    return fpattern.match(s) is not None


def highlighter(s, type=""):
    if type == "":
        return "\033[0;36m " + s + " \033[0m"
    elif type == "filename":
        return "{:<60}".format("\033[0;33m " + s + " \033[0m")
    elif type == "hint":
        return "\033[0;34m " + s + " \033[0m"
    elif type == "hint2":
        return "\033[0;32m " + s + " \033[0m"
    elif type == "hint3":
        return "\033[0;35;46m " + s + " \033[0m"
    elif type == "state":
        return "{:<10}".format("\033[1;37;44m " + s + " \033[0m")
    elif type == "state2":
        return "{:<10}".format("\033[1;37;43m " + s + " \033[0m")


po_files = list(filter(check_match, po_files))

print(highlighter(str(len(po_files)) + " po files detected"))
print(highlighter(" obsolete_entries are omitted as they are abondoned"))

allpo_entry_num = 0
allpo_entry_num_trans = 0


for pofname in po_files:
    pof = polib.pofile(poloc + pofname, encoding="utf-8")
    # ignore all cleared files and megengine.core files
    if pof.percent_translated() < 100 and "megengine.core" not in pofname:
        print(highlighter(pofname, "filename"), end="")

        allpo_entry_num += len(pof) - len(pof.obsolete_entries())
        allpo_entry_num_trans += len(pof.translated_entries())
        single_entry = len(pof) - len(pof.obsolete_entries())
        print(highlighter("该po条目总数：", "hint"), end="")
        print("{:<4}".format(single_entry), end="")
        print(highlighter("该po待翻译数：", "hint2"), end="")
        print(" {:<4}".format(single_entry - len(pof.translated_entries())), end=" ")
        print(highlighter("该po完成度：", "state"), end="")
        print(str(pof.percent_translated()) + "%")


print(highlighter("总条目：", "hint3"), end="")
print(str(allpo_entry_num_trans) + "/" + str(allpo_entry_num))

print(highlighter("总翻译进度：", "hint3"), end="")
print("{:=.2f}".format((allpo_entry_num_trans / allpo_entry_num) * 100) + "%")
