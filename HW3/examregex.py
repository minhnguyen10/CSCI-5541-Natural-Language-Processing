import re
# from tkinter.font import BOLD

def matching(searchstring, regex):
  re.match(searchstring,regex)

def main():
  notexit = True
  print("Hello. Please tell me about your problems.")
  while notexit:
    userint = input("Type input: ").lower()
    if re.match("^exit$",userint):
      notexit=False
    else:
        b = re.search(r"[a-z]*[xz] *([a-z]+)",userint)
        c = re.search(r"\b([a-z]+ed|ing)\b",userint)
        print(b.group(1) if b is not None else 'Not found')


if __name__ == '__main__':
  main()