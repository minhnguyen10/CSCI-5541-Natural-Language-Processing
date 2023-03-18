import re
# from tkinter.font import BOLD

def matching(searchstring, regex):
  re.match(searchstring,regex)

def main():
  notexit = True
  print("Hello. Please tell me about your problems.")
  while notexit:
    userint = input("Type input: ").lower()
    if re.match("^goodbye$",userint):
      print("Goodbye!")
      notexit=False
    else:
      # replacements = [
      #   (r"^yes$", "I see"),
      #   (r"^no$", "Why not?"),
      #   (r".* you .*", "Let's not talk about me."),
      #   (r"what is ([a-z ]+)?\?*.*$", r"Why do you ask about \1?"),
      #   (r"i am ([a-z ]+).*$", r"Do you enjoy being \1?"),
      #   (r"why is ([a-z ]+)\?*.*$", r"Why do you think \1?"),
      #   (r"i am ([1-9]{1,2}) years old", r"Are you \1 years old?"), #additional
      #   (r"my name is (\w+) (\w+)$", r"Nice to meet you \1. Is your last name \2?"), #additional
      #   (r"i am ([1-7]{1})ft([1-9]|1[0-2])in$", r"Are you \1 feet and \2 inch tall?, that's impressive."), #addition
      #   (r"my email is ([a-z0-9]+@[a-z]+(\.[a-z]{2,})+)$", r"Is \1 is the best way to contact you?"), #additional
      #   (r"my", r"Your"),
      # ]
      # sub = userint
      # for old, new in replacements:
      #   sub = re.sub(old, new, sub)
      sub = re.sub(r"^yes$", "I see", userint)
      sub = re.sub(r"^no$", "Why not?", sub)
      sub = re.sub(r".* you .*", "Let's not talk about me.", sub)
      sub = re.sub(r"what is ([a-z ]+)?\?*.*$", r"Why do you ask about \1?", sub)
      sub = re.sub(r"i am ([a-z ]+).*$", r"Do you enjoy being \1?", sub)
      sub = re.sub(r"why is ([a-z ]+)\?*.*$", r"Why do you think \1?", sub)
      sub = re.sub(r"i am ([1-9]{1,2}) years old", r"Are you \1 years old?", sub) #additional
      sub = re.sub(r"my name is (\w+) (\w+)$", r"Nice to meet you \1. Is your last name \2?", sub) #additional
      sub = re.sub(r"i am ([1-7]{1})ft([1-9]|1[0-2])in$", r"Are you \1 feet and \2 inch tall?, that's impressive.", sub) #additional
      sub = re.sub(r"my email is ([a-z0-9]+@[a-z]+(\.[a-z]{2,})+)$", r"Is \1 is the best way to contact you?", sub) #additional
      sub = re.sub(r"my", r"Your", sub)
      
      if re.match(userint, sub):
        sub = ("Please go on.")
      print(sub)
      pass

if __name__ == '__main__':
  main()