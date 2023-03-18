Note:
We are assuming that the words corpus is in ".\" . If not, please change the location of the words corpus.
For subsetting word dictionary, we impose the following strategies:
  - Limit the first letter to only itself and surrounding letters on the keyboard layout.
  - Limit the corrected word length to be with ±2 of the original word.

We only consider a word to be in the suggestions if the edit distance is below 5

For command line usage, it requires an text file input, if no output file is specified, it will save the output under the same directory with prefix "corrected_", the "-x" option will allow user to choose one word to replace the original word from suggestion list or skip correction. By default, the program will only pick the first word from suggestion list to replace.




usage: spellchecker.py [-h] [-o OUTPUT] [-x] input     

Positional arguments:                               
  input                  

Optional arguments:                                    
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT                            
                        output filename                 
  -x, --interactive     interactive model     
