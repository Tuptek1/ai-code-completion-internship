## Usage

To start using the script:

```
pip install transformers

# Installing torch takes a while
pip install torch

pip install sacrebleu

pip install rouge

pip install nltk
```

Run main.py and the script will output the completions and splits to the completions_output.txt based on the .py files in the example_source_codes folder.

You can freely modify the example_source_codes but remember to adjust the global variables in main.py to the average lenght of the functions, this step is needed for the script to work more properly