# Prompt Log
## a hamburger
Train

``
python main.py --text "a hamburger" --workspace hamburger -O
``

Mesh

``
python main.py --text "a hamburger" --workspace hamburger -O --save_mesh 
``
## a high quality photo of a pineapple
Train

``
python main.py --text "a high quality photo of a pineapple" --workspace pineapple -O
``


Mesh

``
python main.py --text "a high quality photo of a pineapple" --workspace pineapple -O  --iters 150000
``

## a DSLR photo of  a bear in a tutu dancing ballet

``
python main.py --text "a DSLR photo of  a bear in a tutu dancing ballet" --workspace bear -O
``
## a high quality photo of  a chimpanzee playing football


``
python main.py --text "a high quality photo of a white unicorn" --workspace unicorn --backbone vanilla --guidance stable-diffusion --albedo
``
