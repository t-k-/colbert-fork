python -m colbert.retrieve --amp --doc_maxlen 180 --mask-punctuation --bsize 8 --queries ./data_download/queries.dev.tsv --faiss_depth 1024 --index_root ./output/indexes/ --index_name MSMARCO-tiny --checkpoint ./output/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert.dnn --root ./output/ --experiment MSMARCO-psg --partitions 32768

python -m colbert.index_faiss --index_root ./output/indexes/ --index_name MSMARCO-tiny --root ./output/ --experiment MSMARCO-psg

python -m colbert.index --amp --doc_maxlen 180 --mask-punctuation --bsize 8 --checkpoint output/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert.dnn --collection data_download/collection.tsv --index_root output/indexes --index_name MSMARCO-fromscratch --root ./output/ --experiment MSMARCO-psg --chunksize 3

python -m colbert.test --amp --mask-punctuation --collection ./data_download/collection.tsv --queries ./data_download/my-queries.tsv --checkpoint ../pyserini/encoders/colbert-400000.dnn --root ./output/ --experiment MSMARCO-psg --topk data_download/my-test.tsv

python -m colbert.train --amp --doc_maxlen 180 --mask-punctuation --bsize 8 --accum 1 --triples ./data_download/triples.train.small.tsv --root ./output/ --experiment MSMARCO-psg --similarity l2 --run msmarco.psg.l2
