import os
import time
import faiss
import random
import torch

from multiprocessing import Pool
from colbert.modeling.inference import ModelInference

from colbert.utils.utils import print_message, flatten, batch
from colbert.indexing.loaders import load_doclens


class FaissIndex():
    def __init__(self, index_path, faiss_index_path, nprobe, part_range=None):
        print(locals())
        print_message("#> Loading the FAISS index from", faiss_index_path, "..")

        faiss_part_range = os.path.basename(faiss_index_path).split('.')[-2].split('-')
        print(faiss_part_range) # ['32768']

        if len(faiss_part_range) == 2:
            faiss_part_range = range(*map(int, faiss_part_range))
            assert part_range[0] in faiss_part_range, (part_range, faiss_part_range)
            assert part_range[-1] in faiss_part_range, (part_range, faiss_part_range)
        else:
            faiss_part_range = None

        self.part_range = part_range
        self.faiss_part_range = faiss_part_range

        self.faiss_index = faiss.read_index(faiss_index_path)
        self.faiss_index.nprobe = nprobe

        print_message("#> Building the emb2pid mapping..")
        all_doclens = load_doclens(index_path, flatten=False)
        # from doclens.k.json

        pid_offset = 0
        if faiss_part_range is not None:
            print(f"#> Restricting all_doclens to the range {faiss_part_range}.")
            pid_offset = len(flatten(all_doclens[:faiss_part_range.start]))
            all_doclens = all_doclens[faiss_part_range.start:faiss_part_range.stop]

        self.relative_range = None
        if self.part_range is not None:
            start = self.faiss_part_range.start if self.faiss_part_range is not None else 0
            a = len(flatten(all_doclens[:self.part_range.start - start]))
            b = len(flatten(all_doclens[:self.part_range.stop - start]))
            self.relative_range = range(a, b)
            print(f"self.relative_range = {self.relative_range}")

        all_doclens = flatten(all_doclens)

        total_num_embeddings = sum(all_doclens) # total number of words
        self.emb2pid = torch.zeros(total_num_embeddings, dtype=torch.int)

        # e.g., emb29 emb30 emb31 are from doc89
        offset_doclens = 0
        for pid, dlength in enumerate(all_doclens):
            self.emb2pid[offset_doclens: offset_doclens + dlength] = pid_offset + pid
            offset_doclens += dlength

        print_message("len(self.emb2pid) =", len(self.emb2pid))

        self.parallel_pool = Pool(16)

    def retrieve(self, faiss_depth, Q, verbose=False):
        embedding_ids = self.queries_to_embedding_ids(faiss_depth, Q, verbose=verbose)
        #print('EMB', embedding_ids.shape) # 6249

        # map embedding position to uniq PIDs
        pids = self.embedding_ids_to_pids(embedding_ids, verbose=verbose)

        #print('Passages', len(pids[0])) # 6249

        if self.relative_range is not None:
            pids = [[pid for pid in pids_ if pid in self.relative_range] for pids_ in pids]

        return pids

    def queries_to_embedding_ids(self, faiss_depth, Q, verbose=True):
        # Flatten into a matrix for the faiss search.
        num_queries, embeddings_per_query, dim = Q.size() # 1, 32, 128

        # break down into query keywords
        Q_faiss = Q.view(num_queries * embeddings_per_query, dim).cpu().contiguous()

        # Search in large batches with faiss.
        print_message("#> Search in batches with faiss. ",
                      f"Q.size() = {Q.size()}, Q_faiss.size() = {Q_faiss.size()}",
                      condition=verbose)

        embeddings_ids = []
        faiss_bsize = embeddings_per_query * 5000 # 32 * 5000
        # almost always in one batch
        for offset in range(0, Q_faiss.size(0), faiss_bsize):
            endpos = min(offset + faiss_bsize, Q_faiss.size(0))
            # from 0 to 32
            print_message("#> Searching from {} to {}...".format(offset, endpos), condition=verbose)

            some_Q_faiss = Q_faiss[offset:endpos].float().numpy()
            _, some_embedding_ids = self.faiss_index.search(some_Q_faiss, faiss_depth)
            print('>>SEARCH FAISS<<', faiss_depth, some_embedding_ids.shape)
            embeddings_ids.append(torch.from_numpy(some_embedding_ids))

        embedding_ids = torch.cat(embeddings_ids)

        # Reshape to (number of queries, non-unique embedding IDs per query)
        embedding_ids = embedding_ids.view(num_queries, embeddings_per_query * embedding_ids.size(1))

        return embedding_ids

    def embedding_ids_to_pids(self, embedding_ids, verbose=True):
        # Find unique PIDs per query.
        print_message("#> Lookup the PIDs..", condition=verbose)
        all_pids = self.emb2pid[embedding_ids] # like numpy index arrays

        print_message(f"#> Converting to a list [shape = {all_pids.size()}]..", condition=verbose)
        all_pids = all_pids.tolist()

        print_message("#> Removing duplicates (in parallel if large enough)..", condition=verbose)

        # see the bottom of this file for uniq() definition
        if len(all_pids) > 5000:
            all_pids = list(self.parallel_pool.map(uniq, all_pids))
        else:
            all_pids = list(map(uniq, all_pids))

        print_message("#> Done with embedding_ids_to_pids().", condition=verbose)

        return all_pids


def uniq(l):
    return list(set(l))
