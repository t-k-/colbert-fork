import os
import torch
import ujson

from math import ceil
from itertools import accumulate
from colbert.utils.utils import print_message, dotdict, flatten

from colbert.indexing.loaders import get_parts, load_doclens
from colbert.indexing.index_manager import load_index_part
from colbert.ranking.index_ranker import IndexRanker


class IndexPart():
    def __init__(self, directory, dim=128, part_range=None, verbose=True):
        print('IndexPart', locals())

        first_part, last_part = (0, None) if part_range is None else (part_range.start, part_range.stop)

        # Load parts metadata
        all_parts, all_parts_paths, _ = get_parts(directory)
        print('all_parts', all_parts) # [0]
        print('all_parts_paths', all_parts_paths) # MSMARCO-tiny/0.pt
        print('first_part, last_part', first_part, last_part) # 0, None
        self.parts = all_parts[first_part:last_part]
        self.parts_paths = all_parts_paths[first_part:last_part]

        # Load doclens metadata
        all_doclens = load_doclens(directory, flatten=False)

        print('all_doclens[0] length', len(all_doclens[0]))

        self.doc_offset = sum([len(part_doclens) for part_doclens in all_doclens[:first_part]])
        print('self.doc_offset', self.doc_offset)

        self.doc_endpos = sum([len(part_doclens) for part_doclens in all_doclens[:last_part]])
        print('self.doc_endpos', self.doc_endpos)
        self.pids_range = range(self.doc_offset, self.doc_endpos)

        self.parts_doclens = all_doclens[first_part:last_part]
        self.doclens = flatten(self.parts_doclens)
        print('self.doclens.length', len(self.doclens))
        self.num_embeddings = sum(self.doclens)
        print('self.num_embeddings', self.num_embeddings)

        # load document code/tensor
        self.tensor = self._load_parts(dim, verbose)

        print('>>LOADED TENSOR<<', self.tensor.shape)

        self.ranker = IndexRanker(self.tensor, self.doclens)

    def _load_parts(self, dim, verbose):
        tensor = torch.zeros(self.num_embeddings + 512, dim, dtype=torch.float16)

        if verbose:
            print_message("tensor.size() = ", tensor.size())

        offset = 0
        for idx, filename in enumerate(self.parts_paths):
            print_message("|> Loading", filename, "...", condition=verbose)

            endpos = offset + sum(self.parts_doclens[idx])
            part = load_index_part(filename, verbose=verbose)

            #print(part.shape) # torch.Size([133, 128])
            #print(tensor.shape, offset, endpos) # torch.Size([133+512=645, 128]), 0, 133

            tensor[offset:endpos] = part
            offset = endpos

        return tensor

    # NOT being used?
    def pid_in_range(self, pid):
        return pid in self.pids_range

    def rank(self, Q, pids):
        """
        Rank a single batch of Q x pids (e.g., 1k--10k pairs).
        """

        assert Q.size(0) in [1, len(pids)], (Q.size(0), len(pids))
        ## print(self.pids_range)
        ## print(pids)
        assert all(pid in self.pids_range for pid in pids), self.pids_range

        # GET relative PIDs
        pids_ = [pid - self.doc_offset for pid in pids]

        # USE IndexRanker module
        scores = self.ranker.rank(Q, pids_)

        return scores

    def batch_rank(self, all_query_embeddings, query_indexes, pids, sorted_pids):
        """
        Rank a large, fairly dense set of query--passage pairs (e.g., 1M+ pairs).
        Higher overhead, much faster for large batches.
        """

        assert ((pids >= self.pids_range.start) & (pids < self.pids_range.stop)).sum() == pids.size(0)

        pids_ = pids - self.doc_offset
        scores = self.ranker.batch_rank(all_query_embeddings, query_indexes, pids_, sorted_pids)

        return scores
