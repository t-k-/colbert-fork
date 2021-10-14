import os
import math
import torch
import ujson
import traceback

from itertools import accumulate
from colbert.parameters import DEVICE
from colbert.utils.utils import print_message, dotdict, flatten

BSIZE = 1 << 14


class IndexRanker():
    def __init__(self, tensor, doclens):
        self.tensor = tensor # Big document words tensor
        print('tensor.shape', tensor.shape) # torch.Size([4469627, 128])
        self.doclens = doclens # of length 69905

        self.maxsim_dtype = torch.float32
        self.doclens_pfxsum = [0] + list(accumulate(self.doclens)) # CDF?

        self.doclens = torch.tensor(self.doclens) # tensorize
        self.doclens_pfxsum = torch.tensor(self.doclens_pfxsum) # tensorize

        self.dim = self.tensor.size(-1) # 128

        self.strides = [torch_percentile(self.doclens, p) for p in [90]]
        self.strides.append(self.doclens.max().item())
        self.strides = sorted(list(set(self.strides))) # [99, 180]
        print_message(f"#> Using strides {self.strides}..")

        self.views = self._create_views(self.tensor)
        #self.buffers = self._create_buffers(BSIZE, self.tensor.dtype, {'cpu', 'cuda:0'})

    def _create_views(self, tensor):
        views = []

        for stride in self.strides: # 99, 180
            outdim = tensor.size(0) - stride + 1 # 4469627 - 99/180 + 1
            print('outdim', outdim) # 4469529, 4469448
            view = torch.as_strided(tensor,  # [4469627, 128]
                (outdim, stride, self.dim), (self.dim, self.dim, 1))
            #4469529/.., 99/180,   128,       128,      128,     1
            print(f'view.shape[i]', view.shape) # [4469529/.., 99/180, 128]
            views.append(view)

        # Example
        # tensor([[ 1.8278, -1.8511],
        #         [ 1.2551,  1.7123],
        #         [-0.4915,  0.6947],
        #         [ 2.3282,  1.8772]])
        # dim = 2
        # stride = 2 # group size. Try to adjust it and see what will happen...
        # outdim = tensor.size(0) - stride + 1 # which equals to 3
        # view = torch.as_strided(tensor, (outdim, stride, dim), (dim, dim, 1))
        #
        # tensor([[[ 1.8278, -1.8511],
        #          [ 1.2551,  1.7123]],
        #         [[ 1.2551,  1.7123],
        #          [-0.4915,  0.6947]],
        #         [[-0.4915,  0.6947],
        #          [ 2.3282,  1.8772]]])

        return views

    #def _create_buffers(self, max_bsize, dtype, devices):
    #    buffers = {}

    #    for device in devices:
    #        buffers[device] = [torch.zeros(max_bsize, stride, self.dim, dtype=dtype,
    #                                       device=device, pin_memory=(device == 'cpu'))
    #                           for stride in self.strides]

    #    return buffers

    def rank(self, Q, pids, views=None, shift=0):
        assert len(pids) > 0
        assert Q.size(0) in [1, len(pids)]

        Q = Q.contiguous().to(DEVICE).to(dtype=self.maxsim_dtype)

        # views is None
        views = self.views if views is None else views
        VIEWS_DEVICE = views[0].device

        #D_buffers = self.buffers[str(VIEWS_DEVICE)]

        raw_pids = pids if type(pids) is list else pids.tolist()
        pids = torch.tensor(pids) if type(pids) is list else pids

        # length == 6249
        doclens, offsets = self.doclens[pids], self.doclens_pfxsum[pids]

        a = doclens.unsqueeze(1) # torch.Size([6249, 1])
        b = torch.tensor(self.strides).unsqueeze(0) # torch.Size([1, 2])
        assignments = (a > b + 1e-6) # [6249, 2]
        assignments = assignments.sum(-1) # [6249]
        print('assignments', assignments) # how many docs exceeds 90 percentile

        one_to_n = torch.arange(len(raw_pids))
        output_pids, output_scores, output_permutation = [], [], []

        # divde by doclen, from [99, 180]
        for group_idx, stride in enumerate(self.strides):
            locator = (assignments == group_idx)

            if locator.sum() < 1e-5:
                continue

            # when group_idx == 0, locator are docs less than 90-percentile
            # when group_idx == 1, locator are docs more than 90-percentile

            group_pids, group_doclens, group_offsets = pids[locator], doclens[locator], offsets[locator]
            group_Q = Q if Q.size(0) == 1 else Q[locator]
            #print('0', group_Q.shape) # torch.Size([1, 128, 32])
            #print('0', group_doclens.shape) # torch.Size([5510])

            group_offsets = group_offsets.to(VIEWS_DEVICE) - shift
            # output,             inverse_indices (where elements in the original input map to in the output)

            #print(pids)
            #print(group_pids) # subset of pids
            #print(group_doclens)
            #print(group_offsets) # subset of offsets, same size as group_pids
            group_offsets_uniq, group_offsets_expand = torch.unique_consecutive(group_offsets, return_inverse=True)
            #print(group_offsets_uniq) # equal pfxsum means same document
            #print(group_offsets_expand) # inverted indices for group_offsets_uniq
            #print()

            D_size = group_offsets_uniq.size(0)
            #print('1', views[group_idx].shape) # torch.Size([4469529, 99, 128]

            #  view[1] = [4469529,  99, 128],       dim      selects
            #  view[2] = [4469448, 180, 128],       dim      selects
            # What is in view? words' embeddings.

            #D = torch.index_select(views[group_idx], 0, group_offsets_uniq, out=D_buffers[group_idx][:D_size])
            D = torch.index_select(views[group_idx], 0, group_offsets_uniq)
            #print('2', D.shape) # torch.Size([5510, 99, 128])

            D = D.to(DEVICE)
            D = D[group_offsets_expand.to(DEVICE)].to(dtype=self.maxsim_dtype)
            # in rare cases, D contains some identical offsets
            #print('3', D.shape) # torch.Size([5510, 99, 128])

            mask_ = torch.arange(stride, device=DEVICE) + 1
            # mask_ = tensor([1, 2, 3 ... 99/180])
            # mask_.unsqueeze(0) = tensor([[1, 2, 3 ... 99/180]])
            mask = mask_.unsqueeze(0) <= group_doclens.to(DEVICE).unsqueeze(-1)
            #print('mask.shape', mask.shape) # torch.Size([5510, 99])

            # [5510, 99, 128] @ [1, 128, 32]
            scores = D @ group_Q
            #print(mask_)
            #print(group_doclens)
            #print(mask.unsqueeze(-1)) # each row is identical
            #print(scores)
            #print(scores * mask.unsqueeze(-1))
            #print()

            #print('4', scores.shape) # torch.Size([5510, 99, 32])

            scores = scores * mask.unsqueeze(-1)
            scores = scores.max(1).values.sum(-1).cpu()
            #print('5', scores.shape) # torch.Size([5510])

            output_pids.append(group_pids)
            output_scores.append(scores)
            output_permutation.append(one_to_n[locator])

        output_permutation = torch.cat(output_permutation).sort().indices
        output_pids = torch.cat(output_pids)[output_permutation].tolist()
        output_scores = torch.cat(output_scores)[output_permutation].tolist()

        assert len(raw_pids) == len(output_pids)
        assert len(raw_pids) == len(output_scores)
        assert raw_pids == output_pids

        return output_scores

    def batch_rank(self, all_query_embeddings, all_query_indexes, all_pids, sorted_pids):
        assert sorted_pids is True

        ######

        scores = []
        range_start, range_end = 0, 0

        for pid_offset in range(0, len(self.doclens), 50_000):
            pid_endpos = min(pid_offset + 50_000, len(self.doclens))

            range_start = range_start + (all_pids[range_start:] < pid_offset).sum()
            range_end = range_end + (all_pids[range_end:] < pid_endpos).sum()

            pids = all_pids[range_start:range_end]
            query_indexes = all_query_indexes[range_start:range_end]

            print_message(f"###--> Got {len(pids)} query--passage pairs in this sub-range {(pid_offset, pid_endpos)}.")

            if len(pids) == 0:
                continue

            print_message(f"###--> Ranking in batches the pairs #{range_start} through #{range_end} in this sub-range.")

            tensor_offset = self.doclens_pfxsum[pid_offset].item()
            tensor_endpos = self.doclens_pfxsum[pid_endpos].item() + 512

            collection = self.tensor[tensor_offset:tensor_endpos].to(DEVICE)
            views = self._create_views(collection)

            print_message(f"#> Ranking in batches of {BSIZE} query--passage pairs...")

            for batch_idx, offset in enumerate(range(0, len(pids), BSIZE)):
                if batch_idx % 100 == 0:
                    print_message("#> Processing batch #{}..".format(batch_idx))

                endpos = offset + BSIZE
                batch_query_index, batch_pids = query_indexes[offset:endpos], pids[offset:endpos]

                Q = all_query_embeddings[batch_query_index]

                scores.extend(self.rank(Q, batch_pids, views, shift=tensor_offset))

        return scores


def torch_percentile(tensor, p):
    assert p in range(1, 100+1)
    assert tensor.dim() == 1
    # p = 90
    # tensor: torch.Size([69905])
    _90_percentile = int(p * tensor.size(0) / 100.0) # 62914
    # kth greatest value
    kth_val = tensor.kthvalue(_90_percentile) #  values=99, indices=59498
    return kth_val.values.item()
