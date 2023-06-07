# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

import torch

from fairseq import utils
from fairseq.data import LanguagePairDataset
from fairseq.dataclass import ChoiceEnum
from fairseq.tasks import register_task
from fairseq.tasks.translation import (
    TranslationConfig,
    TranslationTask,
    load_langpair_dataset,
)
from fairseq.utils import new_arange

NOISE_CHOICES = ChoiceEnum(["random_delete", "random_mask", "no_noise", "full_mask"])


@dataclass
class TranslationLevenshteinConfig(TranslationConfig):
    noise: NOISE_CHOICES = field(
        default="random_delete",
        metadata={"help": "type of noise"},
    )


@register_task("translation_lev", dataclass=TranslationLevenshteinConfig)
class TranslationLevenshteinTask(TranslationTask):
    """
    Translation (Sequence Generation) task for Levenshtein Transformer
    See `"Levenshtein Transformer" <https://arxiv.org/abs/1905.11006>`_.
    """

    cfg: TranslationLevenshteinConfig

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]
        indices = kwargs.get('indices', None)

        # infer langcode
        src, tgt = self.cfg.source_lang, self.cfg.target_lang
        self.datasets[split] = load_langpair_dataset(
            data_path,
            split if indices is None else 'valid',
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            combine=combine,
            dataset_impl=self.cfg.dataset_impl,
            upsample_primary=self.cfg.upsample_primary,
            left_pad_source=self.cfg.left_pad_source,
            left_pad_target=self.cfg.left_pad_target,
            max_source_positions=self.cfg.max_source_positions,
            max_target_positions=self.cfg.max_target_positions,
            prepend_bos=True,
            indices=indices
        )

    def inject_noise(self, target_tokens, *, step_count=-1, step_weight_temp=1.):
        def _random_delete(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()

            max_len = target_tokens.size(1)
            target_mask = target_tokens.eq(pad)
            target_score = target_tokens.clone().float().uniform_()
            target_score.masked_fill_(
                target_tokens.eq(bos) | target_tokens.eq(eos), 0.0
            )
            target_score.masked_fill_(target_mask, 1)
            target_score, target_rank = target_score.sort(1)
            target_length = target_mask.size(1) - target_mask.float().sum(
                1, keepdim=True
            )

            # do not delete <bos> and <eos> (we assign 0 score for them)
            target_cutoff = (
                2
                + (
                    (target_length - 2)
                    * target_score.new_zeros(target_score.size(0), 1).uniform_()
                ).long()
            )
            target_cutoff = target_score.sort(1)[1] >= target_cutoff

            prev_target_tokens = (
                target_tokens.gather(1, target_rank)
                .masked_fill_(target_cutoff, pad)
                .gather(1, target_rank.masked_fill_(target_cutoff, max_len).sort(1)[1])
            )
            prev_target_tokens = prev_target_tokens[
                :, : prev_target_tokens.ne(pad).sum(1).max()
            ]
            return prev_target_tokens


        def _random_mask(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()
            unk = self.tgt_dict.unk()

            target_masks = (
                target_tokens.ne(pad) & target_tokens.ne(bos) & target_tokens.ne(eos)
            )
            target_score = target_tokens.clone().float().uniform_()
            target_score.masked_fill_(~target_masks, 2.0)
            target_length = target_masks.sum(1).float()
            discrete_step_num=-1
            if step_count == -1:
                target_length = target_length * target_length.clone().uniform_()
                target_length = target_length + 1  # make sure to mask at least one token. 
            else:
                categorical=torch.distributions.categorical.Categorical(logits=self.step_weights*step_weight_temp)
                discrete_step_num=categorical.sample(sample_shape=(len(target_length),)).to(target_length.device)
                target_length = target_length * (discrete_step_num+1)/step_count

            _, target_rank = target_score.sort(1)
            target_cutoff = new_arange(target_rank) < target_length[:, None].long()
            prev_target_tokens = target_tokens.masked_fill(
                target_cutoff.scatter(1, target_rank, target_cutoff), unk
            )
            return prev_target_tokens, discrete_step_num

        def _full_mask(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()
            unk = self.tgt_dict.unk()

            target_mask = (
                target_tokens.eq(bos) | target_tokens.eq(eos) | target_tokens.eq(pad)
            )
            return target_tokens.masked_fill(~target_mask, unk), -1

        if self.cfg.noise == "random_delete":
            return _random_delete(target_tokens)
        elif self.cfg.noise == "random_mask":
            return _random_mask(target_tokens)
        elif self.cfg.noise == "full_mask":
            return _full_mask(target_tokens)
        elif self.cfg.noise == "no_noise":
            return target_tokens
        else:
            raise NotImplementedError

    def build_generator(self, models, args, **unused):
        # add models input to match the API for SequenceGenerator
        from fairseq.iterative_refinement_generator import IterativeRefinementGenerator

        return IterativeRefinementGenerator(
            self.target_dictionary,
            eos_penalty=getattr(args, "iter_decode_eos_penalty", 0.0),
            max_iter=getattr(args, "iter_decode_max_iter", 10),
            beam_size=getattr(args, "iter_decode_with_beam", 5),
            reranking=getattr(args, "iter_decode_with_external_reranker", False),
            decoding_format=getattr(args, "decoding_format", None),
            adaptive=False,
            retain_history=getattr(args, "retain_iter_history", False),
            bleu_reranker=getattr(args, "bleu_reranker", False),
            deduplicate=True,
            blank_symbol=getattr(self, "blank_symbol", None)
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        if constraints is not None:
            # Though see Susanto et al. (ACL 2020): https://www.aclweb.org/anthology/2020.acl-main.325/
            raise NotImplementedError(
                "Constrained decoding with the translation_lev task is not supported"
            )

        return LanguagePairDataset(
            src_tokens, src_lengths, self.source_dictionary, append_bos=True
        )

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        model.train()
        if getattr(self.cfg, 'ctc', False):
            sample['target']=sample['target'][:, 1:]
            sample['target'][sample['target']==self.target_dictionary.eos()]=self.target_dictionary.pad()
            if not getattr(self.cfg, 'ctc_distill', False):
                model.args.noise='full_mask' if update_num<getattr(model.args, 'ctc_pretrain_step', 0) else  model.args.noise


        step_count=getattr(model.args, 'step_count', -1)
        if step_count!=-1:
            self.step_weights=getattr(self, 'step_weights', torch.ones(step_count, device=sample["target"].device)*1e-5)
        sample["prev_target"], sample["discrete_step_num"] = self.inject_noise(sample["target"], step_count=step_count, step_weight_temp=getattr(model.args, 'step_weight_temp', 0.))
        teacher_temp=getattr(model.args, 'teacher_temp', 1.)
        std_temp=getattr(model.args, 'std_temp', 1.)
        

        losses, loss, sample_size, logging_output = criterion(model, sample, distill_temp=(std_temp, teacher_temp))
        # losses, masks=losses['losses'], losses['masks']

        if getattr(model.args, 'mid_mask_policy', 'half') == 'discrete':
            group_indices=sample["discrete_step_num"].unsqueeze(-1).expand(-1, masks.shape[-1])[masks].unsqueeze(-1).expand(-1, step_count) == torch.arange(step_count).unsqueeze(0).expand(len(losses), -1).to(losses.device)
            group_counts=group_indices.sum(0)
            new_step_weights=(group_indices*(losses.unsqueeze(-1))).sum(0)/(group_counts+1e-7) + (group_counts==0)*self.step_weights
            self.step_weights = (1-model.args.step_weight_update)*self.step_weights + model.args.step_weight_update*new_step_weights 

        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        if getattr(self.cfg, 'ctc', False):
            sample['target']=sample['target'][:, 1:]
            sample['target'][sample['target']==self.target_dictionary.eos()]=self.target_dictionary.pad()

        with torch.no_grad():
            sample["prev_target"], _ = self.inject_noise(sample["target"], step_count=getattr(model.args, 'step_count', -1), step_weight_temp=0)
            _, loss, sample_size, logging_output = criterion(model, sample)
        if self.cfg.eval_bleu:
            for step_count in [0, 7]:
                    self.sequence_generator.max_iter = step_count
                    bleu = self._inference_with_bleu(self.sequence_generator, sample, model)
                    logging_output[f"_bleu_sys_len_{step_count}"] = bleu.sys_len
                    logging_output[f"_bleu_ref_len_{step_count}"] = bleu.ref_len
                    # we split counts into separate entries so that they can be
                    # summed efficiently across workers using fast-stat-sync
                    EVAL_BLEU_ORDER = 4
                    assert len(bleu.counts) == EVAL_BLEU_ORDER
                    for i in range(EVAL_BLEU_ORDER):
                        logging_output[f"_bleu_counts_{step_count}_" + str(i)] = bleu.counts[i]
                        logging_output[f"_bleu_totals_{step_count}_" + str(i)] = bleu.totals[i]
        return loss, sample_size, logging_output
