"""multi process to test"""
import os
import json
import time
import six
import math
import subprocess
import collections
import logging


class MultiProcessEval(object):
    """multi process test for classifiy tasks"""

    def __init__(self, output_path, eval_phase, dev_count, gpu_id):
        self.output_path = output_path
        self.eval_phase = eval_phase
        self.dev_count = dev_count
        self.gpu_id = gpu_id

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def write_result(self, eval_index, save_lists=None, name_list=None):
        """write result to hard disk"""
        outfile = self.output_path + "/" + self.eval_phase
        if len(eval_index) > 0:
            outfile_part = outfile + ".part" + str(self.gpu_id)
            writer = open(outfile_part, "w")
            write_content = "\t".join([str(i) for i in eval_index]) + "\n"
            writer.write(write_content)
            writer.close()
        if save_lists is not None and name_list is not None:
            #save_list_name = ["qids", "labels", "scores"]
            save_list_name = name_list
            for idx in range(len(save_list_name)):
                save_list = json.dumps(save_lists[idx])
                savefile_part = outfile + "." + save_list_name[idx] + ".part." + str(self.gpu_id)
                list_writer = open(savefile_part, "w")
                list_writer.write(save_list)
                list_writer.close()
        tmp_writer = open(self.output_path + "/" + self.eval_phase + "_dec_finish." + str(self.gpu_id), "w")
        tmp_writer.close()

    def concat_result(self, num_eval_index, num_list=None, name_list=None):
        """read result from hard disk and concat them"""
        outfile = self.output_path + "/" + self.eval_phase
        eval_index_all = [0.0] * num_eval_index
        eval_list_all = collections.defaultdict(list)
        while True:
            ret = subprocess.check_output(['find', self.output_path, '-maxdepth', '1', '-name',
                                              self.eval_phase + '_dec_finish.*'])
            if six.PY3:
                ret = ret.decode()
            ret = ret.rstrip().split("\n")
            if len(ret) != self.dev_count:
                time.sleep(1)
                continue
            for dev_cnt in range(self.dev_count):
                if num_eval_index > 0:
                    fin = open(outfile + ".part" + str(dev_cnt))
                    cur_eval_index_all = fin.readline().strip().split("\t")
                    cur_eval_index_all = [float(i) for i in cur_eval_index_all]
                    eval_index_all = list(map(lambda x: x[0] + x[1], zip(eval_index_all, cur_eval_index_all)))

                if num_list is not None and name_list is not None:
                    #save_list_name = ["qids", "labels", "scores"]
                    save_list_name = name_list
                    for idx in range(len(save_list_name)):
                        fin_list = open(outfile + "." + save_list_name[idx] + ".part." + str(dev_cnt), "r")
                        eval_list_all[save_list_name[idx]].extend(json.loads(fin_list.read()))

            #subprocess.check_output(["rm ", outfile + ".*part*"])
            #subprocess.check_output(["rm ", self.output_path + "/" + self.eval_phase + "_dec_finish.*"])
            os.system("rm " + outfile + ".*part*")
            os.system("rm " + self.output_path + "/" + self.eval_phase + "_dec_finish.*")
            break
        if num_list is not None:
            return eval_index_all, eval_list_all
        return eval_index_all


class MultiProcessEvalForMrc(object):
    """multi process test for mrc tasks"""
    def __init__(self, output_path, eval_phase, dev_count, gpu_id, tokenizer):
        self.output_path = output_path
        self.eval_phase = eval_phase
        self.dev_count = dev_count
        self.gpu_id = gpu_id
        self.tokenizer = tokenizer

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        if not os.path.exists("./output"):
            os.makedirs('./output')
        self.output_prediction_file = os.path.join('./output', self.eval_phase + "_predictions.json")
        self.output_nbest_file = os.path.join('./output', self.eval_phase + "_nbest_predictions.json")

    def write_result(self, all_results):
        """write result to hard disk"""
        outfile = self.output_path + "/" + self.eval_phase
        outfile_part = outfile + ".part" + str(self.gpu_id)
        writer = open(outfile_part, "w")
        save_dict = json.dumps(all_results)
        writer.write(save_dict)
        writer.close()
        tmp_writer = open(self.output_path + "/" + self.eval_phase + "_dec_finish." + str(self.gpu_id), "w")
        tmp_writer.close()


    def concat_result(self, RawResult):
        """read result from hard disk and concat them"""
        outfile = self.output_path + "/" + self.eval_phase
        all_results_read = []
        while True:
            _, ret = subprocess.getstatusoutput('find ' + self.output_path + \
                ' -maxdepth 1 -name ' + self.eval_phase + '_dec_finish.*')
            ret = ret.split("\n")
            if len(ret) != self.dev_count:
                time.sleep(1)
                continue

            for dev_cnt in range(self.dev_count):
                fin_read = open(outfile + ".part" + str(dev_cnt), "rb")
                cur_rawresult = json.loads(fin_read.read())
                for tp in cur_rawresult:
                    assert len(tp) == 3
                    all_results_read.append(
                        RawResult(
                            unique_id=tp[0],
                            start_logits=tp[1],
                            end_logits=tp[2]))

            #subprocess.getstatusoutput("rm " + outfile + ".part*")
            #subprocess.getstatusoutput("rm " + self.output_path + "/" + self.eval_phase + "_dec_finish.*")
            os.system("rm " + outfile + ".*part*")
            os.system("rm " + self.output_path + "/" + self.eval_phase + "_dec_finish.*")
            
            break

        return all_results_read

    def write_predictions(self, all_examples, all_features, all_results, n_best_size,
                          max_answer_length, do_lower_case, output_prediction_file,
                          output_nbest_file):
        """Write final predictions to the json file and log-odds of null if needed."""
        logging.info("Writing predictions to: %s" % (output_prediction_file))
        logging.info("Writing nbest to: %s" % (output_nbest_file))

        example_index_to_features = collections.defaultdict(list)
        for feature in all_features:
            example_index_to_features[feature.example_index].append(feature)

        unique_id_to_result = {}
        for result in all_results:
            unique_id_to_result[result.unique_id] = result

        _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "PrelimPrediction", [
                "feature_index", "start_index", "end_index", "start_logit",
                "end_logit"
            ])

        all_predictions = collections.OrderedDict()
        all_nbest_json = collections.OrderedDict()

        for (example_index, example) in enumerate(all_examples):
            features = example_index_to_features[example_index]

            prelim_predictions = []
            # keep track of the minimum score of null start+end of position 0
            for (feature_index, feature) in enumerate(features):
                result = unique_id_to_result[feature.unique_id]
                start_indexes = self._get_best_indexes(result.start_logits, n_best_size)
                end_indexes = self._get_best_indexes(result.end_logits, n_best_size)

                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # We could hypothetically create invalid predictions, e.g., predict
                        # that the start of the span is in the question. We throw out all
                        # invalid predictions.
                        if start_index >= len(feature.tokens):
                            continue
                        if end_index >= len(feature.tokens):
                            continue
                        if start_index not in feature.token_to_orig_map:
                            continue
                        if end_index not in feature.token_to_orig_map:
                            continue
                        if not feature.token_is_max_context.get(start_index, False):
                            continue
                        if end_index < start_index:
                            continue
                        length = end_index - start_index + 1
                        if length > max_answer_length:
                            continue
                        prelim_predictions.append(
                            _PrelimPrediction(
                                feature_index=feature_index,
                                start_index=start_index,
                                end_index=end_index,
                                start_logit=result.start_logits[start_index],
                                end_logit=result.end_logits[end_index]))

            prelim_predictions = sorted(
                prelim_predictions,
                key=lambda x: (x.start_logit + x.end_logit),
                reverse=True)

            _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
                "NbestPrediction", ["text", "start_logit", "end_logit"])

            seen_predictions = {}
            nbest = []
            for pred in prelim_predictions:
                if len(nbest) >= n_best_size:
                    break
                feature = features[pred.feature_index]
                if pred.start_index > 0:  # this is a non-null prediction
                    tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1
                                                                  )]
                    orig_doc_start = feature.token_to_orig_map[pred.start_index]
                    orig_doc_end = feature.token_to_orig_map[pred.end_index]
                    orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end +
                                                                     1)]
                    tok_text = " ".join(tok_tokens)

                    # De-tokenize WordPieces that have been split off.
                    tok_text = tok_text.replace(" ##", "")
                    tok_text = tok_text.replace("##", "")

                    # Clean whitespace
                    tok_text = tok_text.strip()
                    tok_text = " ".join(tok_text.split())
                    orig_text = "".join(orig_tokens)

                    final_text = self.get_final_text(tok_text, orig_text, do_lower_case)
                    if final_text in seen_predictions:
                        continue

                    seen_predictions[final_text] = True
                else:
                    final_text = ""
                    seen_predictions[final_text] = True

                nbest.append(
                    _NbestPrediction(
                        text=final_text,
                        start_logit=pred.start_logit,
                        end_logit=pred.end_logit))

            # In very rare edge cases we could have no valid predictions. So we
            # just create a nonce prediction in this case to avoid failure.
            if not nbest:
                nbest.append(
                    _NbestPrediction(
                        text="empty", start_logit=0.0, end_logit=0.0))

            total_scores = []
            best_non_null_entry = None
            for entry in nbest:
                total_scores.append(entry.start_logit + entry.end_logit)

            probs = self._compute_softmax(total_scores)

            nbest_json = []
            for (i, entry) in enumerate(nbest):
                output = collections.OrderedDict()
                output["text"] = entry.text
                output["probability"] = probs[i]
                output["start_logit"] = entry.start_logit
                output["end_logit"] = entry.end_logit
                nbest_json.append(output)

            assert len(nbest_json) >= 1

            all_predictions[example.qas_id] = nbest_json[0]["text"]
            all_nbest_json[example.qas_id] = nbest_json

        with open(output_prediction_file, "w") as writer:
            writer.write(json.dumps(all_predictions, indent=4) + "\n")

        with open(output_nbest_file, "w") as writer:
            writer.write(json.dumps(all_nbest_json, indent=4) + "\n")


    def get_final_text(self, pred_text, orig_text, do_lower_case):
        """Project the tokenized prediction back to the original text."""

        # When we created the data, we kept track of the alignment between original
        # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
        # now `orig_text` contains the span of our original text corresponding to the
        # span that we predicted.
        #
        # However, `orig_text` may contain extra characters that we don't want in
        # our prediction.
        #
        # For example, let's say:
        #   pred_text = steve smith
        #   orig_text = Steve Smith's
        #
        # We don't want to return `orig_text` because it contains the extra "'s".
        #
        # We don't want to return `pred_text` because it's already been normalized
        # (the SQuAD eval script also does punctuation stripping/lower casing but
        # our tokenizer does additional normalization like stripping accent
        # characters).
        #
        # What we really want to return is "Steve Smith".
        #
        # Therefore, we have to apply a semi-complicated alignment heruistic between
        # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
        # can fail in certain cases in which case we just return `orig_text`.

        def _strip_spaces(text):
            ns_chars = []
            ns_to_s_map = collections.OrderedDict()
            for (i, c) in enumerate(text):
                if c == " ":
                    continue
                ns_to_s_map[len(ns_chars)] = i
                ns_chars.append(c)
            ns_text = "".join(ns_chars)
            return (ns_text, ns_to_s_map)

        # We first tokenize `orig_text`, strip whitespace from the result
        # and `pred_text`, and check if they are the same length. If they are
        # NOT the same length, the heuristic has failed. If they are the same
        # length, we assume the characters are one-to-one aligned.

        tok_text = " ".join(self.tokenizer.tokenize(orig_text))

        start_position = tok_text.find(pred_text)
        if start_position == -1:
            return orig_text
        end_position = start_position + len(pred_text) - 1

        (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
        (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

        if len(orig_ns_text) != len(tok_ns_text):
            return orig_text

        # We then project the characters in `pred_text` back to `orig_text` using
        # the character-to-character alignment.
        tok_s_to_ns_map = {}
        for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
            tok_s_to_ns_map[tok_index] = i

        orig_start_position = None
        if start_position in tok_s_to_ns_map:
            ns_start_position = tok_s_to_ns_map[start_position]
            if ns_start_position in orig_ns_to_s_map:
                orig_start_position = orig_ns_to_s_map[ns_start_position]

        if orig_start_position is None:
            return orig_text

        orig_end_position = None
        if end_position in tok_s_to_ns_map:
            ns_end_position = tok_s_to_ns_map[end_position]
            if ns_end_position in orig_ns_to_s_map:
                orig_end_position = orig_ns_to_s_map[ns_end_position]

        if orig_end_position is None:
            return orig_text

        output_text = orig_text[orig_start_position:(orig_end_position + 1)]
        return output_text


    def _get_best_indexes(self, logits, n_best_size):
        """Get the n-best logits from a list."""
        index_and_score = sorted(
            enumerate(logits), key=lambda x: x[1], reverse=True)

        best_indexes = []
        for i in range(len(index_and_score)):
            if i >= n_best_size:
                break
            best_indexes.append(index_and_score[i][0])
        return best_indexes


    def _compute_softmax(self, scores):
        """Compute softmax probability over raw logits."""
        if not scores:
            return []

        max_score = None
        for score in scores:
            if max_score is None or score > max_score:
                max_score = score

        exp_scores = []
        total_sum = 0.0
        for score in scores:
            x = math.exp(score - max_score)
            exp_scores.append(x)
            total_sum += x

        probs = []
        for score in exp_scores:
            probs.append(score / total_sum)
        return probs
