#include <vector>
#include <iostream>
#include <time.h>
#include <fstream>
#include <map>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>
#include "senti_cls_dnn.h"
using std::string;
using std::vector;
using std::map;
using std::ifstream;
using std::cout;
using std::cerr;
using std::endl;

namespace senti_cls_dnn {
    
    paddle::platform::CPUPlace SentiClsDnn::_s_place;
    paddle::framework::Scope* SentiClsDnn::_s_scope_ptr = NULL;
    paddle::framework::Executor* SentiClsDnn::_s_executor_ptr = NULL;
    std::unique_ptr<paddle::framework::ProgramDesc> 
        SentiClsDnn::_s_inference_program;
    const string g_unk_word = "<unk>";
    map<std::string, int64_t> SentiClsDnn::_s_vocab_dict; 

    int SentiClsDnn::global_init(const string& config_path) {
        // Define place, executor, scope
        _s_place = paddle::platform::CPUPlace();
        paddle::framework::InitDevices(false);
        _s_executor_ptr = new paddle::framework::Executor(_s_place);
        if (_s_executor_ptr == NULL) {
            cerr << "[ERROR] Allocate executor_ptr failed!" << endl;
            return -1;
        }
        _s_scope_ptr = new paddle::framework::Scope();
        if (_s_scope_ptr == NULL) {
            cerr << "[ERROR] Allocate scope_ptr failed!" << endl;
            return -1;
        }

        // Initialize the inference program
        string model_path = config_path + "/Senta";
        _s_inference_program = paddle::inference::Load(_s_executor_ptr, 
            _s_scope_ptr, model_path);

        // WordSeg Global Init
        int ret = WordSegment::global_init(config_path);
        if (ret != 0) {
            cerr << "[ERROR] WordSegment global init failed!" << endl;
            return -1;
        }
        // Vocab_dict init
        string vocab_dict_path = config_path + "/train.vocab";
        ret = init_vocab_dict(vocab_dict_path);
        if (ret != 0) {
            cerr << "[ERROR] Init vocab_dict failed!" << endl;
            return -1;
        }
        return 0;
    }

    void SentiClsDnn::global_destroy() {
        if (_s_scope_ptr != NULL) {
            delete _s_scope_ptr;
            _s_scope_ptr = NULL;
        } 
        if (_s_executor_ptr != NULL) {
            delete _s_executor_ptr;
            _s_executor_ptr = NULL;
        }
        WordSegment::global_destroy();
    }

    int SentiClsDnn::thread_init(const int thread_id) {
        _copy_program = std::unique_ptr<paddle::framework::ProgramDesc>
            (new paddle::framework::ProgramDesc(*_s_inference_program));
        int ret = _wordseg_tool.thread_init();
        if (ret != 0) {
            cerr << "[ERROR] WordSeg tool thread init faild!" << endl;
            return -1;
        }
        _thread_id = thread_id;
        _feed_holder_name = "feed_" + paddle::string::to_string(_thread_id);
        _fetch_holder_name = "fetch_" + paddle::string::to_string(_thread_id);
        _copy_program->SetFeedHolderName(_feed_holder_name);
        _copy_program->SetFetchHolderName(_fetch_holder_name);
        return 0;
    }

    void SentiClsDnn::thread_destroy() {
        _wordseg_tool.thread_destroy();
        return;
    }

    int SentiClsDnn::init_vocab_dict(const std::string& dict_path) {
        ifstream fin(dict_path.c_str());
        if (fin.is_open() == false) {
            return -1;
        }
        string line;
        int64_t total_count = 0;
        while (getline(fin, line)) {
            vector<string> line_vec;
            boost::split(line_vec, line, boost::is_any_of("\t"));
            if (line_vec.size() != 1) {
                cerr << "[WARNING] Bad line format:\t" << line 
                    << endl;
                continue;
            }
            string word = line_vec[0];
            _s_vocab_dict[word] = total_count;
            total_count += 1;
        }
        cerr << "[NOTICE] Total " << total_count 
            << " words in vocab(include oov)" << endl;
        _s_vocab_dict[g_unk_word] = total_count;
        return 0;
    }

    int SentiClsDnn::trans_word_to_id(const vector<string>& word_list,
        vector<int64_t>& id_list) {
        for (size_t i = 0; i < word_list.size(); i++) {
            const string& cur_word_str = word_list[i];
            if (_s_vocab_dict.find(cur_word_str) != _s_vocab_dict.end()) {
                id_list.push_back(_s_vocab_dict[cur_word_str]);
            }
            else {
                continue;
            }
        }
        if (id_list.size() <= 0) {
            cerr << "[ERROR] Failed to trans word to id!" << endl;
            return -1;
        }
        return 0;
    }

    void SentiClsDnn::normalize_result(SentiClsRes& senti_cls_res) {
        const float neu_threshold = 0.55; // should be (1, 0.5)
        float prob_0 = senti_cls_res._neg_prob;
        float prob_2 = senti_cls_res._pos_prob;
        if (prob_0 > neu_threshold) {
            // if negative probability > threshold, then the classification 
            // label is negative
            senti_cls_res._label = 0;
            senti_cls_res._confidence_val = (prob_0 - neu_threshold)
                / (1 - neu_threshold);
        }
        else if (prob_2 > neu_threshold) {
            // if positive probability > threshold, then the classification
            // label is positive
            senti_cls_res._label = 2;
            senti_cls_res._confidence_val = (prob_2 - neu_threshold)
                / (1 - neu_threshold);
        }
        else {
            // else the classification label is neural
            senti_cls_res._label = 1;
            senti_cls_res._confidence_val = 1.0 - (fabs(prob_2 - 0.5) 
                / (neu_threshold - 0.5));
        }
    }

    int SentiClsDnn::predict(const string& input_str, SentiClsRes& senti_cls_res) {
        // do wordsegment
        vector<string> word_list;
        int ret = _wordseg_tool.word_segment(input_str, word_list);
        if (ret != 0) {
            cerr << "[ERROR] Failed in word_segment!" << endl;
            return -1;
        }
        // trans words to ids
        vector<int64_t> id_list;
        ret = trans_word_to_id(word_list, id_list);
        if (ret != 0) {
            cerr << "[ERROR] Failed in word_to_id!" << endl;
            return -1;
        }

        // get feed_target_name and fetch_target_names
        const std::vector<std::string>& feed_target_names =
            _copy_program->GetFeedTargetNames();
        const std::vector<std::string>& fetch_target_names =
            _copy_program->GetFetchTargetNames();
        
        // set fluid input data
        paddle::framework::LoDTensor input;
        paddle::framework::LoD lod{{0, id_list.size()}};
        input.set_lod(lod);
        int64_t* pdata = input.mutable_data<int64_t>(
            {static_cast<int64_t>(id_list.size()), 1},
            paddle::platform::CPUPlace());
        memcpy(pdata, id_list.data(), input.numel() * sizeof(int64_t));
        
        std::vector<paddle::framework::LoDTensor> feeds;
        feeds.push_back(input);
        std::vector<paddle::framework::LoDTensor> fetchs;

        // define map for feed and fetch targerts
        std::map<std::string, const paddle::framework::LoDTensor*> feed_targets;
        std::map<std::string, paddle::framework::LoDTensor*> fetch_targets;

        // set feed variables
        for (size_t i = 0; i < feed_target_names.size(); ++i) {
            feed_targets[feed_target_names[i]] = &feeds[i];
        }

        // set fetch variables
        fetchs.resize(fetch_target_names.size());
        for (size_t i = 0; i < fetch_target_names.size(); ++i) {
            fetch_targets[fetch_target_names[i]] = &fetchs[i];
        }
        
        // run the fluid inference
        _s_executor_ptr->Run(*_copy_program, _s_scope_ptr, &feed_targets, 
            &fetch_targets, true, true, _feed_holder_name, _fetch_holder_name);

        // get the classification probability
        float* output_ptr = fetchs[0].data<float>();
        senti_cls_res._pos_prob = output_ptr[1];
        senti_cls_res._neg_prob = output_ptr[0];
        // compute the classification label and confidence score
        normalize_result(senti_cls_res);
        return 0;
    }
}

